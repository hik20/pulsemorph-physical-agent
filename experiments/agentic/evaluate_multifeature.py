"""Independent physical feature screening and deterministic eight-trial search.

Features describe observed joints, not perceived expression or hardware safety.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import zipfile
import types
import xml.etree.ElementTree as ET
import numpy as np
from scipy.signal import find_peaks

_spec = importlib.util.spec_from_file_location('_multifeature_audit', Path(__file__).with_name('evaluate_trial.py'))
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)
PARAMETER_NAMES = ['left_gain_a', 'right_gain_a', 'left_gain_b', 'right_gain_b', 'timing_shift_s']


def _fresh_fk_orientations(xml_path, qpos, asset_base):
    """Compute new FK values for audit only; these are not recorded observations."""
    import mujoco
    root = ET.parse(xml_path).getroot()
    for sensor in root.findall('sensor'):
        root.remove(sensor)
    compiler = root.find('compiler')
    for key in ('meshdir', 'texturedir'):
        if compiler is not None and key in compiler.attrib:
            compiler.set(key, str((asset_base / compiler.get(key)).resolve()))
    world = root.find('worldbody')
    if world is not None and not any(g.get('name')=='floor' for g in world.findall('geom')):
        ET.SubElement(world,'geom',name='floor',type='plane',size='0 0 0.05')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    _require(model.nq == 36, 'FK model qpos profile mismatch')
    anchor = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'torso_link')
    _require(anchor >= 0, 'FK torso body missing')
    data = mujoco.MjData(model)
    orientations = []
    for row in qpos:
        data.qpos[:] = row
        mujoco.mj_kinematics(model,data)
        orientations.append(data.xquat[anchor][[1,2,3,0]].copy())
    return np.asarray(orientations)


def _audit_fresh_anchor(directory, audit):
    manifest = audit['manifest']
    _require(manifest.get('control_boundary_forward_count')==200, 'fresh boundary forward count mismatch')
    pins = manifest['preexecution_input_hashes']
    xml_names = [name for name in pins if name.replace('\\','/').endswith('/g1_holo_compat.xml')]
    _require(len(xml_names)==1, 'unique pinned FK body XML required')
    name = xml_names[0]
    snapshot = directory/'source_snapshot'/name
    _require(snapshot.is_file() and _legacy._sha(snapshot)==pins[name], 'pinned FK XML snapshot missing or changed')
    computed = _fresh_fk_orientations(snapshot,audit['arrays']['qpos'][:-1],(_legacy.ROOT/name).parent)
    with np.load(directory/'policy_inputs.npz',allow_pickle=False) as archive:
        observed = np.asarray([archive[f'call_{i:04d}__current_anchor_rot'][0] for i in range(1,201)])
    _require(computed.shape==observed.shape==(200,4) and np.isfinite(computed).all(), 'FK/observed orientation shape mismatch')
    _require(np.allclose(np.linalg.norm(observed,axis=1),1,rtol=0,atol=1e-5), 'observed anchor quaternion norm mismatch')
    from scipy.spatial.transform import Rotation
    error = float(np.max((Rotation.from_quat(computed).inv()*Rotation.from_quat(observed)).magnitude()))
    _require(error<=1e-6, 'same-time observed current_anchor_rot/FK mismatch')
    return dict(control_rows=200,max_rotation_error_rad=error,tolerance_rad=1e-6,
                computation='independently recomputed mj_kinematics; no mj_step; not a recorded observation')


def _audit(directory):
    """Verify the full nested inventory before adapting the legacy audit's view.

    Legacy inventory ignored every file named manifest.json. No evidence is
    changed: an isolated globals dictionary only hides already-verified nested
    manifests from its inventory comparison.
    """
    directory = Path(directory).resolve()
    original = _legacy._json(directory, 'manifest.json')
    hashes = original['artifact_hashes']
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*')
              if p.is_file() and p != directory/'manifest.json'}
    _require(actual == set(hashes), 'full artifact inventory mismatch')
    for name, digest in hashes.items():
        path = (directory/name).resolve()
        _require(directory in path.parents and path.is_file() and _legacy._sha(path)==digest,
                 'artifact hash mismatch: '+name)
    def read_json(path, name):
        value = _legacy._json(path, name)
        if name == 'manifest.json':
            value = {**value, 'artifact_hashes': {key: digest for key,digest in value['artifact_hashes'].items()
                                               if Path(key).name != 'manifest.json'}}
        return value
    namespace = {**_legacy._audit.__globals__, '_json': read_json}
    audit = types.FunctionType(_legacy._audit.__code__, namespace)
    result = audit(directory)
    result['manifest'] = original
    if original.get('schema') == 'pulsemorph.agentic_multifeature_trial.v1':
        result['fresh_anchor_audit'] = _audit_fresh_anchor(directory,result)
    return result


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _vector(value, name):
    value = np.asarray(value, dtype=float)
    _require(value.shape == (5,) and np.isfinite(value).all(), 'invalid '+name)
    return value


def _configuration(protocol, goal):
    target = _vector(goal['feature_targets'], 'feature targets')
    tolerance = _vector(protocol['feature_tolerances'], 'feature tolerances')
    bounds = np.asarray(protocol['bounds'], dtype=float)
    _require(protocol['parameter_names'] == PARAMETER_NAMES, 'parameter order mismatch')
    _require(bounds.shape == (5,2) and np.isfinite(bounds).all() and np.all(bounds[:,0] < bounds[:,1]), 'invalid parameter bounds')
    _require(np.all(tolerance > 0), 'positive feature tolerances required')
    return target, tolerance, bounds


def accent_metrics(velocity, times, window, prominence=.15):
    """Centered 100ms RMS joint-speed proxy, with explicit missing-event state."""
    velocity, times = np.asarray(velocity), np.asarray(times)
    _require(velocity.shape == (len(times),29) and np.isfinite(velocity).all(), 'invalid accent velocities')
    speed = np.sqrt(np.mean(velocity**2, axis=1))
    dt = float(times[1]-times[0])
    width = max(1, round(.1/dt))
    if width % 2 == 0:
        width += 1
    smooth = np.convolve(np.pad(speed,(width//2,width//2),mode='edge'), np.ones(width)/width, mode='valid')
    peaks, properties = find_peaks(smooth, prominence=prominence, distance=max(1,round(.35/dt)))
    inside = (times[peaks] >= window[0]) & (times[peaks] <= window[1])
    candidates = peaks[inside]
    strongest = int(candidates[np.argmax(smooth[candidates])]) if len(candidates) else None
    return dict(available=strongest is not None, peak_time_s=None if strongest is None else float(times[strongest]),
                peak_rms_speed_rad_s=None if strongest is None else float(smooth[strongest]),
                event_times_s=times[peaks].tolist(), event_count=len(peaks), window_event_count=len(candidates),
                window_event_coverage=int(len(candidates)>0), smoothing_samples=width, prominence_rad_s=prominence,
                rms_speed_rad_s=speed.tolist(), smoothed_rms_speed_rad_s=smooth.tolist())


def evaluate_multifeature(directory, baseline_directory, protocol, goal):
    result = dict(schema='pulsemorph.agentic_multifeature_evaluation.v1', integrity_passed=False,
                  integrity_errors=[], physical_gate_passed=False, intent_passed=False, features=None,
                  feature_errors=None, per_dimension_pass=[False]*5,
                  utility=None, limitation='observed joint proxies; not aesthetics, comprehensive balance or hardware safety')
    try:
        target, tolerance, _ = _configuration(protocol, goal)
        trial, baseline = _audit(directory), _audit(baseline_directory)
        q,b = trial['arrays']['qpos'],baseline['arrays']['qpos']
        _require(trial['names']==baseline['names'] and np.array_equal(q[0],b[0]) and
                 np.array_equal(trial['arrays']['qvel'][0],baseline['arrays']['qvel'][0]), 'baseline initial state/profile mismatch')
        for key in ('onnx_sha256','motion_sha256','physics_contract','versions','preexecution_input_hashes'):
            _require(trial['manifest'][key] == baseline['manifest'][key], 'baseline controller/source mismatch: '+key)
        groups=[]
        for side in ('left','right'):
            group=[i for i,name in enumerate(trial['names']) if name.startswith(side+'_') and any(part in name for part in ('shoulder_', 'elbow_', 'wrist_'))]
            _require(len(group)==7, 'expected seven '+side+' arm joints')
            groups.append(group)
        _require(set(groups[0]).isdisjoint(groups[1]), 'arm groups overlap')
        times=trial['arrays']['state_times']; dimensions=[]; features=[]
        windows=protocol['feature_windows_s']
        _require(len(windows)==2, 'two feature windows required')
        for label,(start,end) in zip(('a','b'),windows):
            _require(0 <= start < end <= 4, 'invalid feature window')
            mask=(times>=start-1e-10)&(times<=end+1e-10)
            _require(mask.any(), 'empty feature window')
            for side,group in zip(('left','right'),groups):
                center=b[0,7:][group]
                actual=float(np.sqrt(np.mean((q[mask,7:][:,group]-center)**2)))
                denominator=float(np.sqrt(np.mean((b[mask,7:][:,group]-center)**2)))
                reference=float(np.sqrt(np.mean((trial['cache']['committed__dof_pos'][mask][:,group]-center)**2)))
                _require(denominator>1e-10, 'zero baseline amplitude')
                features.append(actual/denominator)
                dimensions.append(dict(name=side+'_ratio_'+label, actual_rms_rad=actual, reference_rms_rad=reference,
                                       baseline_rms_rad=denominator,joint_names=[trial['names'][i] for i in group]))
        window=protocol['accent_window_s']
        _require(len(window)==2 and 0<=window[0]<window[1]<=4, 'invalid accent window')
        prominence=float(protocol.get('accent_minimum_prominence',.15))
        _require(np.isfinite(prominence) and prominence>0, 'positive accent prominence required')
        accent=accent_metrics(trial['arrays']['qvel'][:,6:],times,window,prominence)
        reference_velocity=trial['cache']['committed__dof_vel']
        accent['reference']=accent_metrics(reference_velocity,times,window,prominence)
        height=float(np.min(trial['sub'][:,2])); rmse=trial['mean_joint_rmse_rad']; gate=protocol['physical_gate']
        result.update(integrity_passed=True, min_root_height_m=height,mean_joint_rmse_rad=rmse,
                      physical_gate_passed=height>=gate['min_root_height_m'] and rmse<=gate['max_mean_joint_rmse_rad'],
                      dimensions=dimensions,accent=accent,feature_tolerances=tolerance.tolist())
        result['audit_implementation_sha256'] = _legacy._sha(Path(__file__).with_name('evaluate_trial.py'))
        result['fresh_anchor_audit'] = trial.get('fresh_anchor_audit')
        features.append(accent['peak_time_s'])
        result['features']=features
        if not accent['available']:
            result['feature_errors']=[float(features[i]-target[i]) for i in range(4)]+[None]
            result['per_dimension_pass']=[bool(abs(features[i]-target[i])<=tolerance[i]) for i in range(4)]+[False]
            result['intent_unavailable_reason']='no eligible observed accent peak'
            return result
        errors=np.asarray(features)-target; normalized=np.abs(errors)/tolerance
        result.update(feature_errors=errors.tolist(), per_dimension_pass=(normalized<=1).tolist(),
                      intent_passed=bool(np.all(normalized<=1)),utility=float(np.max(normalized)),
                      utility_l2=float(np.sum(normalized**2)))
    except (OSError,ValueError,KeyError,TypeError,IndexError,ImportError,ET.ParseError,zipfile.BadZipFile) as error:
        result.update(integrity_passed=False,physical_gate_passed=False,intent_passed=False)
        result['integrity_errors'].append(str(error))
    return result


def next_jacobian_candidate(history, protocol, goal):
    """Anchor + five probes + damped Jacobian + one Broyden update; no hidden trials.

    Each history row has ``parameters`` and ``features`` (or ``evaluation.features``).
    Failed/missing observations reject optimization rather than inventing feedback.
    """
    target,tolerance,bounds=_configuration(protocol,goal)
    n=len(history)
    _require(n<8, 'eight-trial budget exhausted')
    anchor=_vector(protocol.get('jacobian_anchor',[1,1,1,1,0]),'anchor')
    _require(np.all((anchor>=bounds[:,0])&(anchor<=bounds[:,1])), 'anchor outside bounds')
    if n==0:
        return anchor.tolist()
    if n<=5:
        candidate=anchor.copy(); candidate[n-1]+= .10 if n==5 else .15
        return np.clip(candidate,bounds[:,0],bounds[:,1]).tolist()
    x=np.array([_vector(row['parameters'],'history parameters') for row in history])
    y=np.array([_vector(row.get('features',row.get('evaluation',{}).get('features')),'history features') for row in history])
    dx=x[1:6]-x[0]; dy=y[1:6]-y[0]
    _require(np.linalg.matrix_rank(dx)==5,'singular probe design')
    jacobian=np.linalg.solve(dx,dy).T
    if n==7:
        step=x[6]-x[0]
        if step@step>1e-16:
            jacobian += np.outer(y[6]-y[0]-jacobian@step,step)/(step@step)
    # Choose an actually observed iterate, with identical normalized objective as selector.
    norms=np.abs((y-target)/tolerance)
    feasible=[i for i,row in enumerate(history) if row.get('physical_gate_passed',row.get('evaluation',{}).get('physical_gate_passed',True))
              and row.get('integrity_passed',row.get('evaluation',{}).get('integrity_passed',True))]
    _require(bool(feasible),'no physically valid observed repair center')
    best=min(feasible,key=lambda i:(float(np.max(norms[i])),float(np.sum(norms[i]**2)),i))
    weighted=jacobian/tolerance[:,None]; residual=(target-y[best])/tolerance
    damping=float(protocol.get('jacobian_damping',1e-4))
    _require(np.isfinite(damping) and damping>0,'positive Jacobian damping required')
    step=np.linalg.solve(weighted.T@weighted+damping*np.eye(5),weighted.T@residual)
    trust=_vector(protocol.get('jacobian_trust_radius',[.65,.65,.65,.65,.3]),'trust radius')
    _require(np.all(trust>0),'positive trust radius required')
    step/=max(1.,float(np.max(np.abs(step)/trust)))
    return np.clip(x[best]+step,bounds[:,0],bounds[:,1]).tolist()
