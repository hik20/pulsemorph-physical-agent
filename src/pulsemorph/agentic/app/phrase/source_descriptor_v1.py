"""Pinned source-knot activity descriptors, distinct from physical response."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
from ...phrase_program import _arm_indices
from ..exports import JOINT_NAMES
from .assets import BODY_SHA,CONTROLLER_SHA,MANIFEST_SHA256,REFERENCE_HASHES

SCHEMA='pulsemorph.phrase_source_descriptor.v1'
def _sha(data):return hashlib.sha256(data).hexdigest()
def _canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def _phase(reference,source_id,global_window):
    reference.validate();meta=reference.metadata
    if tuple(meta['body']['joint_names'])!=JOINT_NAMES or meta['body']['profile_id']!='g1' or meta['body']['profile_sha256']!=BODY_SHA or meta['controller']['profile_sha256']!=CONTROLLER_SHA:raise ValueError('Descriptor requires admitted G1 source order/profile')
    t=np.asarray(reference.time_offsets_s);q=np.asarray(reference.joint_position_rad)
    if t.ndim!=1 or not 162<=len(t)<=250 or q.shape!=(len(t),29) or t.dtype.kind!='f' or q.dtype.kind!='f' or not np.isfinite(t).all() or not np.isfinite(q).all() or not np.allclose(t,np.arange(len(t))*.02,rtol=0,atol=1e-10):raise ValueError('Finite 20ms source knots and G1 joint positions required')
    keep=(t>=1.2-1e-10)&(t<=3.2+1e-10)
    if int(keep.sum())!=101:raise ValueError('Exact 101-knot source measurement window required')
    arms={}
    for side,indices in zip(('left','right'),_arm_indices(JOINT_NAMES)):
        values=q[keep][:,indices].copy();offset=values-q[0,indices];mean=values.mean(axis=0);centered=values-mean;peak_to_peak=np.ptp(values,axis=0)
        numbers={'q0_displacement_rms_rad':float(np.sqrt(np.mean(offset**2))),'centered_temporal_rms_rad':float(np.sqrt(np.mean(centered**2))),'mean_pose_offset_rms_rad':float(np.sqrt(np.mean((mean-q[0,indices])**2))),'max_joint_peak_to_peak_rad':float(peak_to_peak.max())}
        if any(not np.isfinite(v) or v<0 or v>100 for v in numbers.values()) or not np.isfinite(peak_to_peak).all() or np.any(peak_to_peak>100):raise ValueError('Descriptor numeric range invalid')
        arms[side]={'joint_names':[JOINT_NAMES[i] for i in indices],**numbers,'per_joint_peak_to_peak_rad':[float(v) for v in peak_to_peak]}
    lineage=meta.get('provenance',{}).get('sources')
    if not isinstance(lineage,(list,tuple)) or not 1<=len(lineage)<=16:raise ValueError('Source lineage missing or oversized')
    hashes=[]
    for record in lineage:
        digest=record.get('sha256')
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Invalid source lineage digest')
        hashes.append(digest)
    return {'phase_index':0 if source_id=='wave' else 1,'source_id':source_id,'global_window_s':list(global_window),'source_window_s':[1.2,3.2],'sample_count':101,'source_q0_time_s':0.,'source_hashes':{'manifest_sha256':REFERENCE_HASHES[source_id]['manifest.json'],'arrays_sha256':REFERENCE_HASHES[source_id]['arrays.npz'],'recorded_input_sha256':hashes},'arms':arms}

def describe_sources(assets):
    """No cache/files supplied by callers; verify admitted assets before/after."""
    module_path=Path(__file__);module_hash=_sha(module_path.read_bytes());assets.verify();before=copy.deepcopy(assets.manifest)
    if before.get('schema')!='pulsemorph.private_phrase_assets.v1':raise ValueError('Unknown source asset profile')
    for source,files in REFERENCE_HASHES.items():
        for name,digest in files.items():
            if before.get('files',{}).get(source+'/'+name)!=digest:raise ValueError('Descriptor source hash not admitted')
    refs=assets.references()
    if type(refs) is not dict or set(refs)!={'wave','high5'}:raise ValueError('Exact source pair required')
    phases=[_phase(refs['wave'],'wave',[1.2,3.2]),_phase(refs['high5'],'high5',[5.2,7.2])]
    result={'schema':SCHEMA,'asset_manifest_sha256':MANIFEST_SHA256,'source_module_sha256':module_hash,'body_profile_id':'g1','body_profile_sha256':BODY_SHA,'controller_profile_sha256':CONTROLLER_SHA,'sampling':{'basis':'source-reference stored 20ms knots, not continuous extrema or physical states','dt_s':.02,'window_mask_tolerance_s':1e-10,'joint_count_per_arm':7,'joint_units':'radians','same_G1_joint_order':True},'definitions':{'q0_displacement_rms_rad':'sqrt(mean over 101 samples and 7 joints of (q[t,j]-q[0,j])^2)','centered_temporal_rms_rad':'sqrt(mean over 101 samples and 7 joints of (q[t,j]-mean_time(q[:,j]))^2); per-joint temporal mean inside the selected window','mean_pose_offset_rms_rad':'sqrt(mean over 7 joints of (mean_time(q[:,j])-q[0,j])^2); per-joint temporal mean inside the selected window','per_joint_peak_to_peak_rad':'max_time(q[:,j])-min_time(q[:,j]) inside the selected window, aligned with joint_names','max_joint_peak_to_peak_rad':'maximum of the seven per_joint_peak_to_peak_rad values'},'phases':phases,'qualification':'Pinned source-reference activity only, not measured physical response, controllability, goal feasibility, style or beat quality. Held pose offset and temporal variation are different. No inactive flag, goal disabling or target change. No model/ONNX/FK/physics/compilation.'}
    payload=_canonical(result)
    if len(payload)>32*1024:raise ValueError('Descriptor exceeds compact DTO bound')
    assets.verify()
    if assets.manifest!=before or _sha(module_path.read_bytes())!=module_hash:raise ValueError('Descriptor source changed during measurement')
    return json.loads(payload)
