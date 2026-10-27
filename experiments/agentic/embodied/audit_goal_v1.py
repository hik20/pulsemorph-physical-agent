"""Optional explicitly capped saved-state wrist FK trajectory, never task admission.

This descriptive 20ms analysis cannot retroactively change the preregistered
endpoint task result. No NN/controller/physics stepping or robot rendering.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from pulsemorph.agentic.embodied.tasks_v1 import validate_task

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def inventory(path): return {p.relative_to(path).as_posix():sha(p) for p in path.rglob('*') if p.is_file()}
def require(value, message):
    if not value: raise ValueError(message)

def summarize_trajectory(times, positions, target, tolerance, start_s):
    times=np.asarray(times);positions=np.asarray(positions);target=np.asarray(target)
    require(times.ndim==1 and positions.shape==(len(times),3) and target.shape==(3,), 'Trajectory shapes')
    require(all(np.isfinite(v).all() for v in (times,positions,target)) and np.all(np.diff(times)>0), 'Finite monotone trajectory')
    require(type(tolerance) in (int,float) and np.isfinite(tolerance) and tolerance>0 and np.isfinite(start_s), 'Fixed finite task bounds')
    eligible=times>=start_s-1e-10
    require(eligible.any(), 'Task has no sampled state')
    distances=np.linalg.norm(positions-target,axis=1);indices=np.flatnonzero(eligible)
    inside=indices[distances[indices]<=tolerance]
    best=int(indices[np.argmin(distances[indices])]);first=int(indices[0]);last=int(indices[-1])
    return distances,{'sample_count':len(indices),'start_distance_m':float(distances[first]),
        'minimum_distance_m':float(distances[best]),'minimum_time_s':float(times[best]),
        'final_distance_m':float(distances[last]),'first_in_tolerance_time_s':None if not len(inside) else float(times[inside[0]]),
        'in_tolerance_sample_count':len(inside),'final_sample_inside_tolerance':bool(distances[last]<=tolerance),
        'keyframe_indices':list(dict.fromkeys([first,best,last])),
        'qualification':'Descriptive sampled trajectory only. Transient proximity is not preregistered endpoint task_verified or continuous hold.'}

def audit_goal(directory, task_path, output, *, max_fk, task_start_s=0):
    directory=Path(directory).resolve();task_path=Path(task_path).resolve();output=Path(output).absolute()
    require(not output.exists() and not output.resolve().is_relative_to(directory), 'New output outside raw episode required')
    for p in (directory,task_path,output,*output.parents):
        require(not p.is_symlink() and not (p.exists() and getattr(p.lstat(),'st_file_attributes',0)&1024), 'Linked paths forbidden')
    before=inventory(directory);task_sha=sha(task_path);own_sha=sha(__file__)
    frozen=json.loads(task_path.read_bytes());task=validate_task(frozen['task'] if 'task' in frozen else frozen)
    backend=json.loads((directory/'backend.json').read_bytes());episode=json.loads((directory/'episode.json').read_bytes())
    require(backend.get('schema')=='pulsemorph.embodied_backend.v1' and episode.get('schema')=='pulsemorph.embodied_episode.v1','Original episode/backend schema')
    require(type(backend.get('source_pins')) is dict and bool(backend['source_pins']),'Nonempty execution source pins')
    require(backend['initialization_count']==episode['initialization_count']==backend['mjdata_instances']==backend['onnx_sessions']==1,'Persistent initialization/session counts')
    for path,digest in backend['source_pins'].items(): require(sha(path)==digest,'Execution source pin changed')
    q=np.load(directory/'qpos.npy',allow_pickle=False);dq=np.load(directory/'qvel.npy',allow_pickle=False);times=np.load(directory/'state_times.npy',allow_pickle=False)
    require(q.shape==(len(times),36) and dq.shape==(len(times),35) and len(times)==backend['online_calls']+1,'Actual saved state counts')
    require(all(np.isfinite(a).all() for a in (q,dq,times)) and np.allclose(times,np.arange(len(times))*.02,rtol=0,atol=1e-9),'Actual finite clock')
    require(type(max_fk) is int and max_fk>=len(times) and max_fk<=501,'Explicit finite FK cap required for every saved state')
    # Recreate the pinned source-body patch independently, rather than trust the
    # saved display XML as a new source of truth. These imports create no actor.
    sys.path.insert(0,str(ROOT/'research/vendor/ProtoMotions'))
    import mujoco
    from deployment import test_tracker_mujoco as upstream
    from pulsemorph.motion.g1 import profile_from_yaml
    from pulsemorph.agentic.app.phrase.assets import MODEL
    profile=profile_from_yaml(ROOT/(MODEL[:-5]+'.yaml'))
    doc=ET.fromstring(upstream._patch_mjcf_xml(Path(profile['body_path'])))
    compiler=doc.find('compiler')
    for name in ('meshdir','texturedir'):
        if compiler is not None and compiler.get(name):compiler.set(name,str((Path(profile['body_path']).parent/compiler.get(name)).resolve()))
    patched=ET.tostring(doc,encoding='unicode')
    require((directory/'render_model.xml').read_text(encoding='utf8')==patched,'Actual pinned patched XML differs')
    model=mujoco.MjModel.from_xml_string(patched);data=mujoco.MjData(model)
    wrist=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,task['body']);torso=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'torso_link')
    require(wrist>=0 and torso>=0,'Actual named wrist/torso bodies required')
    positions=[];fk=0
    with np.load(directory/'policy_inputs.npz',allow_pickle=False) as feeds:
        for i in range(len(times)):
            data.qpos[:]=q[i];data.qvel[:]=dq[i];data.time=float(times[i]);mujoco.mj_forward(model,data);fk+=1
            positions.append(data.xpos[wrist].copy())
            if i<len(times)-1:
                require(np.allclose(data.xquat[torso][[1,2,3,0]][None],feeds[f'call_{i+1:04d}__current_anchor_rot'],rtol=0,atol=1e-6),'Actual current torso input FK differs')
    positions=np.asarray(positions);distances,summary=summarize_trajectory(times,positions,task['target_world_m'],task['tolerance_m'],task_start_s)
    final_pose=episode['final_observation']['body_poses'][task['body']]['position_world_m']
    require(np.allclose(positions[-1],final_pose,rtol=0,atol=1e-9),'Final observed wrist FK differs')
    require(inventory(directory)==before and sha(task_path)==task_sha and sha(__file__)==own_sha,'Raw/task/auditor changed')
    for path,digest in backend['source_pins'].items(): require(sha(path)==digest,'Execution source pin changed during FK')
    output.mkdir(parents=True,exist_ok=False)
    with (output/'wrist_trajectory.csv').open('x',newline='',encoding='utf8') as stream:
        writer=csv.writer(stream);writer.writerow(['control_state_index','sim_time_s','wrist_world_x_m','wrist_world_y_m','wrist_world_z_m','target_distance_m','task_active'])
        for i,(t,p,d) in enumerate(zip(times,positions,distances)):writer.writerow([i,float(t),*map(float,p),float(d),bool(t>=task_start_s-1e-10)])
    keys=summary['keyframe_indices'];np.savez(output/'keyframe_states.npz',indices=np.asarray(keys),times=times[keys],qpos=q[keys],qvel=dq[keys],wrist_world_m=positions[keys])
    result={'schema':'pulsemorph.embodied_goal_trajectory_audit.v1','input_directory':str(directory),'task_sha256':task_sha,'task':task,'task_start_s':task_start_s,'summary':summary,'fresh_FK':fk,'max_fk_authorized':max_fk,'current_torso_feed_FK_rows':len(times)-1,'new_NN':0,'new_physics_steps':0,'robot_render_frames':0,'raw_inventory':before,'auditor_sha256':own_sha,'endpoint_task_verified_unchanged':True,'limits':['20ms sampled positions, no between-state occupancy claim.','No new task success is admitted from a transient within tolerance.','Saved keyframes are numeric states, not newly rendered robot images.','Scalar substep records do not independently reconstruct full substep body trajectories.']}
    (output/'receipt.json').write_text(json.dumps(result,indent=2),encoding='utf8');return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('task',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--max-fk',type=int,required=True);parser.add_argument('--task-start-s',type=float,default=0);a=parser.parse_args()
    print(json.dumps(audit_goal(a.directory,a.task,a.output,max_fk=a.max_fk,task_start_s=a.task_start_s)['summary']))
