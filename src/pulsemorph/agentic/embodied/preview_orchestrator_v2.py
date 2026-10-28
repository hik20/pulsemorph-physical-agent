"""Version 2: ordered actual-control samples reach the unchanged task verifier.

No search, inverse controller or automatic best-candidate selection lives here.
Fork backend owns closure validation and byte-exact preview/commit replay proof.
"""
import copy,json,math,re,threading,time
from pathlib import Path
from jsonschema import Draft202012Validator
from .orchestrator_v1 import CACHE,bounded,canonical,digest,write_new,observation,response_schema
from .preview_provider_v1 import plan_preview_decision

def preview_schema(capabilities,eligible_ids):
    schema=response_schema(capabilities)
    schema['properties']['decision']['enum']=['preview','commit','finish','stop']
    schema['properties']['preview_id']={'type':['string','null'],'enum':[None,*eligible_ids]}
    schema['required'].append('preview_id')
    return schema

def compact_assessment(value):
    return {k:copy.deepcopy(value[k]) for k in ('done','failed','reason','distance_m','signed_error_world_m','inside_tolerance','sampled_hold_s','physical_gate_passed') if k in value}

def compact_observation(value,*,joints=False):
    result={k:copy.deepcopy(value[k]) for k in ('session_id','revision','sim_time_s','physical_flags') if k in value}
    result['observation_sha256']=digest(value)
    result['wrist_body_poses']={k:copy.deepcopy(v) for k,v in value.get('body_poses',{}).items() if k in ('left_rubber_hand','right_rubber_hand')}
    if joints:
        for key in ('qpos','qvel'):result[key]=copy.deepcopy(value[key])
    return result

def projected_context(fixed,last,assessment,history,previews,remaining):
    """Deterministic size-bounded view; never chooses or rewrites an action."""
    candidates=[]
    for row in previews.values():
        candidates.append({'preview_id':row['preview_id'],'parent_observation_sha256':row['parent_observation_sha256'],'action':copy.deepcopy(row['action']),'eligible':row['eligible'],'hypothetical_assessment':compact_assessment(row['hypothetical_assessment']),'hypothetical_observation':compact_observation(row['hypothetical_observation'],joints=True),'qualification':row['qualification']})
    past=[]
    for row in history:
        item={'kind':row['kind']}
        if 'action' in row:item['action']=copy.deepcopy(row['action'])
        if row['kind']=='hypothetical_preview':item.update(preview_id=row['preview']['preview_id'],observation_sha256=digest(row['preview']['hypothetical_observation']),assessment=compact_assessment(row['preview']['hypothetical_assessment']))
        elif row['kind']=='actual_committed':item.update(preview_id=row['preview_id'],actual_after=compact_observation(row['transition']['observation']),actual_assessment=compact_assessment(row['actual_assessment']))
        elif 'report' in row:
            report=row['report'];item['report']={k:copy.deepcopy(report[k]) for k in ('schema','preview_id','parent_observation_sha256','parent_revision','parent_unchanged','error_type','hypothetical_only','category','message','violations','physics_steps') if k in report}
            if 'suffix' in report:item['sandbox_suffix']={k:copy.deepcopy(report['suffix'][k]) for k in ('new_physics_steps','new_online_calls','new_forward_calls','sha256') if k in report['suffix']}
        past.append(item)
    return {'schema':'pulsemorph.embodied_preview_decision.v1','fixed':copy.deepcopy(fixed),'observation':copy.deepcopy(last),'observation_sha256':digest(last),'actual_assessment':compact_assessment(assessment),'history':past,'previews':candidates,'remaining':remaining,'context_projection':{'mode':'current_actual_full; past transitions and candidates projected','raw_observations_and_receipts_preserved':True,'selection_or_action_numbers_changed':False,'past_omitted':'Full past feed/history/observation arrays retained on disk; hashes and wrist/task summaries supplied here.'}}

class PreviewOrchestrator:
    def __init__(self,provider,*,task_verifier,max_model_calls=8,max_previews=4,max_sandbox_steps=8000,max_actual_steps=4000,wall_seconds=1800,action_rejection_type=None,preview_physical_failure_type=None,allow_fixture=False):
        for v,maximum in [(max_model_calls,16),(max_previews,8),(max_sandbox_steps,24000),(max_actual_steps,12000)]:
            if type(v) is not int or not 1<=v<=maximum:raise ValueError('Finite independent preview budget required')
        if type(wall_seconds) not in (int,float) or not 1<=wall_seconds<=3600 or not callable(task_verifier):raise ValueError('Bounded verifier/wall contract required')
        if type(allow_fixture) is not bool or any(t is not None and (not isinstance(t,type) or not issubclass(t,Exception)) for t in (action_rejection_type,preview_physical_failure_type)):raise ValueError('Explicit fixture flag and typed local rejection required')
        self.provider=provider;self.verifier=task_verifier;self.max_model_calls=max_model_calls;self.max_previews=max_previews;self.max_sandbox_steps=max_sandbox_steps;self.max_actual_steps=max_actual_steps;self.wall_seconds=wall_seconds;self.rejection_type=action_rejection_type;self.physical_failure_type=preview_physical_failure_type;self.allow_fixture=allow_fixture;self.lock=threading.Lock()
    def run(self,intent,task,fork,output,*,cancel_event=None):
        if type(intent) is not str or not 3<=len(intent.strip())<=3000 or type(task) is not dict:raise ValueError('Fixed intent/task required')
        output=Path(output).absolute()
        if re.fullmatch('[A-Za-z0-9_-]{1,80}',output.name) is None or output.resolve()==CACHE.resolve() or not output.resolve().is_relative_to(CACHE.resolve()):raise ValueError('Exclusive private run path required')
        for p in (output,*output.parents):
            if p.is_symlink() or p.exists() and getattr(p.lstat(),'st_file_attributes',0)&1024:raise ValueError('Linked run path forbidden')
        if not self.lock.acquire(blocking=False):raise RuntimeError('Preview orchestrator busy')
        try:return self._run(intent.strip(),bounded(task),fork,output,cancel_event or threading.Event())
        finally:self.lock.release()
    def _run(self,intent,task,fork,out,cancel):
        out.mkdir(parents=True,exist_ok=False);start=time.monotonic();caps=bounded(fork.capabilities());step_s=caps.get('physics_step_s')
        if type(step_s) not in (int,float) or not 0<step_s<=.02:raise ValueError('Backend step duration required')
        fixed={'intent':intent,'task':task,'capabilities':caps,'argument_transport':'Unique named target LIST explicitly mapped to backend target dictionary.','budgets':{'model_calls':self.max_model_calls,'previews':self.max_previews,'sandbox_steps':self.max_sandbox_steps,'actual_steps':self.max_actual_steps,'wall_seconds':self.wall_seconds}}
        write_new(out/'fixed.json',fixed);fixed_hash=digest(fixed);history=[];previews={};sandbox_steps=actual_steps=preview_attempts=commit_attempts=0;pending=None;status='stopped';reason='model_budget';last=None;state=None;assessment=None
        def check():
            if cancel.is_set():raise RuntimeError('CANCELLED')
            if time.monotonic()-start>=self.wall_seconds:raise RuntimeError('WALL_BUDGET')
            if digest(json.loads((out/'fixed.json').read_bytes()))!=fixed_hash or bounded(fork.capabilities())!=caps:raise RuntimeError('FIXED_CONTRACT_CHANGED')
        def assess(obs,prior):
            result=bounded(self.verifier(copy.deepcopy(task),copy.deepcopy(obs),copy.deepcopy(prior)))
            if type(result) is not dict or type(result.get('done')) is not bool or type(result.get('reason')) is not str or result['done'] and result.get('failed') is True:raise ValueError('Independent task assessment required')
            return result
        def bounded_samples(value):
            if type(value) is not list or not 1<=len(value)<=50:
                raise RuntimeError('BOUNDED_SAMPLE_JOURNAL_REQUIRED')
            return [bounded(sample) for sample in value]
        def assess_samples(samples,previous,final,prior,receipt):
            count=receipt.get('sampled_observations_count')
            if type(samples) is not list or type(count) is not int or count!=len(samples) or not count or digest(samples)!=receipt.get('sampled_observations_sha256'):
                raise RuntimeError('SAMPLE_JOURNAL_PROOF_MISMATCH')
            period=caps.get('sampled_observation_period_s')
            if period!=.02 or count!=round((final['sim_time_s']-previous['sim_time_s'])/period):
                raise RuntimeError('SAMPLE_JOURNAL_COVERAGE_MISMATCH')
            current=previous;result=None;fold=copy.deepcopy(prior)
            for sample_index,sample in enumerate(samples):
                sample=observation(sample,last['session_id'])
                expected_revision=previous['revision']+(1 if sample_index==len(samples)-1 else 0)
                if sample['revision']!=expected_revision or sample.get('control_index')!=current.get('control_index')+1 or abs(sample['sim_time_s']-current['sim_time_s']-period)>1e-9:
                    raise RuntimeError('SAMPLE_JOURNAL_ORDER_OR_GAP_MISMATCH')
                result=assess(sample,fold);fold=copy.deepcopy(result.get('state'));current=sample
            if current!=final:raise RuntimeError('SAMPLE_JOURNAL_ENDPOINT_MISMATCH')
            return result
        def parent_unchanged():
            current=observation(fork.observe(),last['session_id'])
            if current!=last:raise RuntimeError('ACTUAL_PARENT_CHANGED')
        try:
            check();last=observation(fork.observe());write_new(out/'actual_observation_00.json',last);assessment=assess(last,state);state=copy.deepcopy(assessment.get('state'));write_new(out/'actual_assessment_00.json',assessment)
            for index in range(self.max_model_calls):
                check()
                if assessment.get('failed') is True:status='failed';reason='actual_physical_or_task_failure';break
                if assessment['done']:status='completed';reason='actual_task_verified';break
                eligible=[pid for pid,row in previews.items() if row['eligible'] and row['parent_observation_sha256']==digest(last)]
                request=projected_context(fixed,last,assessment,history,previews,{'model_calls':self.max_model_calls-index,'previews':self.max_previews-preview_attempts,'sandbox_steps':self.max_sandbox_steps-sandbox_steps,'actual_steps':self.max_actual_steps-actual_steps})
                result=plan_preview_decision(self.provider,request,preview_schema(caps,eligible),out/'model'/('call_%02d'%index),cancel_event=cancel,timeout=min(120,max(.1,self.wall_seconds-(time.monotonic()-start))))
                check();parent_unchanged()
                if result.get('observation_sha256')!=digest(last) or result['receipt'].get('live') is not True and not self.allow_fixture:raise RuntimeError('STALE_OR_NONLIVE_PROPOSAL')
                p=result['proposal'];Draft202012Validator(preview_schema(caps,eligible)).validate(p);decision=p['decision']
                if decision=='preview':
                    if p['action'] is None or p['preview_id'] is not None:raise ValueError('Preview decision mismatch')
                    if preview_attempts>=self.max_previews:reason='preview_budget';break
                    args=copy.deepcopy(p['action']['arguments']);entries=args['joint_targets_rad'];names=[v['joint_name'] for v in entries]
                    if len(names)!=len(set(names)):raise ValueError('Duplicate joint target')
                    args['joint_targets_rad']={v['joint_name']:v['position_rad'] for v in entries};duration=args.get('duration_s')
                    if type(duration) not in (int,float) or not math.isfinite(duration) or duration<=0:raise ValueError('Finite duration required')
                    expected=round(duration/step_s)
                    if expected<1 or abs(expected*step_s-duration)>1e-9 or sandbox_steps+expected>self.max_sandbox_steps:reason='sandbox_budget';break
                    action={'action_id':out.name+'__preview_%02d'%preview_attempts,'expected_revision':last['revision'],'observation_sha256':digest(last),'tool':p['action']['tool'],'arguments':args}
                    d=out/('preview_%02d'%preview_attempts);d.mkdir();write_new(d/'pending.json',{'action':action,'expected_steps':expected});preview_attempts+=1;pending='preview'
                    try:receipt=bounded(fork.preview(copy.deepcopy(action)))
                    except Exception as error:
                        if self.physical_failure_type is not None and isinstance(error,self.physical_failure_type):
                            parent_unchanged();report=bounded(error.report);suffix=report.get('suffix',{});used=suffix.get('new_physics_steps')
                            if (report.get('parent_unchanged') is not True or report.get('parent_observation_sha256')!=digest(last) or type(report.get('parent_revision')) is not int or report['parent_revision']!=last['revision'] or report.get('hypothetical_only') is not True or type(used) is not int or not 0<=used<=expected or sandbox_steps+used>self.max_sandbox_steps):raise RuntimeError('SANDBOX_FAILURE_PROOF_MISMATCH')
                            sandbox_steps+=used;write_new(d/'physical_failure.json',report);history.append({'kind':'hypothetical_physical_failure','action':action,'report':report,'actual_parent_unchanged':True});pending=None;continue
                        if self.rejection_type is None or not isinstance(error,self.rejection_type):raise
                        parent_unchanged();report=bounded(error.report)
                        if type(report) is not dict or type(report.get('physics_steps')) is not int or report['physics_steps']!=0 or type(report.get('revision_unchanged')) is not int or report['revision_unchanged']!=last['revision'] or report.get('observation_sha256')!=digest(last):raise RuntimeError('REJECTION_PROOF_MISMATCH')
                        write_new(d/'rejected.json',report);history.append({'kind':'hypothetical_preview_rejected','action':action,'report':report,'actual_parent_unchanged':True});pending=None;continue
                    write_new(d/'returned.json',receipt);parent_unchanged()
                    if receipt.get('parent_observation_sha256')!=digest(last) or type(receipt.get('parent_revision')) is not int or receipt['parent_revision']!=last['revision'] or receipt.get('action')!=action or type(receipt.get('steps')) is not int or receipt['steps']!=expected or type(receipt.get('preview_id')) is not str or re.fullmatch('[A-Za-z0-9_-]{1,100}',receipt['preview_id']) is None or receipt['preview_id'] in previews or type(receipt.get('trajectory_sha256')) is not str or re.fullmatch('[a-f0-9]{64}',receipt['trajectory_sha256']) is None:raise RuntimeError('PREVIEW_BINDING_FAILED')
                    hypothetical=observation(receipt['hypothetical_observation'],last['session_id'],last);samples=bounded_samples(fork.preview_samples(receipt['preview_id']));write_new(d/'sampled_observations.json',samples);hyp_assessment=assess_samples(samples,last,hypothetical,state,receipt)
                    sandbox_steps+=expected;pending=None;pid=receipt['preview_id'];row={'preview_id':pid,'parent_observation_sha256':digest(last),'action':action,'hypothetical_observation':hypothetical,'hypothetical_assessment':hyp_assessment,'eligible':hyp_assessment.get('failed') is not True,'qualification':'HYPOTHETICAL sandbox future, not actual parent state or achieved task.'};previews[pid]=row
                    write_new(d/'candidate.json',row)
                    write_new(d/'committed_receipt.json',{'receipt_sha256':digest(receipt),'sandbox_steps_total':sandbox_steps,'actual_parent_unchanged':True});history.append({'kind':'hypothetical_preview','preview':copy.deepcopy(row)});continue
                if decision=='commit':
                    if p['action'] is not None or p['preview_id'] not in eligible:raise ValueError('Commit must name exact current eligible preview')
                    row=previews[p['preview_id']];expected=round(row['action']['arguments']['duration_s']/step_s)
                    if actual_steps+expected>self.max_actual_steps:reason='actual_step_budget';break
                    commit={'commit_id':out.name+'__commit_%02d'%commit_attempts,'preview_id':p['preview_id'],'expected_revision':last['revision'],'observation_sha256':digest(last)}
                    d=out/('commit_%02d'%commit_attempts);d.mkdir();write_new(d/'pending.json',commit);commit_attempts+=1;pending='commit';transition=bounded(fork.commit(copy.deepcopy(commit)));write_new(d/'returned.json',transition)
                    if transition.get('preview_id')!=p['preview_id'] or transition.get('preview_actual_byte_equal') is not True or transition.get('before_observation_sha256')!=digest(last) or type(transition.get('before_revision')) is not int or transition['before_revision']!=last['revision'] or type(transition.get('steps')) is not int or transition['steps']!=expected:raise RuntimeError('COMMIT_PREVIEW_PARITY_FAILED')
                    actual=observation(transition['observation'],last['session_id'],last)
                    if type(transition.get('after_revision')) is not int or transition['after_revision']!=actual['revision']:raise RuntimeError('COMMIT_REVISION_MISMATCH')
                    if actual!=row['hypothetical_observation']:raise RuntimeError('COMMIT_OBSERVATION_PARITY_FAILED')
                    samples=bounded_samples(fork.committed_samples(commit['commit_id']));write_new(d/'sampled_observations.json',samples);assessment=assess_samples(samples,last,actual,state,transition);actual_steps+=expected;pending=None;last=actual;state=copy.deepcopy(assessment.get('state'));write_new(out/('actual_observation_%02d.json'%commit_attempts),last);write_new(out/('actual_assessment_%02d.json'%commit_attempts),assessment);write_new(d/'committed.json',{'transition_sha256':digest(transition),'actual_steps_total':actual_steps});history.append({'kind':'actual_committed','preview_id':p['preview_id'],'transition':transition,'actual_assessment':assessment})
                    for value in previews.values():value['eligible']=False
                    if assessment.get('failed') is True:status='failed';reason='actual_physical_or_task_failure';break
                    if assessment['done']:status='completed';reason='actual_task_verified';break
                    continue
                if p['action'] is not None or p['preview_id'] is not None:raise ValueError('Finish/stop must not carry an action')
                reason='unverified_finish' if decision=='finish' else 'model_stopped';break
        except Exception as error:
            status='cancelled' if cancel.is_set() else 'failed';reason=type(error).__name__;write_new(out/'failure.json',{'category':reason,'pending_completion_unconfirmed':pending,'retry_allowed':False,'external_error_text_omitted':True})
        attempts=list((out/'model').glob('*/attempt.json'));receipts=[json.loads(p.read_bytes()) for p in (out/'model').glob('*/receipt.json')]
        result={'schema':'pulsemorph.embodied_preview_run.v2','status':status,'reason':reason,'task_verified_actual':bool(assessment and assessment['done']),'fixed_sha256':fixed_hash,'model_attempts':len(attempts),'live_model_calls':sum(r.get('live') is True for r in receipts),'model_completion_unconfirmed':max(0,len(attempts)-len(receipts)),'preview_attempts':preview_attempts,'accepted_previews':len(previews),'commit_attempts':commit_attempts,'confirmed_sandbox_steps':sandbox_steps,'confirmed_actual_steps':actual_steps,'completion_unconfirmed':pending,'last_actual_observation_sha256':digest(last) if last else None,'wall_seconds':time.monotonic()-start,'fixture_run':self.allow_fixture}
        write_new(out/'result.json',result);return result
