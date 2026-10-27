"""Durable observe/propose/execute loop for an admitted persistent backend.

This module supplies no physics or tools. Backend and independent task verifier
are trusted local dependencies; model proposals never acquire execution authority.
"""
import copy,hashlib,json,math,re,threading,time
from pathlib import Path
from jsonschema import Draft202012Validator
from .provider_v1 import plan_action

CACHE=Path.home()/'.cache/pulsemorph-agentic'
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def write_new(p,v):
    with Path(p).open('xb') as f:f.write(canonical(v))
def bounded(v):
    b=canonical(v)
    if len(b)>512*1024:raise ValueError('Bounded finite JSON required')
    return copy.deepcopy(v)
def response_schema(capabilities):
    tools=capabilities.get('tools') if type(capabilities) is dict else None
    if type(tools) is not dict or len(tools)!=1:raise ValueError('V1 requires exactly one admitted body-control tool')
    choices=[]
    for name,definition in tools.items():
        if type(name) is not str or re.fullmatch('[a-z][a-z0-9_]{1,63}',name) is None or type(definition) is not dict:raise ValueError('Typed backend tool required')
        args=definition.get('arguments_schema');Draft202012Validator.check_schema(args)
        if type(args) is not dict or args.get('type')!='object' or args.get('additionalProperties') is not False:raise ValueError('Closed argument object required')
        args=copy.deepcopy(args)
        targets=args.get('properties',{}).get('joint_targets_rad')
        if type(targets) is not dict or targets.get('type')!='object' or not targets.get('properties'):raise ValueError('Named backend joint targets required')
        names=list(targets['properties'])
        args['properties']['joint_targets_rad']={'type':'array','minItems':1,'maxItems':len(names),'items':{'type':'object','additionalProperties':False,'properties':{'joint_name':{'type':'string','enum':names},'position_rad':{'type':'number'}},'required':['joint_name','position_rad']}}
        choices.append({'type':'object','additionalProperties':False,'properties':{'tool':{'type':'string','enum':[name]},'arguments':args},'required':['tool','arguments']})
    action=choices[0];action['type']=['object','null']
    return {'type':'object','additionalProperties':False,'properties':{'decision':{'type':'string','enum':['act','finish','stop']},'action':action,'summary':{'type':'string','maxLength':1200},'reason':{'type':'string','maxLength':1200}},'required':['decision','action','summary','reason']}
def observation(value,session_id=None,previous=None):
    value=bounded(value)
    if type(value) is not dict or not {'schema','session_id','revision','sim_time_s','qpos','qvel','body_poses','contacts','physical_flags'}<=set(value):raise ValueError('Actual full body observation required')
    if type(value['schema']) is not str or type(value['session_id']) is not str or not value['session_id'] or type(value['revision']) is not int or value['revision']<0:raise ValueError('Observation identity required')
    if session_id is not None and value['session_id']!=session_id:raise RuntimeError('Session changed')
    if type(value['sim_time_s']) not in (int,float) or not math.isfinite(value['sim_time_s']) or value['sim_time_s']<0:raise ValueError('Finite simulation clock required')
    for key in ('qpos','qvel'):
        if type(value[key]) is not list or not value[key] or len(value[key])>256 or any(type(x) not in (int,float) or not -1e12<x<1e12 for x in value[key]):raise ValueError('Actual finite generalized state required')
    if type(value['body_poses']) is not dict or type(value['contacts']) is not list or type(value['physical_flags']) is not dict:raise ValueError('Body/contact/flag observation required')
    if previous is not None and (value['revision']<=previous['revision'] or value['sim_time_s']<previous['sim_time_s']):raise RuntimeError('Stale or regressing observation')
    return value

class EmbodiedOrchestrator:
    def __init__(self,provider,*,task_verifier,max_model_calls=6,max_actions=6,max_physics_steps=12000,wall_seconds=1200,min_actions=0,allow_fixture=False,action_rejection_type=None):
        for v,lo,hi in [(max_model_calls,1,12),(max_actions,1,12),(max_physics_steps,1,48000)]:
            if type(v) is not int or not lo<=v<=hi:raise ValueError('Finite fixed budget required')
        if type(wall_seconds) not in (int,float) or not 1<=wall_seconds<=3600:raise ValueError('Finite wall budget required')
        if not callable(task_verifier):raise ValueError('Independent local task verifier required')
        if type(min_actions) is not int or not 0<=min_actions<=max_actions:raise ValueError('Bounded minimum action witness required')
        if action_rejection_type is not None and (not isinstance(action_rejection_type,type) or not issubclass(action_rejection_type,Exception)):raise ValueError('Typed backend preexecution rejection required')
        self.provider=provider;self.task_verifier=task_verifier;self.max_model_calls=max_model_calls;self.max_actions=max_actions;self.max_physics_steps=max_physics_steps;self.wall_seconds=wall_seconds;self.min_actions=min_actions;self.allow_fixture=allow_fixture;self.action_rejection_type=action_rejection_type;self._lock=threading.Lock()
    def run(self,intent,task,backend,output,*,cancel_event=None):
        if type(intent) is not str or not 3<=len(intent.strip())<=3000:raise ValueError('Bounded fixed intent required')
        task=bounded(task)
        if type(task) is not dict:raise ValueError('Fixed task criteria required')
        output=Path(output).absolute()
        if re.fullmatch('[A-Za-z0-9_-]{1,80}',output.name) is None:raise ValueError('Safe unique run ID required')
        if output.resolve()==CACHE.resolve() or not output.resolve().is_relative_to(CACHE.resolve()):raise ValueError('Private dedicated run directory required')
        for p in (output,*output.parents):
            if p.is_symlink() or p.exists() and getattr(p.lstat(),'st_file_attributes',0)&1024:raise ValueError('Linked output forbidden')
        if not self._lock.acquire(blocking=False):raise RuntimeError('Orchestrator already running')
        try:return self._run(intent.strip(),task,backend,output,cancel_event or threading.Event())
        finally:self._lock.release()
    def _run(self,intent,task,backend,output,cancel):
        output.mkdir(parents=True,exist_ok=False);started=time.monotonic();capabilities=bounded(backend.capabilities());schema=response_schema(capabilities)
        step_s=capabilities.get('physics_step_s')
        if type(step_s) not in (int,float) or not 0<step_s<=.02:raise ValueError('Backend physical step duration required')
        fixed={'intent':intent,'task':task,'capabilities':capabilities,'argument_transport':'Model joint_targets_rad is a unique-name list; locally mapped to backend joint-name dictionary without changing numeric targets.','budgets':{'model_calls':self.max_model_calls,'actions':self.max_actions,'physics_steps':self.max_physics_steps,'wall_seconds':self.wall_seconds,'minimum_committed_actions_for_completion':self.min_actions}}
        write_new(output/'fixed.json',fixed);fixed_hash=digest(fixed);history=[];actions=steps=0;pending=False;status='stopped';reason='budget_exhausted';last=None
        def check():
            if cancel.is_set():raise RuntimeError('CANCELLED')
            if time.monotonic()-started>=self.wall_seconds:raise RuntimeError('WALL_BUDGET')
            if digest(json.loads((output/'fixed.json').read_bytes()))!=fixed_hash or bounded(backend.capabilities())!=capabilities:raise RuntimeError('FIXED_CONTRACT_CHANGED')
        verifier_state=None;assessments={}
        def verified(obs):
            nonlocal verifier_state
            key=digest(obs)
            if key in assessments:return copy.deepcopy(assessments[key])
            result=bounded(self.task_verifier(copy.deepcopy(task),copy.deepcopy(obs),copy.deepcopy(verifier_state)))
            if type(result) is not dict or type(result.get('done')) is not bool or type(result.get('reason')) is not str or result.get('done') and result.get('failed') is True:raise ValueError('Local task result required')
            verifier_state=copy.deepcopy(result.get('state'));assessments[key]=result
            write_new(output/('assessment_%02d.json'%len(assessments)),{'observation_sha256':key,'assessment':result})
            return result
        try:
            check();last=observation(backend.observe());write_new(output/'observation_00.json',last)
            for index in range(self.max_model_calls):
                check();assessment=verified(last)
                if assessment.get('failed') is True:status='failed';reason='local_task_failed';break
                committed=sum('transition' in item for item in history)
                if assessment['done']:
                    status='completed' if committed>=self.min_actions else 'stopped';reason='local_task_verified' if status=='completed' else 'minimum_action_witness_unmet';break
                request={'schema':'pulsemorph.embodied_decision_request.v1','fixed':fixed,'observation':last,'observation_sha256':digest(last),'local_assessment':assessment,'history':history,'remaining':{'model_calls':self.max_model_calls-index,'actions':self.max_actions-actions,'physics_steps':self.max_physics_steps-steps}}
                result=plan_action(self.provider,request,schema,output/'model'/('call_%02d'%index),cancel_event=cancel,timeout=min(120,max(.1,self.wall_seconds-(time.monotonic()-started))))
                check()
                if result.get('observation_sha256')!=digest(last):raise RuntimeError('STALE_PROPOSAL')
                receipt=result['receipt']
                if receipt.get('live') is not True and not self.allow_fixture:raise RuntimeError('LIVE_PROVIDER_REQUIRED')
                proposal=result['proposal'];Draft202012Validator(schema).validate(proposal);decision=proposal['decision']
                if (decision=='act')!=(proposal['action'] is not None):raise ValueError('Decision/action mismatch')
                if decision in ('finish','stop'):
                    assessment=verified(last);status='completed' if decision=='finish' and assessment['done'] and committed>=self.min_actions else 'stopped';reason='local_task_verified' if status=='completed' else 'minimum_action_witness_unmet' if assessment['done'] and committed<self.min_actions else 'model_finish_unverified' if decision=='finish' else 'model_stopped';break
                if actions>=self.max_actions:reason='action_budget';break
                # Paused state must still match the exact decision observation.
                current=observation(backend.observe(),last['session_id'])
                if current!=last:raise RuntimeError('STATE_CHANGED_WHILE_PLANNING')
                arguments=copy.deepcopy(proposal['action']['arguments']);duration=arguments.get('duration_s')
                target_list=arguments['joint_targets_rad'];names=[item['joint_name'] for item in target_list]
                if len(names)!=len(set(names)):raise ValueError('Duplicate joint target forbidden')
                arguments['joint_targets_rad']={item['joint_name']:item['position_rad'] for item in target_list}
                if type(duration) not in (int,float) or not math.isfinite(duration) or duration<=0:raise ValueError('Finite action duration required')
                predicted_steps=round(duration/step_s)
                if predicted_steps<1 or abs(predicted_steps*step_s-duration)>1e-9 or steps+predicted_steps>self.max_physics_steps:raise RuntimeError('ACTION_EXCEEDS_PHYSICS_BUDGET')
                action={'action_id':output.name+'__action_%02d'%actions,'observation_sha256':digest(last),'expected_revision':last['revision'],'tool':proposal['action']['tool'],'arguments':arguments}
                d=output/('action_%02d'%actions);d.mkdir();write_new(d/'pending.json',{'action':action,'expected_steps':predicted_steps});pending=True;actions+=1
                try:transition=bounded(backend.execute(copy.deepcopy(action)))
                except Exception as error:
                    if self.action_rejection_type is None or not isinstance(error,self.action_rejection_type):raise
                    after=observation(backend.observe(),last['session_id'])
                    if after!=last:raise RuntimeError('REJECTED_ACTION_CHANGED_STATE')
                    report=bounded(error.report)
                    if type(report) is not dict or type(report.get('physics_steps')) is not int or report['physics_steps']!=0 or type(report.get('revision_unchanged')) is not int or report['revision_unchanged']!=last['revision'] or report.get('observation_sha256')!=digest(last):raise RuntimeError('REJECTION_PROOF_MISMATCH')
                    write_new(d/'rejected.json',{'action':action,'report':report,'unchanged_observation_sha256':digest(after),'new_physics_steps':0})
                    pending=False;history.append({'action':action,'rejection_report':report,'observation_unchanged':True,'model_summary':proposal['summary']});continue
                write_new(d/'returned.json',transition)
                if type(transition) is not dict or transition.get('action_id')!=action['action_id'] or transition.get('before_observation_sha256')!=digest(last) or type(transition.get('before_revision')) is not int or transition['before_revision']!=last['revision'] or type(transition.get('steps')) is not int or transition['steps']!=predicted_steps:raise RuntimeError('TRANSITION_ASSIGNMENT_OR_BUDGET_FAILED')
                new=observation(transition['observation'],last['session_id'],last)
                if type(transition.get('after_revision')) is not int or transition['after_revision']!=new['revision']:raise RuntimeError('TRANSITION_REVISION_FAILED')
                steps+=transition['steps'];pending=False;write_new(d/'committed.json',{'action_sha256':digest(action),'transition_sha256':digest(transition),'steps_total':steps});write_new(output/('observation_%02d.json'%actions),new)
                history.append({'action':action,'transition':transition,'model_summary':proposal['summary']});last=new
                assessment=verified(last)
                if assessment.get('failed') is True:status='failed';reason='local_task_failed';break
                if assessment['done']:
                    status='completed' if sum('transition' in item for item in history)>=self.min_actions else 'stopped';reason='local_task_verified' if status=='completed' else 'minimum_action_witness_unmet';break
                if steps>=self.max_physics_steps:reason='physics_budget';break
        except Exception as error:
            status='cancelled' if cancel.is_set() else 'failed';reason=type(error).__name__
            write_new(output/'failure.json',{'category':reason,'action_completion_unconfirmed':pending,'qualification':'No retry/resume; pending action may have changed backend state. External error text omitted.'})
        calls=list((output/'model').glob('*/attempt.json'));receipts=[json.loads(p.read_bytes()) for p in (output/'model').glob('*/receipt.json')]
        result={'schema':'pulsemorph.embodied_run.v1','status':status,'reason':reason,'fixed_sha256':fixed_hash,'model_attempts':len(calls),'live_model_calls':sum(r.get('live') is True for r in receipts),'model_completion_unconfirmed':max(0,len(calls)-len(receipts)),'action_attempts':actions,'committed_actions':sum('transition' in item for item in history),'rejected_actions':sum('rejection_report' in item for item in history),'action_completion_unconfirmed':pending,'confirmed_physics_steps':steps,'last_observation_sha256':digest(last) if last else None,'wall_seconds':time.monotonic()-started,'fixture_run':self.allow_fixture,'task_verified':bool(last and assessments.get(digest(last),{}).get('done')),'minimum_actions_met':sum('transition' in item for item in history)>=self.min_actions}
        write_new(output/'result.json',result);return result
