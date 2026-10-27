"""Metadata-only Windows publication stress; no simulator/provider execution."""
from pathlib import Path
import json,os,sys,tempfile,threading,time,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.embodied.app import worker_v2 as w
class MetadataTests(unittest.TestCase):
 def test_permission_only_retries_same_bytes_once_write(self):
  with tempfile.TemporaryDirectory() as t:
   path=Path(t)/'progress.json';path.write_text('{"old":true}');real=os.replace;calls=[]
   def replace(src,dst):
    calls.append((Path(src),Path(src).read_bytes()))
    if len(calls)<3:raise PermissionError('sharing fixture')
    return real(src,dst)
   with patch.object(w.os,'replace',side_effect=replace):w.atomic(path,{'value':7},timeout_s=.2,sleep_s=.001)
   self.assertEqual(json.loads(path.read_bytes()),{'value':7});self.assertEqual(len(calls),3);self.assertEqual(len({str(x[0]) for x in calls}),1);self.assertEqual(len({x[1] for x in calls}),1);self.assertEqual(list(Path(t).glob('*.tmp')),[])
 def test_timeout_keeps_old_target_and_cleans_owned_temporary(self):
  with tempfile.TemporaryDirectory() as t:
   path=Path(t)/'progress.json';old=b'{"old":true}';path.write_bytes(old);start=time.monotonic()
   with patch.object(w.os,'replace',side_effect=PermissionError('held target')):
    with self.assertRaises(w.MetadataPublicationError) as caught:w.atomic(path,{'new':1},timeout_s=.03,sleep_s=.005)
   self.assertLess(time.monotonic()-start,.5);self.assertGreaterEqual(caught.exception.attempts,2);self.assertEqual(path.read_bytes(),old);self.assertEqual(list(Path(t).glob('*.tmp')),[])
 def test_other_errors_and_interruption_never_retry(self):
  for error in (OSError('other failure'),KeyboardInterrupt()):
   with tempfile.TemporaryDirectory() as t:
    path=Path(t)/'progress.json';path.write_text('{}')
    with patch.object(w.os,'replace',side_effect=error) as replace:
     with self.assertRaises(type(error)):w.atomic(path,{'x':1})
    self.assertEqual(replace.call_count,1);self.assertEqual(path.read_bytes(),b'{}');self.assertEqual(list(Path(t).glob('*.tmp')),[])
 def test_actual_windows_concurrent_readers_only_valid_json(self):
  with tempfile.TemporaryDirectory() as t:
   path=Path(t)/'progress.json';w.atomic(path,{'sequence':0,'text':'fixture'});stop=threading.Event();errors=[];readcounts=[]
   def reader():
    count=0
    while not stop.is_set():
     try:
      with path.open('rb') as stream:data=stream.read();time.sleep(.0005)
      parsed=json.loads(data);self.assertIs(type(parsed['sequence']),int);count+=1
     except OSError:pass
     except Exception as error:errors.append(type(error).__name__)
     time.sleep(.0005)
    readcounts.append(count)
   threads=[threading.Thread(target=reader) for _ in range(3)]
   for thread in threads:thread.start()
   try:
    for sequence in range(1,31):w.atomic(path,{'sequence':sequence,'text':'fixture'},timeout_s=2.,sleep_s=.002)
   finally:
    stop.set()
    for thread in threads:thread.join(3)
   self.assertFalse(errors);self.assertEqual(json.loads(path.read_bytes())['sequence'],30);self.assertGreater(sum(readcounts),0);self.assertEqual(list(Path(t).glob('*.tmp')),[])
 def test_terminal_failure_saved_without_restarting_inner_job(self):
  with tempfile.TemporaryDirectory() as t:
   jobs=Path(t);out=jobs/'body_fixture/execution';out.mkdir(parents=True)
   w.write_new(out/'terminal_numeric_result.json',{'schema':'pulsemorph.body_agent_result.v1','status':'completed','task_verified_actual':True,'counts':{'actual_steps':600}})
   with patch.object(w,'JOBS_ROOT',jobs),patch.object(w,'_run_job',side_effect=w.MetadataPublicationError('progress.json',3,.1)) as run:
    with self.assertRaises(w.MetadataPublicationError):w.run_job(ROOT,out.parent/'request.json',out)
   self.assertEqual(run.call_count,1);result=json.loads((out/'result.json').read_bytes());self.assertEqual(result['status'],'failed');self.assertTrue(result['task_verified_actual']);self.assertTrue(result['terminal_publication_failed']);self.assertTrue((out/'terminal_publication_failure.json').is_file())
if __name__=='__main__':unittest.main()
