"""Windows producer→frozen consumer media contract, no workers or native calls."""
import ast,json,sys,tempfile,unittest
from pathlib import Path,PureWindowsPath
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from pulsemorph.agentic.embodied.app import server_v1,server_v4,manager_v1
import test_agentic_body_app as manager_fixtures

class Tests(unittest.TestCase):
    def test_frozen_http_reused_and_worker4_selected(self):
        self.assertIs(server_v4.Server,server_v1.Server);self.assertIs(server_v4.Handler,server_v1.Handler)
        text=Path(server_v4.__file__).read_text();self.assertIn("with_name('worker_v4.py')",text);self.assertNotIn("with_name('worker_v3.py')",text)
    def test_windows_posix_producer_accepted_without_loosening_consumer(self):
        fixture=manager_fixtures.Tests('test_worker_video_path_and_decision_events_merge_once');fixture.setUp()
        try:
            manager=fixture.manager(result={'status':'completed','task_verified_actual':True});identity=manager.create({'scenario':'right_reach','instruction':''});manager.threads[identity].join(3);execution=manager.jobs_dir/identity/'execution';(execution/'media').mkdir();video=execution/'media/episode.mp4';video.write_bytes(b'fixture actual recorded movie');relative=PureWindowsPath('media\\episode.mp4').as_posix();self.assertEqual(relative,'media/episode.mp4');self.assertFalse(manager_v1.media_relative('episode.mp4','media\\episode.mp4'));self.assertTrue(manager_v1.media_relative('episode.mp4',relative));progress={'phase':'finished','media':{'episode.mp4':{'path':relative,'role':'actual_recorded','sha256':manager_v1.sha(video)}}};(execution/'progress.json').write_text(json.dumps(progress));manager._progress(manager.jobs[identity],execution);job=manager.get(identity);self.assertEqual(job['status'],'completed');self.assertTrue(job['task_verified_actual']);self.assertIsNotNone(job['video_url']);self.assertEqual(manager.media_path(identity,'episode.mp4'),video)
        finally:fixture.tearDown();fixture.doCleanups()

if __name__=='__main__':unittest.main()
