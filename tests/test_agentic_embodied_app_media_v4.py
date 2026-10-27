"""Windows producer/parentmedia boundary only; no worker execution."""
from pathlib import Path,PureWindowsPath
import sys,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pulsemorph.agentic.embodied.app.manager_v1 import media_relative
class MediaBoundaryTests(unittest.TestCase):
 def test_native_windows_path_must_publish_posix(self):
  path=PureWindowsPath('C:/private/job/execution/media/episode.mp4').relative_to(PureWindowsPath('C:/private/job/execution'))
  self.assertFalse(media_relative('episode.mp4',str(path)));self.assertEqual(path.as_posix(),'media/episode.mp4');self.assertTrue(media_relative('episode.mp4',path.as_posix()))
 def test_worker4_only_changed_path_serialization(self):
  old=(ROOT/'src/pulsemorph/agentic/embodied/app/worker_v3.py').read_bytes();new=(ROOT/'src/pulsemorph/agentic/embodied/app/worker_v4.py').read_bytes()
  self.assertEqual(new.replace(b"'path':video.relative_to(output).as_posix()",b"'path':str(video.relative_to(output))"),old)
if __name__=='__main__':unittest.main()
