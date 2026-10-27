"""Regression: native JSON scalars at render recording boundary, zero FK."""
import json
import unittest
import numpy as np
from pulsemorph.agentic.embodied.app.renderer_v2 import frame_error_record,encoded

class BodyRenderSerializationTests(unittest.TestCase):
    def test_numpy_comparison_result_is_native_bool_in_frame_record(self):
        active=np.float64(2.6)>=np.float64(2.)
        self.assertIsInstance(active,np.bool_)
        row=frame_error_record(np.int64(150),np.float64(3.),[('right_rubber_hand',np.float64(.008471),active),('left_rubber_hand',np.float32(.006274),np.bool_(False))])
        self.assertIs(type(row['state_index']),int);self.assertIs(type(row['sim_time_s']),float)
        for goal in row['actual_goals']:
            self.assertIs(type(goal['distance_m']),float);self.assertIs(type(goal['active']),bool)
        restored=json.loads(encoded([row]));self.assertTrue(restored[0]['actual_goals'][0]['active']);self.assertFalse(restored[0]['actual_goals'][1]['active'])
    def test_all_75_frame_error_rows_serialize_before_output(self):
        rows=[frame_error_record(i,np.float64(i*.04),[('right_rubber_hand',np.float64(.04-i*.0004),np.bool_(i>=50))]) for i in range(75)]
        result=json.loads(encoded(rows));self.assertEqual(len(result),75);self.assertFalse(result[0]['actual_goals'][0]['active']);self.assertTrue(result[-1]['actual_goals'][0]['active'])
    def test_nonfinite_is_not_silently_coerced_or_serialized(self):
        for invalid in (np.nan,np.inf,-np.inf):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):frame_error_record(1,.04,[('right_rubber_hand',invalid,True)])
if __name__=='__main__':unittest.main()
