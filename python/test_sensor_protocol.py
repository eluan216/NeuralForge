import unittest
import numpy as np
from sensor_benchmark import subject_split, VALIDATION_SUBJECTS
class SubjectProtocol(unittest.TestCase):
    def test_partition_uses_participants_not_random_windows(self):
        subjects=np.repeat([1,3,5,15,17,26,30],3)
        fitting,validation=subject_split(subjects,np.array([2,4,9]))
        self.assertFalse(np.any(fitting&validation))
        self.assertTrue(np.all(fitting|validation))
        self.assertEqual(set(subjects[validation]),set(VALIDATION_SUBJECTS))
        for subject in set(subjects):
            self.assertIn(int(validation[subjects==subject].sum()),(0,3))
    def test_overlap_or_missing_validation_participants_fails(self):
        with self.assertRaises(ValueError):
            subject_split(np.array([1,3,15,17,26,30]),np.array([3]))
        with self.assertRaises(ValueError):
            subject_split(np.array([1,3]),np.array([2]))
if __name__=="__main__":
    unittest.main()
