import tempfile
import unittest
from pathlib import Path

from experiments.functional_step_geometry.work import QualificationWork


class WorkTests(unittest.TestCase):
    def test_failed_attempt_survives_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'work.jsonl'
            work = QualificationWork(path)
            with self.assertRaisesRegex(ValueError, 'intentional'):
                with work.operation('adam'):
                    raise ValueError('intentional')
            resumed = QualificationWork(path)
            self.assertEqual(resumed.summary()['adam_attempted'], 1)
            self.assertEqual(resumed.summary()['adam_failed'], 1)

    def test_ceiling_precedes_ninth_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            work = QualificationWork(Path(directory) / 'work.jsonl')
            for _ in range(8):
                with work.operation('adam'):
                    pass
            with self.assertRaisesRegex(RuntimeError, 'ceiling'):
                with work.operation('adam'):
                    self.fail('Over-budget work executed')
            self.assertEqual(work.summary()['adam_attempted'], 8)


if __name__ == '__main__':
    unittest.main()
