"""The retained environment pin names imported modules, not wheel metadata."""
import unittest
from copy import deepcopy
from experiments.persistent_prefix_withholding.preflight import check_environment


class EnvironmentPinTests(unittest.TestCase):
    def setUp(self):
        self.expected = dict(python='3.12.15', torch='2.13.0+cu130', transformers='5.12.1')
        self.observed = dict(versions=dict(self.expected),
            package_metadata=dict(torch='2.13.0', transformers='5.12.1'),
            cuda_version='13.0', cuda_initialized=False)

    def test_wheel_suffix_does_not_replace_imported_version(self):
        before = deepcopy(self.observed)
        check_environment(self.observed, self.expected, 'training')
        self.assertEqual(before, self.observed)

    def test_actual_imported_version_change_still_fails(self):
        self.observed['versions']['torch'] = '2.13.0'
        with self.assertRaisesRegex(ValueError, 'Pinned environment differs'):
            check_environment(self.observed, self.expected, 'training')

    def test_serving_cuda_and_cpu_only_contract_remain_strict(self):
        expected = dict(self.expected, cuda='13.0')
        check_environment(self.observed, expected, 'serving')
        self.observed['cuda_version'] = '12.8'
        with self.assertRaisesRegex(ValueError, 'Pinned CUDA'):
            check_environment(self.observed, expected, 'serving')
        self.observed['cuda_version'] = '13.0'
        self.observed['cuda_initialized'] = True
        with self.assertRaisesRegex(ValueError, 'initialized CUDA'):
            check_environment(self.observed, expected, 'serving')


if __name__ == '__main__':
    unittest.main()
