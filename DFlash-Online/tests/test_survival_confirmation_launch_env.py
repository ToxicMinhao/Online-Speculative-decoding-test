"""Exercise launch environment and compiler admission without importing a model."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION = ROOT / 'experiments/survival_confirmation/run_all.sh'
PILOT = ROOT / 'experiments/survival_head/run_all.sh'
SETUP = '/data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh'
CONTROLLER = '/data/conda_envs/minhao/dflash_specforge_20261022'
COMPILER = '/data/conda_envs/minhao/dflash_tree_qualification_v2'
COMMON_KEYS = (
    'PYTHONDONTWRITEBYTECODE', 'CONDA_PREFIX', 'PATH', 'CC', 'CXX',
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'TOKENIZERS_PARALLELISM',
    'LAUNCH_SETUP_SENTINEL',
)


class LaunchEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.setup = self.root / 'setup.sh'
        self.setup.write_text('export LAUNCH_SETUP_SENTINEL=shared_setup\n')
        self.environment = dict(PATH='/usr/bin:/bin', OMP_NUM_THREADS='99',
                                HF_HUB_OFFLINE='0', TOKENIZERS_PARALLELISM='true')

    def prefix_environment(self, path, end_marker):
        script = path.read_text().split(end_marker, 1)[0]
        script = script.replace(SETUP, shlex.quote(str(self.setup)))
        result = subprocess.run(
            ['/bin/bash', '-c', script + '\n/usr/bin/env -0', 'launch-prefix',
             str(self.root), str(ROOT)], env=self.environment,
            capture_output=True, check=True, timeout=10,
        )
        return dict(item.split('=', 1) for item in result.stdout.decode().split('\0') if item)

    def fixture_launcher(self):
        controller = self.root / 'controller'
        compiler = self.root / 'compiler'
        (controller / 'bin').mkdir(parents=True)
        (compiler / 'bin').mkdir(parents=True)
        for name in ('gcc', 'g++'):
            executable = compiler / 'bin' / ('x86_64-conda-linux-gnu-' + name)
            executable.write_text('#!/bin/sh\nexit 0\n')
            executable.chmod(0o755)
        python = controller / 'bin/python'
        python.write_text(
            '#!' + sys.executable + '\n'
            'import json, os, sys\n'
            'with open(os.environ["LAUNCH_TEST_CALLS"], "a") as output:\n'
            '    output.write(json.dumps({"argv": sys.argv[1:], "env": dict(os.environ)}) + "\\n")\n'
        )
        python.chmod(0o755)
        script = CONFIRMATION.read_text().replace(SETUP, str(self.setup))
        script = script.replace(CONTROLLER, str(controller)).replace(COMPILER, str(compiler))
        launcher = self.root / 'run.sh'
        launcher.write_text(script)
        run = self.root / 'run'
        run.mkdir()
        calls = self.root / 'calls.jsonl'
        environment = dict(self.environment, LAUNCH_TEST_CALLS=str(calls))
        return launcher, run, calls, environment, compiler

    def test_shell_syntax(self):
        for path in (CONFIRMATION, PILOT):
            with self.subTest(path=path.name):
                subprocess.run(['/bin/bash', '-n', str(path)], check=True, timeout=10)

    def test_startup_environment_matches_completed_pilot(self):
        pilot = self.prefix_environment(PILOT, 'test ! -e "$run_root/START_UTC"')
        confirmation = self.prefix_environment(CONFIRMATION, 'test -x "$CC"')
        self.assertEqual({key: confirmation[key] for key in COMMON_KEYS},
                         {key: pilot[key] for key in COMMON_KEYS})
        self.assertEqual(confirmation['CONDA_PREFIX'], CONTROLLER)
        self.assertEqual(confirmation['CC'], COMPILER + '/bin/x86_64-conda-linux-gnu-gcc')
        self.assertEqual(confirmation['CXX'], COMPILER + '/bin/x86_64-conda-linux-gnu-g++')
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            self.assertEqual(confirmation[key], '4')
        for key in ('PYTHONDONTWRITEBYTECODE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
            self.assertEqual(confirmation[key], '1')
        self.assertEqual(confirmation['TOKENIZERS_PARALLELISM'], 'false')

    def test_runtime_and_report_receive_restored_environment(self):
        launcher, run, calls, environment, compiler = self.fixture_launcher()
        result = subprocess.run(['/bin/bash', str(launcher), str(run), str(ROOT)],
                                env=environment, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        rows = [json.loads(line) for line in calls.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertIn('experiments.survival_confirmation.runtime', rows[0]['argv'])
        self.assertIn('experiments.survival_confirmation.report', rows[1]['argv'])
        for row in rows:
            self.assertEqual(row['env']['CXX'], str(compiler / 'bin/x86_64-conda-linux-gnu-g++'))
            self.assertEqual(row['env']['OMP_NUM_THREADS'], '4')
            self.assertEqual(row['env']['HF_HUB_OFFLINE'], '1')
            self.assertEqual(row['env']['CUDA_VISIBLE_DEVICES'], '0')
        self.assertEqual((run / 'EXIT_CODE').read_text().strip(), '0')

    def test_missing_compiler_stops_before_any_python_or_started_marker(self):
        launcher, run, calls, environment, compiler = self.fixture_launcher()
        for name in ('gcc', 'g++'):
            with self.subTest(compiler=name):
                executable = compiler / 'bin' / ('x86_64-conda-linux-gnu-' + name)
                executable.chmod(0o644)
                result = subprocess.run(['/bin/bash', str(launcher), str(run), str(ROOT)],
                                        env=environment, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 1)
                self.assertFalse(calls.exists())
                self.assertFalse((run / 'STARTED_AT').exists())
                executable.chmod(0o755)


if __name__ == '__main__':
    unittest.main()
