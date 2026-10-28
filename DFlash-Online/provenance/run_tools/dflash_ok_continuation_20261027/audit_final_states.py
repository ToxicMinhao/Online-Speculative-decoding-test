"""CPU-only evidence inventory after execution; never constructs a model/optimizer."""
import hashlib
import json
from pathlib import Path
import sys
from experiments.ok_continuation.resume import load_snapshot, differences

root = Path(sys.argv[1]).resolve()
if not (root / 'EXIT_CODE').is_file():
    raise RuntimeError('Wait for scientific execution and postflight to finish')
result = {'status': 'pending', 'scope': 'Read-only CPU inspection; zero model forwards or optimizer steps',
          'arms': {}, 'files': {}, 'errors': []}
try:
    result['supervisor_exit_code'] = int((root / 'EXIT_CODE').read_text())
    status = json.loads((root / 'campaign_001/STATUS.json').read_text())
    result['campaign_status'] = status['status']
    result['postflight'] = json.loads((root / 'POSTFLIGHT.json').read_text()) if (root / 'POSTFLIGHT.json').is_file() else None
    plan = json.loads((root / 'sealed/PLAN.json').read_text())
    prior_rng = None
    for arm, objective in [('O', 'osd'), ('K', 'osd_first_error')]:
        try:
            folder = root / 'campaign_001' / arm
            state_path = folder / 'training_state.pt'
            files = [state_path] if state_path.is_file() else []
            files += sorted(p for p in (folder / 'events').glob('event_*/export/*') if p.is_file())
            for path in files:
                with path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                result['files'][str(path)] = {'bytes': path.stat().st_size, 'sha256': digest}
            if status['status'] == 'completed':
                state, receipt = load_snapshot(state_path, folder / 'events/event_004/export',
                    objective=objective, config_text=Path(plan['config']).read_text(), completed_event=4)
                result['arms'][arm] = receipt
                if prior_rng is not None:
                    mismatches = differences(prior_rng, state['rng'], 'rng')
                    result['final_O_K_rng_mismatches'] = mismatches
                    if mismatches:
                        result['errors'].append({'scope': 'final_O_K_rng', 'mismatches': mismatches})
                prior_rng = state['rng']
                del state
        except Exception as error:
            result['errors'].append({'scope': arm, 'error': repr(error)})
    if result['supervisor_exit_code'] != 0 or result['postflight'] is None or result['postflight'].get('status') != 'passed':
        result['errors'].append({'scope': 'execution', 'error': 'Supervisor or postflight did not pass'})
    result['status'] = 'failed' if result['errors'] else 'passed' if status['status'] == 'completed' else 'incomplete'
except Exception as error:
    result['status'] = 'failed'
    result['errors'].append({'scope': 'inventory', 'error': repr(error)})
finally:
    (root / 'FINAL_STATE_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'files': len(result['files']),
                  'arms': list(result['arms']), 'errors': result['errors']}))
raise SystemExit(0 if result['status'] == 'passed' else 1)
