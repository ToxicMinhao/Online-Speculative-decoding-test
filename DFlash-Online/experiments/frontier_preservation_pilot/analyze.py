"""Export compact derivative-feasibility evidence without prompts or anchors."""

import argparse
import csv
import json
from pathlib import Path

from online_feedback.storage import write_json


def analyze(root, destination):
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    probe = root / 'derivative_support_001'
    read = lambda path: json.loads(path.read_text())
    result = read(probe / 'RESULT.json')
    process = read(root / 'QUALIFICATION_PROCESS.json')
    work = read(probe / 'WORK.json')
    window = read(probe / 'WINDOW.json')
    methods = []
    for method in result['methods']:
        rows = []
        for index, attempt in enumerate(method['responses']):
            rows.append({key: attempt[key] for key in ('status', 'wall_s', 'error') if key in attempt})
            if 'traceback' in attempt:
                (destination / f'{method["method"]}_{index}_TRACEBACK.txt').write_text(attempt['traceback'])
        methods.append(dict(method=method['method'], passed=method['passed'], responses=rows,
                            unattempted_responses=method['unattempted_responses']))
    summary = dict(
        status=result['status'], execution_commit=result['execution_commit'],
        parent_execution_commit=result['parent_execution_commit'],
        restored_parent=result['restored_parent'], restoration=result['restoration'],
        first_order_control=result['first_order_control'], methods=methods,
        runtime=result['runtime'], optimizer_recipe=result['optimizer_recipe'],
        selected_training_response_count=len(window['responses']),
        frozen_anchor_counts=[sum(sum(row) for row in response['keep']) for response in window['responses']],
        physical_forward_entries=result['physical_work_delta'], work=work,
        peak_allocated_bytes=result['peak_allocated_bytes'], peak_reserved_bytes=result['peak_reserved_bytes'],
        probe_wall_s=result['wall_s'], process=process,
        full_coefficient_sensitivity_qualified=result['full_coefficient_sensitivity_qualified'],
        production_completed=False, serving_comparison=None,
        outcome='Native derivative prerequisite blocked; candidate efficacy untested',
    )
    write_json(destination / 'QUALIFICATION.json', summary)
    with (destination / 'WORK.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['operation', 'attempted', 'completed', 'incomplete'])
        writer.writeheader()
        for operation, attempted in work['attempted'].items():
            completed = work['completed'].get(operation, 0)
            writer.writerow(dict(operation=operation, attempted=attempted, completed=completed,
                                 incomplete=attempted - completed))
    write_json(destination / 'STATE_CHECKS.json', dict(
        initial=read(probe / 'trainer/RESUME_CHECK.json'),
        final=read(probe / 'FINAL_RESTORATION.json'),
    ))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.output)
    print(json.dumps(dict(status=result['status'], production_completed=result['production_completed'])))
