"""Summarize existing training receipts; no model calls or new experiments."""
import argparse
import csv
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def event_rows(campaign, event):
    receipts = {arm: read(campaign/arm/'events'/f'event_{event:03d}'/'RESULT.json') for arm in ('O','K')}
    rows = []
    for arm, receipt in receipts.items():
        records = receipt['records']
        assert receipt['status'] == 'completed' and len(records) == 256
        updates = sum(row['optimizer_stepped'] for row in records)
        assert updates == 128 and receipt['end_step'] == event*128
        anchors = sum(sum(draw['keep_mask'][0]) for row in records for draw in row['sampled_anchors'])
        # Boundary metrics cover both accumulated responses. Count each window once.
        boundaries = [row for row in records if row['optimizer_stepped']]
        assert sum(row['metrics']['response_presentations'] for row in boundaries) == 256
        exposed = sum(row['metrics']['supervised_position_exposure'] for row in boundaries)
        row = dict(event=event, arm=arm, presentations=256, optimizer_steps=updates,
                   start_step=receipt['start_step'], end_step=receipt['end_step'],
                   sampled_anchor_occurrences=anchors, supervised_position_occurrences=exposed)
        if arm == 'K':
            assert receipt['selection_totals_source'] == 'optimizer_boundary_accumulation_windows_once'
            totals = receipt['selection_totals']
            for key in ('available_position_occurrences','selected_position_occurrences',
                        'available_position_weight','selected_position_weight'):
                actual = sum(record['metrics'][key] for record in boundaries)
                assert actual == totals[key], (event, key, actual, totals[key])
                row[key] = actual
            assert exposed == row['selected_position_occurrences']
            row['selected_fraction_of_available'] = exposed/row['available_position_occurrences']
        else:
            row['available_position_occurrences'] = exposed
            row['selected_position_occurrences'] = exposed
            row['selected_fraction_of_available'] = 1.0
        rows.append(row)
    assert rows[0]['sampled_anchor_occurrences'] == rows[1]['sampled_anchor_occurrences']
    assert rows[0]['supervised_position_occurrences'] == rows[1]['available_position_occurrences']
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def run(root, output):
    campaign = root/'campaign_001'
    output.mkdir(exist_ok=True,parents=True)
    exposures, feedback = [], []
    for event in (5,6,7,8):
        exposures.extend(event_rows(campaign,event))
        event_root = campaign/f'event_{event:03d}'
        pool, preparations = read(event_root/'POOL.json'), read(event_root/'PREPARATIONS.json')
        records = [read(event_root/'responses'/(rid+'.json')) for rid in pool['record_ids']]
        assert len(records) == len(preparations) == 128
        assert all(row['status']=='prepared' for row in preparations)
        feedback.append(dict(event=event,responses=128,
            prompt_tokens=sum(len(row['prompt_ids']) for row in records),
            response_tokens=sum(len(row['response_ids']) for row in records),
            maximum_response_tokens=max(len(row['response_ids']) for row in records),
            outputs_at_2048_tokens=sum(len(row['response_ids'])==2048 for row in records),
            length_terminations=sum(row['finish_reason']=='length' for row in records),
            eos_terminations=sum(row['finish_reason']=='eos' for row in records),
            other_terminations=sum(row['finish_reason'] not in ('length','eos') for row in records),
            terminal_teacher_preparations=pool['teacher_terminal_forwards']))
    write_csv(output/'TRAINING_EXPOSURE.csv', exposures)
    write_csv(output/'FEEDBACK_LENGTHS.csv', feedback)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); run(args.root,args.output)
