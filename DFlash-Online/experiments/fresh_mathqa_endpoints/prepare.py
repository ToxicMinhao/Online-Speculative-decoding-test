"""Assemble already authenticated inputs; selection and model work happen elsewhere."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from online_feedback.storage import write_json
from .evaluation import dataset_contract


def identity(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(bytes=Path(path).stat().st_size, sha256=digest)


def prepare(task, package, remote_root, remote_source, commit, compiler_baseline):
    """Create a new immutable execution package after private archive delivery."""
    task, package = Path(task), Path(package)
    if package.exists():
        raise FileExistsError('Never overwrite a staged plan')
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Pin a complete published source commit')
    closeout_path = task/'closeout_audit/CLOSEOUT.json'
    closeout = json.loads(closeout_path.read_text())
    if closeout['status'] != 'passed':
        raise ValueError('Complete PR54 archive delivery and keepers first')
    package.mkdir(parents=True)
    shutil.copytree(task/'panel_audit/sealed', package/'sealed')
    for source, name in [(task/'panel_audit/POOL_AUDIT.json','POOL_AUDIT.json'),
                         (closeout_path,'PR54_CLOSEOUT.json'),
                         (task/'EXPECTED_INPUT_IDENTITIES.json','EXPECTED_INPUT_IDENTITIES.json')]:
        shutil.copy2(source, package/'sealed'/name)
    old = task.parent/'dflash_full_c_gsm8k_20261118/package_001/sealed'
    shutil.copy2(old/'RUNTIME_PIN.json', package/'sealed/RUNTIME_PIN.json')
    expected = json.loads((package/'sealed/EXPECTED_INPUT_IDENTITIES.json').read_text())
    by_role = {role:[r for r in expected if r['role']==role] for role in
               ('O58_selected_endpoint','AREPO58','C58','released_target')}
    checkpoints = {}
    checkpoint_ids = {}
    for method, role in [('O58','O58_selected_endpoint'),('AREPO58','AREPO58'),('KPLUS_C58','C58')]:
        row = next(r for r in by_role[role] if r['path'].endswith('/model.safetensors'))
        checkpoints[method] = str(Path(row['path']).parent)
        checkpoint_ids[method] = row['expected_identity']
    target_files = {Path(r['path']).name:r['expected_identity'] for r in by_role['released_target']}
    tokenizer_files = {k:v for k,v in target_files.items() if k in
                       ('tokenizer.json','tokenizer_config.json','merges.txt','vocab.json')}
    warmups = json.loads((package/'sealed/WARMUP_CANONICALS.json').read_text())
    producers = {r['reference_checkpoint'] for r in warmups.values()}
    if len(producers) != 1:
        raise ValueError('Fixed warmup witness producer is ambiguous')
    sealed = str(Path(remote_root)/'sealed')
    plan = dict(schema='fresh_mathqa_endpoints_v1', root=remote_root,
        source_root=remote_source, source_manifest=str(Path(remote_root)/'SOURCE.json'),
        execution_commit=commit, session=dataset_contract('mathqa')['session'],
        target=str(Path(by_role['released_target'][0]['path']).parent),
        training_python='/data/conda_envs/minhao/dflash_specforge_20261022/bin/python',
        worker_python='/data/conda_envs/minhao/adaflash_author/bin/python',
        devices=dict(A='GPU-9986735a-3e68-d1bd-5998-9634ef24341e',
                     B='GPU-c6943f33-bcd0-a724-0b35-f77af6b0a3f1'),
        checkpoints=checkpoints, checkpoint_identities=checkpoint_ids,
        canonical_provenance=dict(
            target_identity=dict(revision='b968826d9c46dd6066d109eabc6255188de91218',files=target_files),
            tokenizer_identity=dict(revision='b968826d9c46dd6066d109eabc6255188de91218',files=tokenizer_files),
            runtime_identity=identity(package/'sealed/RUNTIME_PIN.json')),
        reference_checkpoints=dict(scored=checkpoints['O58'],warmup=producers.pop()),
        panel=sealed+'/PANEL.json', panel_seal=sealed+'/SEAL.json',
        pool_audit=sealed+'/POOL_AUDIT.json', warmups=sealed+'/WARMUPS.json',
        warmup_canonicals=sealed+'/WARMUP_CANONICALS.json', schedule=sealed+'/SCHEDULE.json',
        pr54_closeout=sealed+'/PR54_CLOSEOUT.json', runtime_pin=sealed+'/RUNTIME_PIN.json',
        authenticated_files=expected,
        allocation_roots=[str(Path(remote_root).parent),str(Path(remote_source).parent)],
        compiler_baseline=json.loads(Path(compiler_baseline).read_text()),
        projected_peak_bytes=27_000_000_000,
        storage_components_bytes=dict(pr54_archive_and_variants=3_000_000_000,
            source_and_inputs=1_000_000_000,canonical_and_scored_evidence=12_000_000_000,
            compiler_growth=4_000_000_000,analysis_delivery_and_temporaries=7_000_000_000),
        estimate_elapsed_s=14900, estimate_card_s=16300,
        estimate_canonical_elapsed_s=1100, estimate_scored_s=13500,
        serving=dict(block_size=16,max_new_tokens=2048,context_limit=4096,concurrency=1,
                     greedy=True,precision='BF16',attention_backend='fa3',capture=False,
                     cache_policy='idle_full_flush_before_every_request_and_publication'),
        limits=dict(elapsed_s=28800,occupied_card_s=43200,peak_bytes=30_000_000_000,
                    shared_reserve_bytes=400_000_000_000),
        automatic_retry=False, optimizer_steps=0, new_trained_checkpoints=0)
    plan['input_identities'] = {sealed+'/'+p.name:identity(p)
                               for p in sorted((package/'sealed').iterdir()) if p.is_file()}
    plan['binding'] = hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write_json(package/'PLAN.json', plan)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('task','package','remote-root','remote-source','commit','compiler-baseline'):
        parser.add_argument('--'+name, required=True)
    args = vars(parser.parse_args())
    prepare(**args)
