"""CPU receipt audit; no production analysis imports and no model execution."""
from pathlib import Path
import json, collections, math, hashlib
BASE=Path(__file__).resolve().parents[2]
RAW=BASE/'delivery/execution_003/raw'
OUT=Path(__file__).resolve().parent
checks=0

def read(name):return json.loads((RAW/name).read_text())
def check(value,why):
 global checks
 checks+=1
 if not value:raise AssertionError(why)
def passed(d,why):
 check(d['status'] in ('passed','completed'),why)
 check(not d.get('mismatches'),why+' mismatches')
def rows(name):
 with (RAW/name).open() as f:
  for line in f:yield json.loads(line)

status=read('STATUS.json');passed(status,'campaign');check((RAW/'EXIT_CODE').read_text().strip()=='0','exit')
plan=read('sealed/PLAN.json');check(status['execution_commit']==plan['execution_commit']=='5d4e0ff65c5976318a50ebb181923447068169c8','source')
gate=read('gate0/GATE0.json');passed(gate,'Gate0');check(gate['cpu_only'] and not gate['cuda_initialized'] and gate['gpu_work']==gate['optimizer_steps']==0,'cpu-only')
serializers=list((RAW/'gate0').glob('*_SERIALIZER.json'));check(len(serializers)==6,'six serializers')
for p in serializers:
 d=json.loads(p.read_text());passed(d,p.name);check(d['original_identity']==d['reproduced_identity'],p.name+' hash');check(d['tensors']==58 and d['new_copy_removed_after_pass'],p.name+' release')
for w in [8,128]:
 d=read(f'gate0/E_{w:03d}_NORM.json');passed(d,'E norm');check(abs(d['measured']['difference_norm']-d['retained_norm'])<=1e-12+1e-10*abs(d['retained_norm']),'E tolerance')
qualification=read('training/qualification/RESULT.json');passed(qualification,'qualification');check(len(qualification['cases'])==6,'six cases')
for case in qualification['cases']:
 passed(case,case['case']);passed(case['prestate'],case['case']+' initial state');check(case['update']['optimizer_applications']==1,'one Adam')
case_by={x['case']:x for x in qualification['cases']}
for n in ['alpha_1','alpha_1_repeat']:check(case_by[n]['displacement']==case_by['N']['displacement'],'alpha1 displacement')
check(case_by['alpha_half']['displacement']==case_by['alpha_half_repeat']['displacement'],'half repeat')
check(case_by['alpha_0']['displacement']['difference_norm']==case_by['alpha_0']['displacement']['changed_entries']==0,'zero neutrality')
restored=read('training/qualification/RESTORED_ORIGIN.json');passed(restored,'origin');check(restored['global_step']==768 and restored['presentations']==1536 and restored['exact_weights_adam_rng_gradients_runtime'],'origin exact')
windows=read('training/WINDOWS.json');check([d['window'] for d in windows]==list(range(8,129,8)),'16 windows')
trial_count=0;identity_mode=collections.Counter();bracket_mismatch=[]
for w in range(8,129,8):
 d=read(f'training/window_{w:03d}/RESULT.json');passed(d,'window');check(d['trials']<=10,'trial bound')
 check(d['lower_norm']<=d['target_norm']<=d['upper_norm'],'bracket sides')
 for x in ['lower_norm','upper_norm']:
  rel=abs(d[x]-d['target_norm'])/d['target_norm'];bracket_mismatch.append(rel);check(rel<=.0025,'bracket precision')
 for state in ['P','N']:
  p=read(f'training/window_{w:03d}/{state}_IDENTITY.json');passed(p,'identity');identity_mode[p['mode']]+=1
  if 'reproduced_identity' in p:check(p['reproduced_identity']==p['original_identity'],'exact original-file hash')
  else:check(p['tensors']==58 and not p['mismatches'],'exact native tensors')
 passed(read(f'training/window_{w:03d}/RESTORED_NATIVE.json'),'restored N')
 if w in (8,128):passed(read(f'training/window_{w:03d}/WITNESS_PARITY.json'),'boundary witness')
 trials=sorted((RAW/f'training/window_{w:03d}/trials').glob('TRIAL_*.json'));check(len(trials)==d['trials'],'raw trial count')
 for p in trials:
  t=json.loads(p.read_text());passed(t,'trial');trial_count+=1
  check(t['optimizer_applications']==1 and all(t[k]==0 for k in ['response_forwards','response_backwards','response_presentations','sampling_draws']),'cached work')
  for key in ['restored_pre','moment_clip_option_parity','cached_state_semantics','restored_native']:passed(t[key],key)
  check(t['adam_observation']['clipped_gradients']==d['native_update']['adam_observation']['clipped_gradients'],'exact clipped gradient signatures')
  check(t['clip_norm']==d['native_update']['combined_native_clip_norm'] and t['clip_factor']==d['native_update']['clip_factor'],'clip equality')
  check(t['restored_group_lrs']==t['original_lrs']==[.0003],'LR restored')
  check(t['requested_lrs']==[.0003*t['alpha']],'effective LR')
check(trial_count==149,'149 trials')
seal=read('training/CONTROL_SEAL.json');check(seal['status']=='sealed' and seal['scoring_started'] is False,'pre-outcome seal');check(len(seal['exports'])==32,'32 exports')
for exp in seal['exports']:
 check(exp['seal']=='weights_only_norm_selected_before_scoring','norm only selection');check(exp['relative_mismatch']<=.0025,'sealed bracket');check(exp['tensors']==58,'58 tensors')
parity=read('training/FINAL_NATIVE_PARITY.json');passed(parity,'final');check(parity['global_step']==896 and parity['presentations']==1792 and parity['weights_optimizer_counters_rng_exact'],'complete final state')
# Recount start/completion pairs; summary fields alone are not accepted.
training_counts=collections.Counter();training_categories=collections.Counter();training_open={};physical=collections.Counter()
for row in rows('training/PHYSICAL_WORK.jsonl'):
 if row['phase']=='ledger_closed':continue
 key=(row['serial'],row['kind'])
 if row['phase']=='started':check(key not in training_open,'duplicate training ticket');training_open[key]=row
 elif row['phase']=='completed':
  check(key in training_open,'training completion without attempt');start=training_open.pop(key)
  training_counts[row['kind']]+=1;training_categories[(start['category'],row['kind'])]+=1
  # Physical fields are cumulative snapshots, not per-operation increments.
  for name,value in row.get('physical',{}).items():
   check(value>=physical[name], 'monotone physical counters');physical[name]=value
 else:check(False,'failed/unknown training work')
check(not training_open,'training pending');check(training_counts=={'response_forward':260,'response_backward':260,'optimizer_step':283},'training totals')
check(training_categories[('native','optimizer_step')]==128 and training_categories[('qualification','optimizer_step')]==6,'native/qualification Adam')
check(training_counts['optimizer_step']<=294 and training_counts['response_forward']<=260 and training_counts['response_backward']<=260,'caps')
check(physical==read('WORK_AND_COSTS.json')['training']['physical'],'physical instrumentation totals')
counts=collections.Counter();native_open={};first_draft_time=None;native_lines=0
for row in rows('NATIVE_WORK.jsonl'):
 native_lines+=1
 if row.get('status')=='attempted':
  check(row['ticket'] not in native_open,'duplicate native ticket');native_open[row['ticket']]=row
  if row['kind']=='draft_block' and first_draft_time is None:first_draft_time=row['gpu_seconds']
 elif row.get('status')=='completed':
  check(row['ticket'] in native_open,'native completion without attempt');start=native_open.pop(row['ticket']);check(row['kind']==start['kind'],'native kind');counts[row['kind']]+=1
 elif row.get('status') in ('failed','unknown'):check(False,'native failure/unknown')
check(not native_open,'native pending');check(counts==status['budget']['completed'],'native totals');check(status['budget']['gpu_seconds']<10800,'180 card min')
check(first_draft_time>=status['phases']['qualification_and_native_calibration']['wall_s'],'scoring follows calibration phase')
probe=read('probes/STATUS.json');passed(probe,'probes');check(probe['requests_reconstructed']==64,'64 histories');check(probe['new_logical_control_rows']==probe['physical_control_blocks']==probe['control_attempts']==8192,'8192 outcomes')
check(probe['neutral_blocks']==8 and probe['sentinel_blocks']==24,'32 qualifier blocks')
check(all(probe[k]==0 for k in ['invalid_control_rows','unknown_control_rows','unattempted_logical_control_rows','unattempted_sentinel_blocks','unattempted_neutral_blocks','new_generations']),'complete valid budget')
for d in probe['sentinel_checks']:passed(d,'sentinel');check(not d['score_differences'] and not d['differing_fields'],'exact sentinel')
alloc=list(rows('ALLOCATION.jsonl'))
for d in alloc:passed(d,'storage check');check(d['projected_new_bytes']<=d['cap_bytes']==110_000_000_000,'new allocation cap');check(d['remaining_free_bytes']>=d['reserve_bytes']==400_000_000_000,'reserve')
post=json.loads((BASE/'delivery/execution_003/POSTFLIGHT_ALLOCATION.json').read_text());check(post['new_allocated_bytes']<110_000_000_000 and post['free_bytes']>400_000_000_000,'postflight storage')
training=read('training/RESULT.json');check(training['source_fitting_forwards']==training['complete_response_generations']==training['teacher_feature_preparation']==0,'no extra model work')
retirement=[dict(path=x['checkpoint']+'/model.safetensors',identity=x['weights_identity'],classification='TEMP_CONTROL',release_condition='Final raw-token/statistical audit and report verified; unique current-created file and no alias into KEEP input') for x in seal['exports']]
(OUT/'CONTROL_RETIREMENT_CANDIDATES.json').write_text(json.dumps(retirement,indent=2)+'\n')
result=dict(status='passed',checks=checks,scope='CPU reconciliation of retained execution receipts and ledgers; no new model execution',execution_commit=status['execution_commit'],gate0_cpu_wall_s=gate['cpu_wall_s'],cpu_preparation_elapsed_s=499,prior_cpu_administrative_failure_s=76,prior_cpu_failure_gpu_or_adam=0,qualification_cases=6,windows=16,controls=32,calibration_trials=149,training_counts=dict(training_counts),training_categories={str(k):v for k,v in training_categories.items()},physical_counts=dict(physical),native_ledger_lines=native_lines,native_counts=dict(counts),scored_endpoint_blocks=8192,extra_endpoint_blocks=32,first_native_draft_gpu_seconds=first_draft_time,identity_modes=dict(identity_mode),maximum_relative_bracket_error=max(bracket_mismatch),native_state_restored_and_final_exact=True,occupied_card_s=status['budget']['gpu_seconds'],campaign_wall_s=status['wall_s'],qualification_training_wall_s=status['phases']['qualification_and_native_calibration']['wall_s'],scoring_wall_s=status['phases']['fixed_native_scoring']['wall_s'],storage_admission_samples=len(alloc),largest_admission_projection_bytes=max(x['projected_new_bytes'] for x in alloc),observed_postflight_new_allocated_bytes=post['new_allocated_bytes'],postflight_free_bytes=post['free_bytes'],peak_scope='Admission samples and postflight observed allocation, not continuous storage high-water measurement',retirement_control_bytes=sum(x['identity']['bytes'] for x in retirement),retirement_performed=False,limitations=['Exact GPU tensor/moment comparisons rely on retained runtime parity receipts; CPU audit rechecks their identities and consistency, not reruns absent GPU states.','The two newly published native endpoint/recovery payloads are not included in the 32-control retirement recommendation; compare full retained native identity and retention requirements separately.','Earlier failed CPU preparation consumed76s administrative work and zero GPU/Adam; prior staging is included in allocation roots.'])
(OUT/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
