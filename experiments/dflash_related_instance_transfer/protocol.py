"""Prespecified read-only transfer diagnostic; importing performs no work."""
SCHEMA = 'dflash_related_instance_transfer_v1'
SAMPLE_MANIFEST_SHA256 = '1411a2b76a2b4c4b6352b1f338f3ce47125ae4b929f7f715876d8a590a787f0d'
QUALIFIED_IMPLEMENTATION_COMMIT = 'fb4f0b9d57b14f0d1b2af1dd015b5356feb89971'
ARMS = ('F', 'S8')
PLANNED = dict(groups=6, children=12, native_generations=24, optimizer_steps=0,
               training_presentations=0, response_kl_evaluations=0,
               new_checkpoints=0, explicit_warmups=0)
SERVING_CONTRACT = dict(concurrency=1, greedy=True, temperature=0, top_p=1, top_k=1,
    dtype='bfloat16', block_size=10, output_cap=1024, ignore_eos=False,
    input_limit=31744, context_limit=32768, effective_prompt_limit=31735,
    proposal_reserve=9, chunked_prefill_size=8192, feedback_capture=False,
    cache_policy='Cold every request and checkpoint switch',
    tokenizer_binding='Exact qualified model-registry tokenizer',
    runtime_binding='Exact pinned native runtime/source inventory',
    independent_requests=True, parent_history=False,
    checkpoint_helpers_refreshed=True, reference='fresh_child_F',
    explicit_generation_warmups=0, normal_runtime_initialization=True)
PARENT_SELECTION = dict(sort='retained F output-token length ascending',
                        tie_break='original manifest order', ranks=[2, 3],
                        selected_before_construction=True)
EXPECTED_PARENT_SUFFIXES = {
    'gsm8k_00': ('01', '02'), 'gsm8k_01': ('00', '01'),
    'codealpaca_00': ('03', '00'), 'codealpaca_01': ('00', '03'),
    'sharegpt_00': ('01', '03'), 'sharegpt_01': ('01', '02'),
}
SEMANTIC_REVIEW_FIELDS = (
    'clarity', 'constructable', 'language_preserved', 'task_family_preserved',
    'format_preserved', 'complexity_comparable', 'answer_relevant_change',
    'not_paraphrase_of_original48', 'not_paraphrase_of_other_children')
