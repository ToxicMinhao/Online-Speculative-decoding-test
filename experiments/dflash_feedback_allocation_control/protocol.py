"""Prespecified P8 budgets and serving contract; no runtime side effects."""
SCHEMA = 'dflash_feedback_allocation_control_v1'
PLANNED = dict(pools=3, optimizer_steps=24, training_presentations=48,
               native_generations=48, response_kl_evaluations=48, explicit_warmups=0)
SERVING_CONTRACT = dict(concurrency=1, greedy=True, temperature=0, top_p=1, top_k=1,
    block_size=10, output_cap=1024, ignore_eos=False, input_limit=31744,
    context_limit=32768, effective_prompt_limit=31735, proposal_reserve=9,
    chunked_prefill_size=8192, feedback_capture=False,
    cache_policy='Cold every request and checkpoint switch',
    tokenizer_binding='Exact qualified model-registry tokenizer',
    runtime_binding='Exact pinned native runtime/source inventory')
