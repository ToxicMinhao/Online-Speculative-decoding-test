"""Fixed ordinary-OSD half-learning-rate campaign boundaries."""

ARM = 'O_HALF'
ARMS = (ARM,)
SEEDS = (17, 29)
EVENTS = (1, 2, 3, 4)
OBJECTIVE = 'osd'
LEARNING_RATE = 1.5e-4
HORIZON = 7424
CONTRACT_KEY = 'o_half_matched_lr_v1'
UPSTREAM_BINDING = '6def4832d12698fc2733e91c427db77558f3fe50e7f51063e9c2a11f9bc961af'
PANEL_BINDING = '8a2d3857a54c069bce655f5dda107ce0bd6ec30ebbc2b72dce20bc8a7d05295f'
ORIGINAL_COMMIT = '2705d992b133196be162229e1d5a08dd9c4f9cbb'
KP_COMMIT = 'd10521a1aa7d256bb75b37a6aae07586ae7cc8b3'
NEW_BUDGET = dict(optimizer_steps=1024, presentations=2048, evaluation=1072,
                  bridge=48, evaluation_main=1024, feedback=0,
                  terminal_teacher_forwards=0, reference_forwards=0,
                  qualification_optimizer_steps=0)
EVENT_WORK = dict(forwards=256, backwards=256, optimizer_steps=128)


def validate_plan(plan):
    """Check training bounds; the evaluator validates its own exact slot list."""
    if plan['new_budget'] != NEW_BUDGET:
        raise ValueError('The bounded O_HALF work budget changed')
    binding = plan['training_binding']
    if (binding['upstream_data_binding'] != UPSTREAM_BINDING
            or binding['evaluation_binding'] != PANEL_BINDING):
        raise ValueError('Original feedback or reused-panel binding changed')
    if (plan['training_seeds'] != list(SEEDS) or plan['stop_step'] != 512
            or plan['inherited_horizon'] != HORIZON
            or plan['no_automatic_retry'] is not True):
        raise ValueError('Preserve both seeds, four events and the native horizon')
    if plan['original_execution_commit'] != ORIGINAL_COMMIT:
        raise ValueError('Original coding execution differs')
    if set(plan['configs']) != {'17', '29'}:
        raise ValueError('Require both original seeds')
    for seed in SEEDS:
        key = str(seed)
        if set(plan['configs'][key]) != {ARM}:
            raise ValueError('Only O_HALF may be trained')
        if (set(plan['pools'][key]) != {'1', '2', '3', '4'}
                or set(plan['original_schedules'][key]) != {'1', '2', '3', '4'}):
            raise ValueError('Require all four original pools and natural schedules')
        algorithm = plan['original_algorithm_contracts'][key]
        if algorithm['online_target_objective']['name'] != OBJECTIVE:
            raise ValueError('Initialization must be original O, not K')
        if plan['original_optimizer_recipes'][key]['optimizer_options']['lr'] != 3e-4:
            raise ValueError('Original native Adam recipe differs')
    source = plan['execution_source']
    if len(source['commit']) != 40 or source['dirty']:
        raise ValueError('Require the clean sealed execution revision')
    return dict(status='passed', budget=NEW_BUDGET, training_objective=OBJECTIVE,
                qualification_model_work=0)
