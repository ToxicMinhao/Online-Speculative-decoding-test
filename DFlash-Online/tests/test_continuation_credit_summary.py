"""Denominator checks for reporting capped, uncapped and no-error outcomes."""
from experiments.continuation_credit_cpu.__main__ import summarize_outcomes


def test_no_error_regressions_stay_in_progress_but_not_repair_rates():
    common = dict(response_ordinal=0, m=5, L=2, K=3, Y_full=2, Y_old=1,
                  beyond_K=1, crossed_next=True, useful_repair=True, lost_prefix=0,
                  delta_u=.1, u_old=.2, u_shared=.3, high=True, has_error=True)
    no_error = dict(common, high=False, has_error=False, L=5, K=5, Y_full=-3,
                    Y_old=-3, beyond_K=0, crossed_next=False, useful_repair=None,
                    lost_prefix=3, delta_u=None, u_old=None, u_shared=None)
    all_rows, high, low, no_error_summary = summarize_outcomes([common, no_error], 'native')
    assert all_rows['Y_full'] == -1
    assert all_rows['Y_old'] == -2
    assert all_rows['useful_repairs'] == all_rows['repair_denominator'] == 1
    assert all_rows['crossed_next'] == all_rows['crossed_next_denominator'] == 1
    assert all_rows['lost_prefix'] == 3
    assert all_rows['old_prefix_occurrences'] == 7
    assert high['blocks'] == no_error_summary['blocks'] == 1
    assert low['blocks'] == 0
