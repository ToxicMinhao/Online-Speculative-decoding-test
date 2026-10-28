import copy
import unittest
from campaign import check_event


class EventAccountingTests(unittest.TestCase):
    def receipt(self):
        return dict(status='completed', objective='osd_first_error', passes=2,
                    response_presentations=256, optimizer_steps=128, start_step=0, end_step=128,
                    features=[{'path': 'retained'}], records=[dict(record_id=str(i // 2),
                    **{'pass': i // 128}, sampled_anchors=[[1, 4]], optimizer_stepped=bool(i % 2),
                    optimizer_step_after=(i + 1) // 2) for i in range(256)])

    def test_exact_original_schedule(self):
        receipt = self.receipt()
        check_event(receipt, copy.deepcopy(receipt))

    def test_reject_anchor_drift(self):
        receipt = self.receipt()
        original = copy.deepcopy(receipt)
        receipt['records'][64]['sampled_anchors'] = [[2, 4]]
        with self.assertRaises(ValueError):
            check_event(receipt, original)

    def test_reject_wrong_step_count(self):
        receipt = self.receipt()
        receipt['end_step'] = 129
        with self.assertRaises(ValueError):
            check_event(receipt, receipt)

    def test_reject_extra_presentations(self):
        receipt = self.receipt()
        original = copy.deepcopy(receipt)
        receipt['records'].append(receipt['records'][0])
        with self.assertRaises(ValueError):
            check_event(receipt, original)

class GuardAccountingTests(unittest.TestCase):
    def test_exact_and_missing_probe(self):
        from campaign import check_guard_report
        report = dict(work=dict(current_response_forwards=256, h_backward_calls=256,
                               k_backward_calls=256, probe_response_forwards=18),
                      steps=[dict(optimizer_step=i) for i in range(1, 129)],
                      probes=[dict(optimizer_step=i, candidate=c, responses=[{}, {}])
                              for i in (1, 64, 128)
                              for c in ('ordinary_K', 'projected_G', 'norm_scaled_K')])
        check_guard_report(report)
        report['probes'].pop()
        with self.assertRaises(ValueError):
            check_guard_report(report)


if __name__ == '__main__':
    unittest.main()
