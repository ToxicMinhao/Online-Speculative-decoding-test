import hashlib
import unittest

from experiments.continuation_value_diagnostic.prepare import select_anchors, SELECTION_PREFIX



class AnchorSelectionTest(unittest.TestCase):
    def test_anchor_selection_hashes_all_valid_candidates_then_sorts(self):
        record=dict(sampled_anchors=[dict(positions=[list(range(150))],keep_mask=[[True]*149+[False]])])
        selected,count=select_anchors('source:id',record)
        independently_ranked=sorted(range(149),key=lambda a:(hashlib.sha256(
            (SELECTION_PREFIX+'source:id|'+str(a)).encode('utf-8')).digest(),a))
        assert selected==sorted(independently_ranked[:64])
        assert count==149
        assert selected!=list(range(64))


    def test_short_empty_and_duplicate_anchor_coverage(self):
        record=dict(sampled_anchors=[dict(positions=[[6,4,6,8]],keep_mask=[[True,True,True,False]])])
        assert select_anchors('x',record)==([4,6],2)
        record['sampled_anchors'][0]['keep_mask']=[[False]*4]
        assert select_anchors('x',record)==([],0)
