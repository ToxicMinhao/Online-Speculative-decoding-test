"""CPU-only input privacy, fixed ranking, narrow loader and schedule contracts."""
from copy import deepcopy
import hashlib
import unittest
from experiments.unchanged_c_mathqa_transfer import data as D


def row(i):
    question=f"Question {i}?";options="a) 1, b) 2";prompt=f"user\n{question}\nOptions: {options}\nassistant"
    result=dict(source_id=f"mathqa:test:{i}",source_row=i,question=question,options=options,
                user_message=question+"\n\nOptions: "+options,rendered_prompt=prompt,prompt_ids=[1,i+10],prompt_tokens=2)
    result.update(D.fingerprints(question,options,prompt,result["prompt_ids"]))
    result["selection_key"]=D.rank_key(result["source_id"],result["normalized_content_sha256"])
    return result


def adaptation():
    return dict(start_ordinal=512,stop_ordinal=768,train_requests=[
        dict(source_id=f"mathqa:train:{i}",request_id=f"train_{i:05d}",request_index=i+1,
             prompt_ids=[2,i+1000],prompt_ids_sha256=D.token_digest([2,i+1000]),
             max_new_tokens=2048,stop_token_ids=[151645,151643],collect_feedback=True)
        for i in range(512,768)])


class DataTests(unittest.TestCase):
    def test_normalization_preserves_case_punctuation_and_options_order(self):
        self.assertEqual(D.normalize("  Ａ\t b\n ?  "),"A b ?")
        self.assertNotEqual(D.normalize("A?"),D.normalize("a?"))
        self.assertNotEqual(D.fingerprints("Q","a,b","p",[1])["normalized_content_sha256"],
                            D.fingerprints("Q","b,a","p",[1])["normalized_content_sha256"])

    def test_literal_hash_ranking_and_outcomes_unused(self):
        records=[row(i) for i in range(260)]
        expected=sorted(records,key=lambda r:(hashlib.sha256(f"kplus_unchanged_c_mathqa_transfer_20261117_v1|{r['source_id']}|{r['normalized_content_sha256']}".encode()).hexdigest(),r["source_id"]))
        changed=deepcopy(records)
        for i,r in enumerate(changed):r["performance"]=999-i;r["answer"]="never select from this"
        self.assertEqual([r["source_id"] for r in D.rank_eligible(changed)],[r["source_id"] for r in expected])

    def test_exact256_plus2_seal_and_narrow_validation(self):
        result=D.seal_requests([row(i) for i in range(260)])
        self.assertEqual((len(result["scored"]),len(result["warmup"])),(256,2))
        self.assertEqual(D.validate_manifests({"requests":result["scored"]},{"requests":result["warmup"]},adaptation())["status"],"passed")
        self.assertEqual([r["request_index"] for r in result["warmup"]],[0,1])

    def test_shortfall_stops_without_replacement(self):
        with self.assertRaisesRegex(ValueError,"258"):D.seal_requests([row(i) for i in range(257)])

    def test_tampered_fingerprint_or_old_panel_contract_is_rejected(self):
        result=D.seal_requests([row(i) for i in range(258)])
        result["scored"][0]["prompt_ids"].append(999)
        with self.assertRaisesRegex(ValueError,"fingerprint"):D.validate_manifests({"requests":result["scored"]},{"requests":result["warmup"]},adaptation())

    def test_adaptation_cannot_silently_use_other_ordinals(self):
        result=D.seal_requests([row(i) for i in range(258)]);train=adaptation();train["train_requests"][0]["request_id"]="train_00000"
        with self.assertRaisesRegex(ValueError,"stream"):D.validate_manifests({"requests":result["scored"]},{"requests":result["warmup"]},train)

    def test_source_and_text_token_history_matches_ignore_answers(self):
        rows=[row(i) for i in range(3)]
        self.assertEqual(set(D.history_matches({"source_id":"mathqa_test:0","prompt_ids":rows[1]["prompt_ids"],"question":rows[2]["question"]},rows)),{0,1,2})
        self.assertEqual(D.history_matches({"answer":rows[0]["question"],"response":{"prompt_ids":rows[1]["prompt_ids"]},"exclusions":[{"source_id":rows[2]["source_id"]}]},rows),{})

    def test_renamed_training_id_catches_cross_split_content(self):
        rows=[row(0)]
        self.assertEqual(set(D.history_matches({"source_id":"mathqa:train:27"},rows,{"mathqa:train:27":(rows[0]["question"],rows[0]["options"])})),{0})

    def test_schedule_has36_balanced_activations2304_scores72_warmups(self):
        from collections import Counter
        requests=D.seal_requests([row(i) for i in range(258)])["scored"]
        schedule=D.evaluation_schedule(requests)
        self.assertEqual(len(schedule),36)
        self.assertEqual(sum(len(s["request_ordinals"]) for s in schedule),2304)
        self.assertEqual(2*len(schedule),72)
        groups=[schedule[i:i+3] for i in range(0,36,3)]
        self.assertEqual(Counter(tuple(s["method"] for s in g) for g in groups),
                         {permutation:2 for permutation in D.PERMUTATIONS})
        for method in D.METHODS:
            self.assertEqual(Counter(s["position"] for s in schedule if s["method"]==method), {0:4,1:4,2:4})
        for ordinal,group in enumerate(groups):
            self.assertEqual(tuple(s["method"] for s in group),D.PERMUTATIONS[ordinal%6])
            self.assertTrue(all(s["request_ordinals"]==group[0]["request_ordinals"] for s in group))

    def test_historical_request_namespace_cannot_enter_new_panel(self):
        roles=D.seal_requests([row(i) for i in range(258)])
        rows=roles["scored"]
        rows[0]["request_id"]=f"mathqa-lr-transfer-{rows[0]['source_row']:04d}"
        with self.assertRaisesRegex(ValueError,"identity"):
            D.validate_manifests({"requests":rows},{"requests":roles["warmup"]},adaptation())


if __name__=="__main__":unittest.main()
