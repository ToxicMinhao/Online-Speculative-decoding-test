"""Input-only selection and retained K+5 provenance checks."""
from copy import deepcopy
import unittest

from experiments.repair_history_margin.prepare import (
    authenticate_trace, exclusion_rows, match_keys, public_panel, question_from_prompt,
    select_panel, text_digest,
)
from experiments.survival_confirmation.data import SUFFIX, digest


def row(i, **overrides):
    result = dict(request_id=f"request_{i}", source_id=f"gsm8k:main:test:{i}",
        dataset_id=f"gsm8k:main:test:{i}", prompt_ids=[100+i],
        prompt_ids_sha256=f"tokens{i}", rendered_prompt_sha256=f"rendered{i}",
        normalized_question_sha256=f"question{i}", prior_project_execution="documented")
    result.update(overrides)
    return result


class SelectionTests(unittest.TestCase):
    def test_original_order_and_shortfall(self):
        rows = [row(7), row(3), row(9)]
        result = select_panel(dict(requests=rows), [], count=4)
        self.assertEqual([r["source_id"] for r in result["requests"]], [r["source_id"] for r in rows])
        self.assertEqual(result["shortfall"], 1)
        self.assertEqual([r["original_panel_ordinal"] for r in result["requests"]], [0,1,2])
        self.assertNotIn("selection_rank", rows[0])

    def test_excludes_alias_ids_and_each_content_key(self):
        for key in ("source_id", "dataset_id", "prompt_ids_sha256", "rendered_prompt_sha256", "normalized_question_sha256"):
            with self.subTest(key=key):
                candidate = row(0)
                excluded = row(9, reason="training")
                excluded[key] = candidate[key]
                result = select_panel(dict(requests=[candidate,row(1)]), [excluded])
                self.assertEqual([r["request_id"] for r in result["requests"]], ["request_1"])
                self.assertEqual(result["exclusion_reason_counts"], {"training":1})

    def test_deduplicates_before_limit_and_does_not_replace_by_outcomes(self):
        rows = [row(0),row(1,normalized_question_sha256="question0"),row(2),row(3)]
        rows[0]["unfavorable_fake_outcome"] = True
        result = select_panel(dict(requests=rows), [], count=2)
        self.assertEqual([r["request_id"] for r in result["requests"]], ["request_0","request_2"])
        self.assertEqual(result["exclusion_status_counts"], dict(selected=2,duplicate=1,eligible_beyond_limit=1))

    def test_multiple_exclusion_reasons_count_once_per_request(self):
        e1=row(0,reason="training")
        e2=row(0,reason="discovery")
        result=select_panel(dict(requests=[row(0)]),[e1,e2])
        self.assertEqual(len(result["exclusion_audit"][0]["matches"]),2)
        self.assertEqual(result["exclusion_reason_counts"],dict(training=1,discovery=1))

    def test_public_manifest_excludes_token_content(self):
        panel=select_panel(dict(requests=[row(0)]),[])
        result=public_panel(panel)
        self.assertNotIn("prompt_ids",result["requests"][0])
        self.assertIn("prompt_ids",panel["requests"][0])


class PromptTests(unittest.TestCase):
    class Tokenizer:
        def __init__(self,text): self.text=text
        def decode(self,ids,**kwargs): return self.text

    def test_only_known_suffix_removed_and_casefold_mapping(self):
        q="How many APPLES?"
        tokenizer=self.Tokenizer("<|im_start|>user\n"+q+SUFFIX+"<|im_end|>\n<|im_start|>assistant\n")
        values=exclusion_rows([dict(request_id="opaque",prompt_ids=[1])],reason="train",split="train",questions=[q.lower()],tokenizer=tokenizer)
        self.assertEqual(values[0]["source_id"],"gsm8k:main:train:0")
        self.assertEqual(values[0]["normalized_question_sha256"],text_digest(q.casefold()))
        self.assertEqual(values[0]["prompt_ids_sha256"],digest([1]))

    def test_ambiguous_source_is_not_guessed(self):
        tokenizer=self.Tokenizer("<|im_start|>user\nA?"+"<|im_end|>\n<|im_start|>assistant\n")
        with self.assertRaisesRegex(ValueError,"uniquely"):
            exclusion_rows([dict(request_id="x",prompt_ids=[1])],reason="train",split="train",questions=["A?","a?"],tokenizer=tokenizer)

    def test_missing_user_or_conversation_is_rejected(self):
        for text in ("A?", "<|im_start|>user\nA?<|im_end|><|im_start|>user\nB?<|im_end|>"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                question_from_prompt([1], self.Tokenizer(text))


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.request=dict(prompt_ids=[1,2])
        self.canonical=dict(prompt_ids=[1,2],response_ids=[3,151645],finish_reason="eos",verified=True)
        self.trace=dict(checkpoint_id="K5",prompt_ids=[1,2],response_ids=[3,151645],finish_reason="eos",
            max_new_tokens=2048,stop_token_ids=[151645,151643],
            metadata=dict(block_size=16,context_limit=4096,trace=[{}]))

    def test_valid_trace_summary(self):
        self.assertEqual(authenticate_trace(self.trace,self.request,self.canonical,"K5"),dict(rounds=1,output_tokens=2,finish_reason="eos"))

    def test_changed_checkpoint_input_output_or_termination_fails(self):
        for key,value in (("checkpoint_id","later"),("prompt_ids",[8]),("response_ids",[4]),("finish_reason","length"),("max_new_tokens",1024)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                authenticate_trace(dict(self.trace,**{key:value}),self.request,self.canonical,"K5")


if __name__ == "__main__":
    unittest.main()
