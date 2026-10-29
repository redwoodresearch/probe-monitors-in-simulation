"""Load-bearing checks: failed responses cannot become labels or free usage."""
import unittest
import tempfile
from pathlib import Path
from probe_shift.blind_crossfamily_audit import classify, reconcile, packed, digest, body_for, write_new

MODEL = {"api": "openai", "input_per_million": 4, "output_per_million": 20,
         "short_context_limit": 272000, "long_input_multiplier": 2, "long_output_multiplier": 1.5}


def response(text):
    return {"status": "completed", "usage": {"input_tokens": 100, "output_tokens": 30},
            "output": [{"content": [{"type": "output_text", "text": text}]}]}


class AuditTests(unittest.TestCase):
    def test_missing_usage_does_not_score(self):
        raw = response('{"same_meaning":true,"confidence":"sure"}')
        raw.pop("usage")
        status, label, cost, _ = classify(MODEL, raw, 200, {"kind": "meaning"})
        self.assertEqual((status, label, cost), ("usage_unknown", None, None))

    def test_duplicate_or_missing_sabotage_ids_rejected(self):
        raw = response('{"labels":[{"id":"c0","label":"not_sabotage","confidence":"sure"},'
                       '{"id":"c0","label":"sabotage","confidence":"sure"}]}')
        status, label, cost, _ = classify(MODEL, raw, 200,
                                         {"kind": "sabotage", "expected_ids": ["c0", "c1"]})
        self.assertEqual((status, label), ("schema_failure", None))
        self.assertGreater(cost, 0)

    def test_truncation_and_refusal_are_not_negative_labels(self):
        raw = response('{"same_meaning":false,"confidence":"sure"}')
        raw["status"] = "incomplete"
        self.assertEqual(classify(MODEL, raw, 200, {"kind": "meaning"})[0], "incomplete")
        raw["status"] = "completed"
        raw["output"] = [{"content": [{"type": "refusal"}]}]
        self.assertEqual(classify(MODEL, raw, 200, {"kind": "meaning"})[:2], ("refusal", None))

    def test_crash_after_receiving_response_recovers_without_resubmission(self):
        model = MODEL | {"name": "sol", "id": "test-model"}
        cfg = {"models": [model], "max_output_tokens": 100}
        job = {"kind": "meaning", "id": "test-item", "prompt": "test", "text": "test"}
        h = digest(packed({"api": "openai", "body": body_for(model, job, cfg)}).encode())
        meta = {"model": "sol", "kind": "meaning", "job_id": "test-item", "request_sha256": h, "git": "test"}
        cached = {"metadata": meta, "response": response('{"same_meaning":true,"confidence":"sure"}'),
                  "http_status": 200, "transport_error": None, "reservation": 1.0}
        state = {"charged_or_reserved_usd": 0, "pending": {h: 1.0}, "completed": {}, "stopped": {}}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_new(root / "raw" / (h + ".json"), cached)
            reconcile(root, state, cfg, [job], {"git": "test"})
            self.assertEqual(state["pending"], {})
            self.assertEqual(state["completed"][h]["status"], "ok")
            self.assertTrue((root / "responses" / (h + ".json")).exists())
            self.assertAlmostEqual(state["charged_or_reserved_usd"], 0.001)


if __name__ == "__main__":
    unittest.main()
