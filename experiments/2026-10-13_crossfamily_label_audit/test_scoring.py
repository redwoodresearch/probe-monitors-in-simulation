"""Small synthetic checks; never open research data or answer keys."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import score_frozen_models as scoring


def example():
    key, rows, sites = {}, [], {}
    for i in range(33):
        sites[i] = {"site": i, "task": i // 3, "e1": f"reply-{i}", "e2": f"reply-{i}", "sim": f"reply-{i}"}
    for group, count, answer in (("breaking", 33, "differs"), ("control", 18, "same")):
        for i in range(count):
            uid = f"{group}{i}"
            key[uid] = {"kind": "plant", "group": group, "site": i}
            rows.append({"item_id": uid, "answer": answer, "confidence": "sure", "sheet_version": "synthetic"})
    for i in range(33):
        for pair, other in (("e1-sim", "sim"), ("e1-e2", "e2")):
            uid = f"{i}-{pair}"
            h = hashlib.sha256(sites[i]["e1"].encode()).hexdigest()
            key[uid] = {"kind": "main", "group": None, "site": i, "pair": pair,
                        "A": "e1", "B": other, "A_sha256": h, "B_sha256": h}
            rows.append({"item_id": uid, "answer": "same", "confidence": "sure", "sheet_version": "synthetic"})
    return {"items": key, "sheet_version": "synthetic"}, rows, sites


class Checks(unittest.TestCase):
    def test_failed_gate_does_not_load_main(self):
        key, _, _ = example()
        def forbidden():
            raise AssertionError("Main metadata read after failed gate")
        r = scoring.meaning_one([], key, forbidden)
        self.assertFalse(r["primary_read"])
        self.assertEqual(r["p_le"], 1)

    def test_paired_all_same_and_changed_metadata(self):
        key, rows, sites = example()
        r = scoring.meaning_one(rows, key, lambda: sites)
        self.assertTrue(r["gate"]["pass"])
        self.assertEqual(r["difference"], 0)
        self.assertEqual(r["ci95"], [0, 0])
        sites[0]["sim"] = "changed"
        with self.assertRaises(AssertionError):
            scoring.meaning_one(rows, key, lambda: sites)

    def test_holm_keeps_failed_family(self):
        self.assertEqual(scoring.holm({"a": .01, "b": .04, "c": 1, "d": 1}),
                         {"a": .04, "b": .12, "c": 1, "d": 1})

    def test_receipt_cannot_claim_absent_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "freeze.json"
            p.write_text(json.dumps({"files": {}, "coverage": {"judge": {"valid_rows": {"meaning": 1}}}}))
            with self.assertRaises(AssertionError):
                scoring.verified_freeze(p)
            p.write_text(json.dumps({"files": {}, "coverage": {"judge": {"valid_rows": {"meaning": 0}}}}))
            scoring.verified_freeze(p)


if __name__ == "__main__":
    unittest.main()
