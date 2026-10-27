"""Unit tests for the HotpotQA SIM-RAG adapted-controller safety semantics."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_sim_rag_hotpot_adapted import critic_verdict, parse_candidate


class CandidateParsingTests(unittest.TestCase):
    def test_well_formed_candidate(self):
        answer, rationale, valid = parse_candidate(
            "Answer: yes\nRationale: Both people are American."
        )
        self.assertEqual(answer, "yes")
        self.assertEqual(rationale, "Both people are American.")
        self.assertTrue(valid)

    def test_missing_rationale_is_invalid(self):
        answer, rationale, valid = parse_candidate("Answer: yes")
        self.assertEqual(answer, "yes")
        self.assertEqual(rationale, "")
        self.assertFalse(valid)


class CriticVerdictTests(unittest.TestCase):
    def test_only_exact_one_accepts(self):
        self.assertEqual(critic_verdict("1"), (True, "accept_1", False))
        self.assertEqual(critic_verdict(" 1\n"), (True, "accept_1", False))

    def test_blank_is_known_reject_not_accept(self):
        self.assertEqual(critic_verdict(""), (False, "blank_reject", False))
        self.assertEqual(critic_verdict(" \n"), (False, "blank_reject", False))

    def test_zero_is_reject(self):
        self.assertEqual(critic_verdict("0"), (False, "reject_0", False))

    def test_other_output_is_invalid_reject(self):
        self.assertEqual(critic_verdict("maybe"), (False, "other_reject", True))


if __name__ == "__main__":
    unittest.main()
