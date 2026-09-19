"""Unit tests for the Constitution RAG backend (no network, no model download).

Run with:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.prompt import _extract_cited_articles, _verify_citations
from app.retriever import _passes_distance_filter, _reciprocal_rank_fusion
from ingestion.chunk_by_article import is_footnote


class TestExtractCitedArticles(unittest.TestCase):
    """Citation parsing: ranges, sub-articles, and numbering edge cases."""

    def test_simple_citation(self):
        self.assertEqual(_extract_cited_articles("Article 21 says no."), ["21"])

    def test_abbreviated_form(self):
        self.assertEqual(_extract_cited_articles("Per Art. 32, remedies exist."), ["32"])

    def test_two_citations(self):
        self.assertEqual(
            _extract_cited_articles("Per Articles 14 and 21, both apply."),
            ["14", "21"],
        )

    def test_dedupes_and_preserves_order(self):
        self.assertEqual(
            _extract_cited_articles("Article 21 first, then Article 19 and Article 21 again."),
            ["21", "19"],
        )

    def test_sub_article_does_not_become_bare_number(self):
        self.assertEqual(
            _extract_cited_articles("Article 19(1)(g) is the answer."), ["19"]
        )

    def test_compound_suffix_preserved(self):
        self.assertEqual(
            _extract_cited_articles("Article 243ZG bars court interference."), ["243ZG"]
        )

    def test_compound_suffix_beats_truncation(self):
        self.assertEqual(
            _extract_cited_articles("Article 243ZG, unlike Article 243Z, applies."),
            ["243ZG", "243Z"],
        )

    def test_same_prefix_distinct_articles_both_kept(self):
        self.assertEqual(
            _extract_cited_articles("Article 21 and Article 21A differ."), ["21", "21A"]
        )

    def test_ignores_numbers_without_article_word(self):
        self.assertEqual(_extract_cited_articles("Section 21 is irrelevant."), [])

    def test_mixed_case(self):
        self.assertEqual(_extract_cited_articles("article 14 applies."), ["14"])


class TestVerifyCitations(unittest.TestCase):
    """Citation verification: hallucination detection against retrieved context."""

    def test_all_verified(self):
        verified, unverified = _verify_citations(["21"], {"21"})
        self.assertEqual(verified, ["21"])
        self.assertEqual(unverified, [])

    def test_hallucinated_citation_flagged(self):
        verified, unverified = _verify_citations(["99"], {"21"})
        self.assertEqual(verified, [])
        self.assertEqual(unverified, ["99"])

    def test_mixed(self):
        verified, unverified = _verify_citations(["21", "99", "21A"], {"21"})
        self.assertEqual(verified, ["21"])
        self.assertEqual(unverified, ["99", "21A"])


class TestDistanceFilter(unittest.TestCase):
    """Distance guardrail: keyword-only hits exempt, vector distances enforced."""

    def test_vector_within_threshold_passes(self):
        self.assertTrue(_passes_distance_filter({"distance": 0.5, "keyword_only": False}))

    def test_vector_beyond_threshold_fails(self):
        self.assertFalse(_passes_distance_filter({"distance": 0.9, "keyword_only": False}))

    def test_keyword_only_always_passes(self):
        self.assertTrue(_passes_distance_filter({"distance": 1.0, "keyword_only": True}))

    def test_custom_threshold(self):
        self.assertFalse(
            _passes_distance_filter({"distance": 0.5, "keyword_only": False}, threshold=0.4)
        )


class TestReciprocalRankFusion(unittest.TestCase):
    """RRF fusion: ranking, deduplication, and keyword-only bookkeeping."""

    def _chunk(self, num: str, distance: float, source: str = "main_body") -> dict:
        return {
            "article_number": num,
            "source": source,
            "text": f"Text {num}",
            "distance": distance,
        }

    def test_chunk_in_both_lists_wins(self):
        c = self._chunk("21", 0.4)
        fused = _reciprocal_rank_fusion([c], [dict(c)])
        self.assertEqual(fused[0]["article_number"], "21")
        self.assertFalse(fused[0]["keyword_only"])

    def test_deduplicates_same_chunk_from_both_channels(self):
        c = self._chunk("21", 0.4)
        fused = _reciprocal_rank_fusion([c], [dict(c)])
        self.assertEqual(len(fused), 1)

    def test_keyword_only_flag_set_when_absent_from_vector_results(self):
        vec = self._chunk("21", 0.4)
        bm25 = self._chunk("44", 1.0)
        fused = _reciprocal_rank_fusion([vec], [bm25])
        by_num = {c["article_number"]: c for c in fused}
        self.assertTrue(by_num["44"]["keyword_only"])
        self.assertFalse(by_num["21"]["keyword_only"])

    def test_sorted_by_fused_score(self):
        vec = self._chunk("21", 0.4)
        bm25 = self._chunk("44", 1.0)
        fused = _reciprocal_rank_fusion([vec], [bm25])
        self.assertEqual(fused[0]["article_number"], "21")

    def test_main_body_beats_amendment_for_same_number(self):
        vec = self._chunk("21", 0.4)
        amend = self._chunk("21", 1.0, source="amendment_act")
        fused = _reciprocal_rank_fusion([vec], [amend])
        # Same RRF score; distance tiebreaker prefers the main-body chunk
        self.assertEqual(fused[0]["source"], "main_body")


class TestIsFootnote(unittest.TestCase):
    """Chunker footnote heuristics: insertion/substitution notes are not articles."""

    def test_substitution_marker(self):
        self.assertTrue(is_footnote("Subs. by Act 45", "w.e.f. 3-1-1977"))

    def test_insertion_marker(self):
        self.assertTrue(is_footnote("Ins. by Act 45", "w.e.f. 1-4-1978"))

    def test_real_article_passes(self):
        self.assertFalse(
            is_footnote("Protection of life and personal liberty", "No person shall be deprived...")
        )


if __name__ == "__main__":
    unittest.main()
