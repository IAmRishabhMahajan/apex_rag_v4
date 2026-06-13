"""Real unit tests for US-006 Complex Query Reasoning Path implementation."""

from __future__ import annotations

import unittest

from src.apex_rag.complex_reasoning import (
    ClaimStatus,
    EdgeRelationship,
    build_claim_graph,
    compress_context,
    decompose_query,
    is_complex,
    link_evidence_to_claims,
    run_complex_reasoning,
)
from src.apex_rag.evidence_fusion import (
    CitationMetadata,
    EvidenceBundle,
    EvidenceItem,
)
from src.apex_rag.query_intelligence import build_query_profile


def _citation(source_id: str = "src-1") -> CitationMetadata:
    return CitationMetadata(
        source_id=source_id,
        title="Doc",
        url="https://example.com",
        retrieval_expert="policy",
        retrieval_query="q",
    )


def _item(content: str, source_id: str = "src-1") -> EvidenceItem:
    return EvidenceItem(content=content, citation=_citation(source_id), claim_ids=())


def _bundle(*items: EvidenceItem) -> EvidenceBundle:
    return EvidenceBundle(items=list(items), query_id="q1")


class TestComplexityGate(unittest.TestCase):
    def test_simple_fact_lookup_is_not_complex(self) -> None:
        profile = build_query_profile("What is idempotency?")
        self.assertFalse(is_complex(profile))

    def test_investigation_query_is_complex(self) -> None:
        profile = build_query_profile("Why did the deployment pipeline fail last night?")
        self.assertTrue(is_complex(profile))

    def test_comparison_query_is_complex(self) -> None:
        profile = build_query_profile("Compare Python versus Java for data pipelines")
        self.assertTrue(is_complex(profile))

    def test_analysis_query_is_complex(self) -> None:
        profile = build_query_profile("Analyze the performance breakdown of the API gateway")
        self.assertTrue(is_complex(profile))

    def test_keyword_why_triggers_complex(self) -> None:
        profile = build_query_profile("Why does caching improve read latency?")
        self.assertTrue(is_complex(profile))

    def test_keyword_root_cause_triggers_complex(self) -> None:
        profile = build_query_profile("Find the root cause of the memory leak")
        self.assertTrue(is_complex(profile))


class TestClaimDecomposition(unittest.TestCase):
    def test_investigation_produces_three_claims(self) -> None:
        profile = build_query_profile("Why did the deployment fail after the release?")
        claims = decompose_query(profile)
        self.assertEqual(len(claims), 3)

    def test_comparison_produces_three_claims(self) -> None:
        profile = build_query_profile("Compare Python versus Java for data pipelines")
        claims = decompose_query(profile)
        self.assertEqual(len(claims), 3)

    def test_all_claims_start_pending(self) -> None:
        profile = build_query_profile("Why did the deployment fail?")
        claims = decompose_query(profile)
        for claim in claims:
            self.assertEqual(claim.status, ClaimStatus.PENDING)

    def test_claims_contain_original_query_text(self) -> None:
        profile = build_query_profile("Why did the deployment fail?")
        claims = decompose_query(profile)
        for claim in claims:
            self.assertIn("deployment", claim.text)

    def test_analysis_claims_cover_components(self) -> None:
        profile = build_query_profile("Analyze the performance breakdown of the system")
        claims = decompose_query(profile)
        combined = " ".join(c.text for c in claims).lower()
        self.assertIn("components", combined)


class TestClaimGraph(unittest.TestCase):
    def test_graph_has_edges_for_multi_claim(self) -> None:
        profile = build_query_profile("Why did the deployment fail?")
        claims = decompose_query(profile)
        graph = build_claim_graph(claims, profile)
        self.assertTrue(len(graph.edges) > 0)

    def test_edges_have_relationship_type(self) -> None:
        profile = build_query_profile("Why did the deployment fail?")
        claims = decompose_query(profile)
        graph = build_claim_graph(claims, profile)
        for edge in graph.edges:
            self.assertIsInstance(edge.relationship, EdgeRelationship)

    def test_edges_have_rationale(self) -> None:
        profile = build_query_profile("Why did the deployment fail?")
        claims = decompose_query(profile)
        graph = build_claim_graph(claims, profile)
        for edge in graph.edges:
            self.assertTrue(len(edge.rationale) > 0)

    def test_comparison_graph_has_contradiction_edge(self) -> None:
        profile = build_query_profile("Compare Python versus Java for data pipelines")
        claims = decompose_query(profile)
        graph = build_claim_graph(claims, profile)
        relationships = [e.relationship for e in graph.edges]
        self.assertIn(EdgeRelationship.CONTRADICTION, relationships)

    def test_graph_covers_original_query(self) -> None:
        profile = build_query_profile("Why did the deployment fail?")
        claims = decompose_query(profile)
        graph = build_claim_graph(claims, profile)
        self.assertTrue(graph.covers_query(profile.raw_query))


class TestEvidenceLinkage(unittest.TestCase):
    def test_matching_evidence_updates_claim_status(self) -> None:
        profile = build_query_profile("Why did the deployment fail?")
        claims = decompose_query(profile)
        graph = build_claim_graph(claims, profile)
        # Evidence must share ≥2 words >4 chars with the claim template text.
        # Claim templates contain "described", "deployment", "event", "state", "consequences".
        bundle = _bundle(_item("The deployment state described a configuration error."))
        graph = link_evidence_to_claims(graph, bundle)
        supported = [c for c in graph.claims if c.status == ClaimStatus.SUPPORTED]
        self.assertTrue(len(supported) > 0)

    def test_unmatched_claims_are_unsupported(self) -> None:
        profile = build_query_profile("Why did the deployment fail?")
        claims = decompose_query(profile)
        graph = build_claim_graph(claims, profile)
        bundle = _bundle(_item("Docker is a containerisation tool."))
        graph = link_evidence_to_claims(graph, bundle)
        unsupported = [c for c in graph.claims if c.status == ClaimStatus.UNSUPPORTED]
        self.assertTrue(len(unsupported) > 0)

    def test_evidence_links_are_populated(self) -> None:
        profile = build_query_profile("Why did the deployment fail?")
        claims = decompose_query(profile)
        graph = build_claim_graph(claims, profile)
        bundle = _bundle(
            _item("The deployment state described a configuration error.", "doc-5"),
        )
        graph = link_evidence_to_claims(graph, bundle)
        all_links = [link for c in graph.claims for link in c.evidence_links]
        self.assertIn("doc-5", all_links)


class TestContextCompression(unittest.TestCase):
    def test_compressed_context_retains_source_links(self) -> None:
        profile = build_query_profile("Why did the deployment fail?")
        claims = decompose_query(profile)
        graph = build_claim_graph(claims, profile)
        bundle = _bundle(
            _item("The deployment state described a configuration error.", "src-1"),
            _item("Consequences of deployment described in the event report.", "src-2"),
        )
        graph = link_evidence_to_claims(graph, bundle)
        compressed = compress_context(bundle, graph)
        self.assertTrue(len(compressed.source_links) > 0)

    def test_compressed_context_has_summary(self) -> None:
        profile = build_query_profile("Why did the deployment fail?")
        claims = decompose_query(profile)
        graph = build_claim_graph(claims, profile)
        bundle = _bundle(_item("The deployment state described a timeout event."))
        graph = link_evidence_to_claims(graph, bundle)
        compressed = compress_context(bundle, graph)
        self.assertTrue(len(compressed.summary) > 0)

    def test_retained_item_count_matches_source_links(self) -> None:
        profile = build_query_profile("Why did the deployment fail?")
        claims = decompose_query(profile)
        graph = build_claim_graph(claims, profile)
        bundle = _bundle(
            _item("The deployment state described a timeout event.", "src-1"),
            _item("Deployment event consequences described in the post-mortem.", "src-2"),
        )
        graph = link_evidence_to_claims(graph, bundle)
        compressed = compress_context(bundle, graph)
        self.assertEqual(compressed.retained_item_count, len(compressed.source_links))


class TestRunComplexReasoning(unittest.TestCase):
    def test_simple_query_skips_complex_path(self) -> None:
        profile = build_query_profile("What is idempotency?")
        bundle = _bundle(_item("Idempotency means same result every time."))
        result = run_complex_reasoning(profile, bundle)
        self.assertFalse(result.used_complex_path)
        self.assertIsNone(result.graph)
        self.assertIsNone(result.compressed_context)

    def test_complex_query_uses_complex_path(self) -> None:
        profile = build_query_profile("Why did the deployment fail after the release?")
        bundle = _bundle(_item("The deployment failed due to a configuration error."))
        result = run_complex_reasoning(profile, bundle)
        self.assertTrue(result.used_complex_path)
        self.assertIsNotNone(result.graph)
        self.assertIsNotNone(result.compressed_context)

    def test_complex_path_result_has_claims(self) -> None:
        profile = build_query_profile("Why did the deployment fail after the release?")
        bundle = _bundle(_item("The deployment failed due to a configuration error."))
        result = run_complex_reasoning(profile, bundle)
        assert result.graph is not None
        self.assertTrue(len(result.graph.claims) > 0)

    def test_complex_path_compressed_context_has_source_links(self) -> None:
        profile = build_query_profile("Why did the deployment fail after the release?")
        bundle = _bundle(
            _item("The deployment state described a configuration error event.", "src-1"),
        )
        result = run_complex_reasoning(profile, bundle)
        assert result.compressed_context is not None
        self.assertTrue(len(result.compressed_context.source_links) > 0)


if __name__ == "__main__":
    unittest.main()
