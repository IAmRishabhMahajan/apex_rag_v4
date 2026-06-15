"""Real unit tests for US-006 Complex Query Reasoning Path implementation.

US-006 adds an optional reasoning path that kicks in for complex queries
(investigations, comparisons, multi-hop). It decomposes the query into
claims, builds a dependency/cause/contradiction graph, links evidence to
those claims, and compresses context down to a high-signal summary with
source citations. Simple queries bypass this path entirely.
"""

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
    """Build a minimal CitationMetadata stub for use in test evidence items."""
    return CitationMetadata(
        source_id=source_id,
        title="Doc",
        url="https://example.com",
        retrieval_expert="policy",
        retrieval_query="q",
    )


def _item(content: str, source_id: str = "src-1") -> EvidenceItem:
    """Build a single EvidenceItem with the given text and an optional source ID."""
    return EvidenceItem(content=content, citation=_citation(source_id), claim_ids=())


def _bundle(*items: EvidenceItem) -> EvidenceBundle:
    """Wrap one or more EvidenceItems into an EvidenceBundle with a fixed query ID."""
    return EvidenceBundle(items=list(items), query_id="q1")


class TestComplexityGate(unittest.TestCase):
    """Tests for is_complex() — the gate that decides whether a query needs the complex path.

    US-006 acceptance criterion: simple queries must bypass the complex path.
    The gate uses intent classification, keyword signals (why/root-cause/compare),
    and entity count heuristics from the QueryProfile produced by US-001.
    """

    def test_simple_fact_lookup_is_not_complex(self) -> None:
        """A plain factual lookup should not trigger the complex reasoning path."""
        profile = build_query_profile("What is idempotency?")
        self.assertFalse(is_complex(profile))

    def test_investigation_query_is_complex(self) -> None:
        """A 'why did X fail' investigation query should be marked complex."""
        profile = build_query_profile("Why did the deployment pipeline fail last night?")
        self.assertTrue(is_complex(profile))

    def test_comparison_query_is_complex(self) -> None:
        """A 'compare A versus B' query should be marked complex."""
        profile = build_query_profile("Compare Python versus Java for data pipelines")
        self.assertTrue(is_complex(profile))

    def test_analysis_query_is_complex(self) -> None:
        """An 'analyze X' query should be marked complex."""
        profile = build_query_profile("Analyze the performance breakdown of the API gateway")
        self.assertTrue(is_complex(profile))

    def test_keyword_why_triggers_complex(self) -> None:
        """The keyword 'why' alone should be sufficient to flag a query as complex."""
        profile = build_query_profile("Why does caching improve read latency?")
        self.assertTrue(is_complex(profile))

    def test_keyword_root_cause_triggers_complex(self) -> None:
        """The phrase 'root cause' should trigger the complex reasoning path."""
        profile = build_query_profile("Find the root cause of the memory leak")
        self.assertTrue(is_complex(profile))


class TestClaimDecomposition(unittest.TestCase):
    """Tests for decompose_query() — breaks a complex query into a fixed set of retrievable claims.

    US-006 task: decompose complex questions into claims. progress.md notes that
    decompose_query() uses 3-claim templates keyed on query intent (investigation,
    comparison, analysis). Each claim starts in PENDING status and contains
    context from the original question.
    """

    def test_investigation_produces_three_claims(self) -> None:
        """An investigation-intent query should decompose into exactly three claims."""
        profile = build_query_profile("Why did the deployment fail after the release?")
        claims = decompose_query(profile)
        self.assertEqual(len(claims), 3)

    def test_comparison_produces_three_claims(self) -> None:
        """A comparison-intent query should also decompose into exactly three claims."""
        profile = build_query_profile("Compare Python versus Java for data pipelines")
        claims = decompose_query(profile)
        self.assertEqual(len(claims), 3)

    def test_all_claims_start_pending(self) -> None:
        """Every freshly decomposed claim must start in PENDING status (no evidence yet)."""
        profile = build_query_profile("Why did the deployment fail?")
        claims = decompose_query(profile)
        for claim in claims:
            self.assertEqual(claim.status, ClaimStatus.PENDING)

    def test_claims_contain_original_query_text(self) -> None:
        """Claim text must include key terms from the original query for traceability."""
        profile = build_query_profile("Why did the deployment fail?")
        claims = decompose_query(profile)
        for claim in claims:
            self.assertIn("deployment", claim.text)

    def test_analysis_claims_cover_components(self) -> None:
        """Analysis-intent claims should mention 'components' to ensure broad coverage."""
        profile = build_query_profile("Analyze the performance breakdown of the system")
        claims = decompose_query(profile)
        combined = " ".join(c.text for c in claims).lower()
        self.assertIn("components", combined)


class TestClaimGraph(unittest.TestCase):
    """Tests for build_claim_graph() — links claims via typed edges (DEPENDENCY, CAUSE, CONTRADICTION).

    US-006 acceptance criterion: claim graph edges must include a relationship
    type and a rationale string. progress.md notes that comparison queries
    produce CONTRADICTION edges. The graph must also assert it covers the
    original query via covers_query().
    """

    def test_graph_has_edges_for_multi_claim(self) -> None:
        """A graph built from multiple claims should have at least one edge."""
        profile = build_query_profile("Why did the deployment fail?")
        claims = decompose_query(profile)
        graph = build_claim_graph(claims, profile)
        self.assertTrue(len(graph.edges) > 0)

    def test_edges_have_relationship_type(self) -> None:
        """Every graph edge must carry an EdgeRelationship enum value."""
        profile = build_query_profile("Why did the deployment fail?")
        claims = decompose_query(profile)
        graph = build_claim_graph(claims, profile)
        for edge in graph.edges:
            self.assertIsInstance(edge.relationship, EdgeRelationship)

    def test_edges_have_rationale(self) -> None:
        """Every graph edge must include a non-empty rationale string."""
        profile = build_query_profile("Why did the deployment fail?")
        claims = decompose_query(profile)
        graph = build_claim_graph(claims, profile)
        for edge in graph.edges:
            self.assertTrue(len(edge.rationale) > 0)

    def test_comparison_graph_has_contradiction_edge(self) -> None:
        """Comparing two technologies must produce at least one CONTRADICTION edge in the graph."""
        profile = build_query_profile("Compare Python versus Java for data pipelines")
        claims = decompose_query(profile)
        graph = build_claim_graph(claims, profile)
        relationships = [e.relationship for e in graph.edges]
        self.assertIn(EdgeRelationship.CONTRADICTION, relationships)

    def test_graph_covers_original_query(self) -> None:
        """The claim graph must confirm it covers the original user query."""
        profile = build_query_profile("Why did the deployment fail?")
        claims = decompose_query(profile)
        graph = build_claim_graph(claims, profile)
        self.assertTrue(graph.covers_query(profile.raw_query))


class TestEvidenceLinkage(unittest.TestCase):
    """Tests for link_evidence_to_claims() — attaches retrieved evidence to graph claims via word-overlap.

    progress.md: word-overlap matching requires ≥2 words >4 chars shared
    between the evidence text and a claim's template text. Matching evidence
    promotes a claim from PENDING to SUPPORTED; non-matching evidence leaves
    claims UNSUPPORTED. Source IDs from matched evidence are stored in
    claim.evidence_links for downstream citation tracing.
    """

    def test_matching_evidence_updates_claim_status(self) -> None:
        """Evidence that shares key terms with a claim should promote it to SUPPORTED."""
        profile = build_query_profile("Why did the deployment fail?")
        claims = decompose_query(profile)
        graph = build_claim_graph(claims, profile)
        # Evidence must share >=2 words >4 chars with the claim template text.
        # Claim templates contain "described", "deployment", "event", "state", "consequences".
        bundle = _bundle(_item("The deployment state described a configuration error."))
        graph = link_evidence_to_claims(graph, bundle)
        supported = [c for c in graph.claims if c.status == ClaimStatus.SUPPORTED]
        self.assertTrue(len(supported) > 0)

    def test_unmatched_claims_are_unsupported(self) -> None:
        """Evidence that shares no key terms with any claim should leave claims UNSUPPORTED."""
        profile = build_query_profile("Why did the deployment fail?")
        claims = decompose_query(profile)
        graph = build_claim_graph(claims, profile)
        bundle = _bundle(_item("Docker is a containerisation tool."))
        graph = link_evidence_to_claims(graph, bundle)
        unsupported = [c for c in graph.claims if c.status == ClaimStatus.UNSUPPORTED]
        self.assertTrue(len(unsupported) > 0)

    def test_evidence_links_are_populated(self) -> None:
        """Source IDs from matched evidence must appear in the claim's evidence_links list."""
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
    """Tests for compress_context() — distils evidence + graph into a prioritised summary with source links.

    US-006 task: compress large retrieved contexts into high-signal summaries
    tied to source evidence. compress_context() returns a CompressedContext
    with a summary string, a list of source_links (one per retained item),
    and a retained_item_count that must equal len(source_links).
    """

    def test_compressed_context_retains_source_links(self) -> None:
        """Compression must preserve at least one citation link from the input evidence."""
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
        """Compression must produce a non-empty summary string."""
        profile = build_query_profile("Why did the deployment fail?")
        claims = decompose_query(profile)
        graph = build_claim_graph(claims, profile)
        bundle = _bundle(_item("The deployment state described a timeout event."))
        graph = link_evidence_to_claims(graph, bundle)
        compressed = compress_context(bundle, graph)
        self.assertTrue(len(compressed.summary) > 0)

    def test_retained_item_count_matches_source_links(self) -> None:
        """retained_item_count must equal the number of source_links (one link per retained item)."""
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
    """Tests for run_complex_reasoning() — the top-level orchestrator for US-006.

    run_complex_reasoning() checks is_complex(), and if false returns a
    pass-through result with used_complex_path=False and no graph/compressed
    context. For complex queries it runs the full pipeline:
    decompose → build_claim_graph → link_evidence → compress_context,
    returning a result with used_complex_path=True and populated graph and
    compressed_context fields.
    """

    def test_simple_query_skips_complex_path(self) -> None:
        """A simple factual query must return used_complex_path=False with no graph or context."""
        profile = build_query_profile("What is idempotency?")
        bundle = _bundle(_item("Idempotency means same result every time."))
        result = run_complex_reasoning(profile, bundle)
        self.assertFalse(result.used_complex_path)
        self.assertIsNone(result.graph)
        self.assertIsNone(result.compressed_context)

    def test_complex_query_uses_complex_path(self) -> None:
        """A complex investigation query must return used_complex_path=True with graph and context."""
        profile = build_query_profile("Why did the deployment fail after the release?")
        bundle = _bundle(_item("The deployment failed due to a configuration error."))
        result = run_complex_reasoning(profile, bundle)
        self.assertTrue(result.used_complex_path)
        self.assertIsNotNone(result.graph)
        self.assertIsNotNone(result.compressed_context)

    def test_complex_path_result_has_claims(self) -> None:
        """The claim graph produced by the complex path must contain at least one claim."""
        profile = build_query_profile("Why did the deployment fail after the release?")
        bundle = _bundle(_item("The deployment failed due to a configuration error."))
        result = run_complex_reasoning(profile, bundle)
        assert result.graph is not None
        self.assertTrue(len(result.graph.claims) > 0)

    def test_complex_path_compressed_context_has_source_links(self) -> None:
        """The compressed context from the complex path must carry at least one source link."""
        profile = build_query_profile("Why did the deployment fail after the release?")
        bundle = _bundle(
            _item("The deployment state described a configuration error event.", "src-1"),
        )
        result = run_complex_reasoning(profile, bundle)
        assert result.compressed_context is not None
        self.assertTrue(len(result.compressed_context.source_links) > 0)


if __name__ == "__main__":
    unittest.main()
