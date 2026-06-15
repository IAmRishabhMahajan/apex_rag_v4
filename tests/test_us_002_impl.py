"""Real unit tests for US-002 Adaptive Retrieval Planning implementation."""

from __future__ import annotations

import unittest

from src.apex_rag.query_intelligence import build_query_profile
from src.apex_rag.retrieval_planning import (
    EvidenceType,
    RetrievalStrategy,
    plan_retrieval,
)


def _plan(query: str):  # type: ignore[no-untyped-def]
    """Build a QueryProfile and derive a RetrievalPlan in one call."""
    return plan_retrieval(build_query_profile(query))


class TestStandardPlan(unittest.TestCase):
    """Tests for plan_retrieval() standard/fallback path.

    US-002 requires that queries without specialist signals fall back to
    STANDARD strategy with DOCUMENT evidence and a policy expert.
    """

    def test_simple_lookup_uses_standard(self) -> None:
        """A plain fact-lookup query should produce a STANDARD strategy."""
        plan = _plan("What is Kubernetes?")
        self.assertIn(RetrievalStrategy.STANDARD, plan.strategies)

    def test_standard_plan_includes_document_evidence(self) -> None:
        """STANDARD strategy must include DOCUMENT as a required evidence type."""
        plan = _plan("What is Kubernetes?")
        self.assertIn(EvidenceType.DOCUMENT, plan.required_evidence_types)

    def test_plan_has_planning_reason(self) -> None:
        """Every plan must include at least one human-readable planning reason."""
        plan = _plan("What is the capital of France?")
        self.assertTrue(len(plan.planning_reasons) > 0)

    def test_fallback_when_no_specialised_plan(self) -> None:
        """A query with no specialist signals should still get STANDARD strategy."""
        plan = _plan("Define idempotency")
        self.assertIn(RetrievalStrategy.STANDARD, plan.strategies)


class TestMultiHopPlan(unittest.TestCase):
    """Tests for plan_retrieval() multi-hop path triggered by investigation/analysis intent.

    US-002 requires MULTI_HOP for investigation queries and for queries with
    three or more non-date entities.
    """

    def test_investigation_produces_multi_hop(self) -> None:
        """An investigation query should include MULTI_HOP in the strategies."""
        plan = _plan("Why did the deployment pipeline fail after the recent release?")
        self.assertIn(RetrievalStrategy.MULTI_HOP, plan.strategies)

    def test_analysis_produces_multi_hop(self) -> None:
        """An analysis query with multiple entities should include MULTI_HOP."""
        plan = _plan("Analyze how Python, Docker, and Kubernetes interact in CI/CD")
        self.assertIn(RetrievalStrategy.MULTI_HOP, plan.strategies)

    def test_multi_hop_includes_research_expert(self) -> None:
        """MULTI_HOP strategy must select the 'research' expert."""
        plan = _plan("Why did the deployment pipeline fail after the recent release?")
        self.assertIn("research", plan.selected_experts)


class TestFreshnessPlan(unittest.TestCase):
    """Tests for plan_retrieval() freshness path triggered by recency keywords or FORECASTING intent.

    US-002 requires FRESHNESS strategy when the query asks for current or live data.
    """

    def test_latest_keyword_triggers_freshness(self) -> None:
        """The keyword 'latest' should trigger FRESHNESS strategy and set freshness_required."""
        plan = _plan("What is the latest version of Python?")
        self.assertIn(RetrievalStrategy.FRESHNESS, plan.strategies)
        self.assertTrue(plan.freshness_required)

    def test_freshness_includes_real_time_evidence(self) -> None:
        """FRESHNESS strategy must include REAL_TIME as a required evidence type."""
        plan = _plan("What is the current status of the AWS outage?")
        self.assertIn(EvidenceType.REAL_TIME, plan.required_evidence_types)

    def test_forecast_triggers_freshness(self) -> None:
        """A FORECASTING-intent query should set freshness_required to True."""
        plan = _plan("What is the forecast for AI spending in 2026?")
        self.assertTrue(plan.freshness_required)


class TestAnalyticsPlan(unittest.TestCase):
    """Tests for plan_retrieval() analytics path triggered by metric/aggregate keywords.

    US-002 requires ANALYTICS strategy for queries referencing dashboards or KPIs.
    """

    def test_metrics_query_uses_analytics(self) -> None:
        """A query containing 'metrics' should include ANALYTICS strategy."""
        plan = _plan("What are the key metrics for our dashboard this quarter?")
        self.assertIn(RetrievalStrategy.ANALYTICS, plan.strategies)

    def test_analytics_includes_structured_evidence(self) -> None:
        """ANALYTICS strategy must include STRUCTURED as a required evidence type."""
        plan = _plan("Show aggregate count of incidents by region")
        self.assertIn(EvidenceType.STRUCTURED, plan.required_evidence_types)


class TestGraphPlan(unittest.TestCase):
    """Tests for plan_retrieval() graph path triggered by relationship/dependency keywords.

    US-002 requires GRAPH strategy for queries asking about entity relationships.
    """

    def test_relationship_query_uses_graph(self) -> None:
        """A query mentioning 'dependency relationship' should include GRAPH strategy."""
        plan = _plan("What is the dependency relationship between these services?")
        self.assertIn(RetrievalStrategy.GRAPH, plan.strategies)

    def test_graph_includes_graph_node_evidence(self) -> None:
        """GRAPH strategy must include GRAPH_NODE as a required evidence type."""
        plan = _plan("Show the network of connected components")
        self.assertIn(EvidenceType.GRAPH_NODE, plan.required_evidence_types)


class TestMultiStrategyPlan(unittest.TestCase):
    """Tests for plan_retrieval() when multiple specialist signals appear in one query.

    US-002 requires is_multi_strategy=True and expected_coverage='broad' for such queries.
    """

    def test_complex_query_is_multi_strategy(self) -> None:
        """A query with both metrics and graph signals should produce multiple strategies."""
        plan = _plan("Analyze the latest metrics and dependency graph for our network")
        self.assertTrue(plan.is_multi_strategy)

    def test_coverage_is_broad_for_multi_strategy(self) -> None:
        """A multi-strategy plan should report expected_coverage as 'broad'."""
        plan = _plan("Analyze the latest metrics and dependency graph for our network")
        self.assertEqual(plan.expected_coverage, "broad")

    def test_coverage_is_standard_for_simple(self) -> None:
        """A single-strategy fact-lookup plan should report expected_coverage as 'standard'."""
        plan = _plan("What is idempotency?")
        self.assertEqual(plan.expected_coverage, "standard")


if __name__ == "__main__":
    unittest.main()
