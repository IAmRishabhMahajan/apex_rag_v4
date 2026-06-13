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
    return plan_retrieval(build_query_profile(query))


class TestStandardPlan(unittest.TestCase):
    def test_simple_lookup_uses_standard(self) -> None:
        plan = _plan("What is Kubernetes?")
        self.assertIn(RetrievalStrategy.STANDARD, plan.strategies)

    def test_standard_plan_includes_document_evidence(self) -> None:
        plan = _plan("What is Kubernetes?")
        self.assertIn(EvidenceType.DOCUMENT, plan.required_evidence_types)

    def test_plan_has_planning_reason(self) -> None:
        plan = _plan("What is the capital of France?")
        self.assertTrue(len(plan.planning_reasons) > 0)

    def test_fallback_when_no_specialised_plan(self) -> None:
        plan = _plan("Define idempotency")
        self.assertIn(RetrievalStrategy.STANDARD, plan.strategies)


class TestMultiHopPlan(unittest.TestCase):
    def test_investigation_produces_multi_hop(self) -> None:
        plan = _plan("Why did the deployment pipeline fail after the recent release?")
        self.assertIn(RetrievalStrategy.MULTI_HOP, plan.strategies)

    def test_analysis_produces_multi_hop(self) -> None:
        plan = _plan("Analyze how Python, Docker, and Kubernetes interact in CI/CD")
        self.assertIn(RetrievalStrategy.MULTI_HOP, plan.strategies)

    def test_multi_hop_includes_research_expert(self) -> None:
        plan = _plan("Why did the deployment pipeline fail after the recent release?")
        self.assertIn("research", plan.selected_experts)


class TestFreshnessPlan(unittest.TestCase):
    def test_latest_keyword_triggers_freshness(self) -> None:
        plan = _plan("What is the latest version of Python?")
        self.assertIn(RetrievalStrategy.FRESHNESS, plan.strategies)
        self.assertTrue(plan.freshness_required)

    def test_freshness_includes_real_time_evidence(self) -> None:
        plan = _plan("What is the current status of the AWS outage?")
        self.assertIn(EvidenceType.REAL_TIME, plan.required_evidence_types)

    def test_forecast_triggers_freshness(self) -> None:
        plan = _plan("What is the forecast for AI spending in 2026?")
        self.assertTrue(plan.freshness_required)


class TestAnalyticsPlan(unittest.TestCase):
    def test_metrics_query_uses_analytics(self) -> None:
        plan = _plan("What are the key metrics for our dashboard this quarter?")
        self.assertIn(RetrievalStrategy.ANALYTICS, plan.strategies)

    def test_analytics_includes_structured_evidence(self) -> None:
        plan = _plan("Show aggregate count of incidents by region")
        self.assertIn(EvidenceType.STRUCTURED, plan.required_evidence_types)


class TestGraphPlan(unittest.TestCase):
    def test_relationship_query_uses_graph(self) -> None:
        plan = _plan("What is the dependency relationship between these services?")
        self.assertIn(RetrievalStrategy.GRAPH, plan.strategies)

    def test_graph_includes_graph_node_evidence(self) -> None:
        plan = _plan("Show the network of connected components")
        self.assertIn(EvidenceType.GRAPH_NODE, plan.required_evidence_types)


class TestMultiStrategyPlan(unittest.TestCase):
    def test_complex_query_is_multi_strategy(self) -> None:
        plan = _plan("Analyze the latest metrics and dependency graph for our network")
        self.assertTrue(plan.is_multi_strategy)

    def test_coverage_is_broad_for_multi_strategy(self) -> None:
        plan = _plan("Analyze the latest metrics and dependency graph for our network")
        self.assertEqual(plan.expected_coverage, "broad")

    def test_coverage_is_standard_for_simple(self) -> None:
        plan = _plan("What is idempotency?")
        self.assertEqual(plan.expected_coverage, "standard")


if __name__ == "__main__":
    unittest.main()
