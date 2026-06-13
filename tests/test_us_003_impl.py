"""Real unit tests for US-003 Expert Retrieval Routing implementation."""

from __future__ import annotations

import unittest

from src.apex_rag.expert_routing import (
    ExpertType,
    ExpertUnavailableError,
    route_to_experts,
)
from src.apex_rag.query_intelligence import build_query_profile
from src.apex_rag.retrieval_planning import plan_retrieval


def _route(query: str, available: frozenset[ExpertType] | None = None):  # type: ignore[no-untyped-def]
    profile = build_query_profile(query)
    plan = plan_retrieval(profile)
    if available is None:
        return route_to_experts(plan)
    return route_to_experts(plan, available)


class TestPolicyRouting(unittest.TestCase):
    def test_simple_query_routes_to_policy(self) -> None:
        result = _route("What is idempotency?")
        self.assertEqual(result.primary_expert, ExpertType.POLICY)

    def test_policy_selection_has_reason(self) -> None:
        result = _route("What is idempotency?")
        self.assertTrue(len(result.selections[0].reason) > 0)

    def test_policy_confidence_above_threshold(self) -> None:
        result = _route("What is idempotency?")
        self.assertGreater(result.selections[0].confidence, 0.5)


class TestResearchRouting(unittest.TestCase):
    def test_investigation_routes_to_research(self) -> None:
        result = _route("Why did the deployment fail after the last release?")
        experts = {s.expert for s in result.selections}
        self.assertIn(ExpertType.RESEARCH, experts)

    def test_analysis_routes_to_research(self) -> None:
        result = _route("Analyze how Python, Docker, and Kubernetes interact")
        experts = {s.expert for s in result.selections}
        self.assertIn(ExpertType.RESEARCH, experts)


class TestAnalyticsRouting(unittest.TestCase):
    def test_metrics_query_routes_to_analytics(self) -> None:
        result = _route("What are the key metrics on our dashboard?")
        experts = {s.expert for s in result.selections}
        self.assertIn(ExpertType.ANALYTICS, experts)


class TestGraphRouting(unittest.TestCase):
    def test_relationship_query_routes_to_graph(self) -> None:
        result = _route("What is the dependency relationship between services?")
        experts = {s.expert for s in result.selections}
        self.assertIn(ExpertType.GRAPH, experts)


class TestFreshnessRouting(unittest.TestCase):
    def test_latest_routes_to_freshness(self) -> None:
        result = _route("What is the latest status of the AWS outage?")
        experts = {s.expert for s in result.selections}
        self.assertIn(ExpertType.FRESHNESS, experts)


class TestMultiExpertRouting(unittest.TestCase):
    def test_complex_query_is_multi_expert(self) -> None:
        result = _route("Analyze the latest metrics and dependency graph for our network")
        self.assertTrue(result.is_multi_expert)

    def test_all_selections_have_reasons(self) -> None:
        result = _route("Analyze the latest metrics and dependency graph for our network")
        for sel in result.selections:
            self.assertTrue(len(sel.reason) > 0)

    def test_no_duplicate_experts(self) -> None:
        result = _route("Analyze the latest metrics and dependency graph for our network")
        expert_list = [s.expert for s in result.selections]
        self.assertEqual(len(expert_list), len(set(expert_list)))


class TestUnavailableExpert(unittest.TestCase):
    def test_unavailable_with_fallback_is_degraded(self) -> None:
        no_freshness = frozenset(ExpertType) - {ExpertType.FRESHNESS}
        result = _route("What is the current live status?", available=no_freshness)
        self.assertTrue(result.is_degraded)
        self.assertIn(ExpertType.FRESHNESS, result.unavailable_experts)

    def test_fallback_expert_is_included(self) -> None:
        no_freshness = frozenset(ExpertType) - {ExpertType.FRESHNESS}
        result = _route("What is the current live status?", available=no_freshness)
        experts = {s.expert for s in result.selections}
        self.assertIn(ExpertType.SEARCH, experts)

    def test_unavailable_no_fallback_raises(self) -> None:
        no_policy = frozenset(ExpertType) - {ExpertType.POLICY, ExpertType.RESEARCH}
        with self.assertRaises(ExpertUnavailableError) as ctx:
            _route("What is idempotency?", available=no_policy)
        self.assertIsInstance(ctx.exception, ExpertUnavailableError)

    def test_error_message_is_actionable(self) -> None:
        no_policy = frozenset(ExpertType) - {ExpertType.POLICY, ExpertType.RESEARCH}
        try:
            _route("What is idempotency?", available=no_policy)
        except ExpertUnavailableError as exc:
            self.assertIn("unavailable", str(exc).lower())


if __name__ == "__main__":
    unittest.main()
