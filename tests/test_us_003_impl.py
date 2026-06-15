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
    """Build a plan from the query and route it to experts, with optional availability mask."""
    profile = build_query_profile(query)
    plan = plan_retrieval(profile)
    if available is None:
        return route_to_experts(plan)
    return route_to_experts(plan, available)


class TestPolicyRouting(unittest.TestCase):
    """Tests for route_to_experts() default/policy path.

    US-003 requires that simple queries route to the POLICY expert as primary,
    with a confidence above 0.5 and a non-empty reason string.
    """

    def test_simple_query_routes_to_policy(self) -> None:
        """A fact-lookup query should select POLICY as the primary expert."""
        result = _route("What is idempotency?")
        self.assertEqual(result.primary_expert, ExpertType.POLICY)

    def test_policy_selection_has_reason(self) -> None:
        """The POLICY selection must include a non-empty reason string."""
        result = _route("What is idempotency?")
        self.assertTrue(len(result.selections[0].reason) > 0)

    def test_policy_confidence_above_threshold(self) -> None:
        """The POLICY expert's confidence must exceed 0.5."""
        result = _route("What is idempotency?")
        self.assertGreater(result.selections[0].confidence, 0.5)


class TestResearchRouting(unittest.TestCase):
    """Tests for route_to_experts() research path triggered by investigation/analysis intent.

    US-003 requires RESEARCH expert for multi-hop queries.
    """

    def test_investigation_routes_to_research(self) -> None:
        """An investigation query should include the RESEARCH expert in selections."""
        result = _route("Why did the deployment fail after the last release?")
        experts = {s.expert for s in result.selections}
        self.assertIn(ExpertType.RESEARCH, experts)

    def test_analysis_routes_to_research(self) -> None:
        """An analysis query with multiple entities should route to RESEARCH."""
        result = _route("Analyze how Python, Docker, and Kubernetes interact")
        experts = {s.expert for s in result.selections}
        self.assertIn(ExpertType.RESEARCH, experts)


class TestAnalyticsRouting(unittest.TestCase):
    """Tests for route_to_experts() analytics path triggered by metric keywords.

    US-003 requires ANALYTICS expert for structured-data queries.
    """

    def test_metrics_query_routes_to_analytics(self) -> None:
        """A query referencing metrics should include the ANALYTICS expert."""
        result = _route("What are the key metrics on our dashboard?")
        experts = {s.expert for s in result.selections}
        self.assertIn(ExpertType.ANALYTICS, experts)


class TestGraphRouting(unittest.TestCase):
    """Tests for route_to_experts() graph path triggered by relationship keywords.

    US-003 requires GRAPH expert for dependency/relationship queries.
    """

    def test_relationship_query_routes_to_graph(self) -> None:
        """A query about service dependencies should include the GRAPH expert."""
        result = _route("What is the dependency relationship between services?")
        experts = {s.expert for s in result.selections}
        self.assertIn(ExpertType.GRAPH, experts)


class TestFreshnessRouting(unittest.TestCase):
    """Tests for route_to_experts() freshness path triggered by recency keywords.

    US-003 requires FRESHNESS expert for real-time or live-data queries.
    """

    def test_latest_routes_to_freshness(self) -> None:
        """A query asking for the 'latest status' should include the FRESHNESS expert."""
        result = _route("What is the latest status of the AWS outage?")
        experts = {s.expert for s in result.selections}
        self.assertIn(ExpertType.FRESHNESS, experts)


class TestMultiExpertRouting(unittest.TestCase):
    """Tests for route_to_experts() when multiple experts are needed.

    US-003 requires is_multi_expert=True when more than one expert is selected,
    no duplicate experts in the selection list, and all selections have reasons.
    """

    def test_complex_query_is_multi_expert(self) -> None:
        """A query with mixed signals should produce more than one expert selection."""
        result = _route("Analyze the latest metrics and dependency graph for our network")
        self.assertTrue(result.is_multi_expert)

    def test_all_selections_have_reasons(self) -> None:
        """Every expert selection must carry a non-empty reason string."""
        result = _route("Analyze the latest metrics and dependency graph for our network")
        for sel in result.selections:
            self.assertTrue(len(sel.reason) > 0)

    def test_no_duplicate_experts(self) -> None:
        """No expert should appear more than once in the selection list."""
        result = _route("Analyze the latest metrics and dependency graph for our network")
        expert_list = [s.expert for s in result.selections]
        self.assertEqual(len(expert_list), len(set(expert_list)))


class TestUnavailableExpert(unittest.TestCase):
    """Tests for route_to_experts() fallback and error behaviour when experts are unavailable.

    US-003 requires degraded routing via fallback chains and ExpertUnavailableError
    when no fallback exists.
    """

    def test_unavailable_with_fallback_is_degraded(self) -> None:
        """Removing FRESHNESS from available experts should set is_degraded=True."""
        no_freshness = frozenset(ExpertType) - {ExpertType.FRESHNESS}
        result = _route("What is the current live status?", available=no_freshness)
        self.assertTrue(result.is_degraded)
        self.assertIn(ExpertType.FRESHNESS, result.unavailable_experts)

    def test_fallback_expert_is_included(self) -> None:
        """When FRESHNESS is unavailable, its fallback SEARCH should be selected instead."""
        no_freshness = frozenset(ExpertType) - {ExpertType.FRESHNESS}
        result = _route("What is the current live status?", available=no_freshness)
        experts = {s.expert for s in result.selections}
        self.assertIn(ExpertType.SEARCH, experts)

    def test_unavailable_no_fallback_raises(self) -> None:
        """Removing POLICY and RESEARCH should raise ExpertUnavailableError for a fact query."""
        no_policy = frozenset(ExpertType) - {ExpertType.POLICY, ExpertType.RESEARCH}
        with self.assertRaises(ExpertUnavailableError) as ctx:
            _route("What is idempotency?", available=no_policy)
        self.assertIsInstance(ctx.exception, ExpertUnavailableError)

    def test_error_message_is_actionable(self) -> None:
        """ExpertUnavailableError message should mention 'unavailable' so it is actionable."""
        no_policy = frozenset(ExpertType) - {ExpertType.POLICY, ExpertType.RESEARCH}
        try:
            _route("What is idempotency?", available=no_policy)
        except ExpertUnavailableError as exc:
            self.assertIn("unavailable", str(exc).lower())


if __name__ == "__main__":
    unittest.main()
