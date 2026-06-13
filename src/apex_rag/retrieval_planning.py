"""US-002 Adaptive Retrieval Planning — maps a QueryProfile to a RetrievalPlan."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.apex_rag.query_intelligence import (
    ConstraintType,
    EntityType,
    Intent,
    QueryProfile,
)


class RetrievalStrategy(str, Enum):
    STANDARD = "standard"
    MULTI_HOP = "multi_hop"
    GRAPH = "graph"
    ANALYTICS = "analytics"
    FRESHNESS = "freshness"
    WEB = "web"


class EvidenceType(str, Enum):
    DOCUMENT = "document"
    STRUCTURED = "structured"
    GRAPH_NODE = "graph_node"
    REAL_TIME = "real_time"
    WEB_PAGE = "web_page"


@dataclass(frozen=True)
class RetrievalPlan:
    strategies: tuple[RetrievalStrategy, ...]
    selected_experts: tuple[str, ...]
    required_evidence_types: tuple[EvidenceType, ...]
    freshness_required: bool
    expected_coverage: str
    planning_reasons: tuple[str, ...]

    @property
    def is_multi_strategy(self) -> bool:
        return len(self.strategies) > 1


# ---------------------------------------------------------------------------
# Planner rules
# ---------------------------------------------------------------------------

_FRESHNESS_INTENTS = frozenset({Intent.FORECASTING})
_INVESTIGATION_INTENTS = frozenset({Intent.INVESTIGATION, Intent.ANALYSIS})
_COMPARISON_INTENTS = frozenset({Intent.COMPARISON})

_ANALYTICS_KEYWORDS = frozenset(
    {"metric", "metrics", "dashboard", "kpi", "count", "aggregate", "sum", "average", "ratio"}
)
_GRAPH_KEYWORDS = frozenset(
    {"relationship", "connected", "linked", "dependency", "network", "graph", "path between"}
)
_FRESHNESS_KEYWORDS = frozenset(
    {"latest", "recent", "current", "today", "now", "live", "real-time", "breaking"}
)


def _query_contains(query: str, keywords: frozenset[str]) -> bool:
    lower = query.lower()
    return any(kw in lower for kw in keywords)


def _needs_freshness(profile: QueryProfile) -> bool:
    if profile.intent in _FRESHNESS_INTENTS:
        return True
    if _query_contains(profile.raw_query, _FRESHNESS_KEYWORDS):
        return True
    has_time_constraint = any(
        c.constraint_type == ConstraintType.TIME_RANGE for c in profile.constraints
    )
    return has_time_constraint and _query_contains(profile.raw_query, _FRESHNESS_KEYWORDS)


def _needs_analytics(profile: QueryProfile) -> bool:
    return _query_contains(profile.raw_query, _ANALYTICS_KEYWORDS)


def _needs_graph(profile: QueryProfile) -> bool:
    return _query_contains(profile.raw_query, _GRAPH_KEYWORDS)


def _needs_multi_hop(profile: QueryProfile) -> bool:
    if profile.intent in _INVESTIGATION_INTENTS:
        return True
    entity_count = sum(1 for e in profile.entities if e.entity_type not in (EntityType.DATE,))
    return entity_count >= 3


def plan_retrieval(profile: QueryProfile) -> RetrievalPlan:
    """Derive a RetrievalPlan from a validated QueryProfile."""

    strategies: list[RetrievalStrategy] = []
    experts: list[str] = []
    evidence_types: list[EvidenceType] = []
    reasons: list[str] = []
    freshness_required = False

    if _needs_analytics(profile):
        strategies.append(RetrievalStrategy.ANALYTICS)
        experts.append("analytics")
        evidence_types.append(EvidenceType.STRUCTURED)
        reasons.append("Query references metrics or aggregated data.")

    if _needs_graph(profile):
        strategies.append(RetrievalStrategy.GRAPH)
        experts.append("graph")
        evidence_types.append(EvidenceType.GRAPH_NODE)
        reasons.append("Query asks about relationships or connected entities.")

    if _needs_freshness(profile):
        freshness_required = True
        strategies.append(RetrievalStrategy.FRESHNESS)
        experts.append("freshness")
        evidence_types.append(EvidenceType.REAL_TIME)
        reasons.append("Query requires up-to-date or time-sensitive information.")

    if _needs_multi_hop(profile):
        strategies.append(RetrievalStrategy.MULTI_HOP)
        experts.append("research")
        evidence_types.append(EvidenceType.DOCUMENT)
        reasons.append("Query involves investigation or multiple related entities.")

    # Fallback: always include standard document retrieval
    if not strategies or RetrievalStrategy.STANDARD not in strategies:
        if not evidence_types or EvidenceType.DOCUMENT not in evidence_types:
            evidence_types.append(EvidenceType.DOCUMENT)
        if not strategies:
            strategies.append(RetrievalStrategy.STANDARD)
            experts.append("policy")
            reasons.append("No specialised retrieval required; using standard document search.")

    coverage = (
        "broad"
        if len(strategies) > 1
        else ("deep" if RetrievalStrategy.MULTI_HOP in strategies else "standard")
    )

    return RetrievalPlan(
        strategies=tuple(dict.fromkeys(strategies)),
        selected_experts=tuple(dict.fromkeys(experts)),
        required_evidence_types=tuple(dict.fromkeys(evidence_types)),
        freshness_required=freshness_required,
        expected_coverage=coverage,
        planning_reasons=tuple(reasons),
    )
