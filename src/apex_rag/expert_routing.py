"""US-003 Expert Retrieval Routing — maps a RetrievalPlan to one or more experts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.apex_rag.retrieval_planning import RetrievalPlan, RetrievalStrategy


class ExpertType(str, Enum):
    POLICY = "policy"
    RESEARCH = "research"
    ANALYTICS = "analytics"
    GRAPH = "graph"
    FRESHNESS = "freshness"
    SEARCH = "search"


@dataclass(frozen=True)
class ExpertSelection:
    expert: ExpertType
    confidence: float
    reason: str


@dataclass(frozen=True)
class RoutingResult:
    selections: tuple[ExpertSelection, ...]
    is_degraded: bool
    unavailable_experts: tuple[ExpertType, ...]

    @property
    def primary_expert(self) -> ExpertType:
        return self.selections[0].expert

    @property
    def is_multi_expert(self) -> bool:
        return len(self.selections) > 1


class ExpertUnavailableError(Exception):
    """Raised when a required expert cannot be satisfied and no fallback exists."""

    def __init__(self, expert: ExpertType) -> None:
        super().__init__(f"Expert '{expert.value}' is unavailable and no fallback is configured.")
        self.expert = expert


# ---------------------------------------------------------------------------
# Expert registry — in production these would be live clients/adapters
# ---------------------------------------------------------------------------

_DEFAULT_AVAILABLE: frozenset[ExpertType] = frozenset(ExpertType)

_STRATEGY_TO_EXPERT: dict[RetrievalStrategy, ExpertType] = {
    RetrievalStrategy.STANDARD: ExpertType.POLICY,
    RetrievalStrategy.MULTI_HOP: ExpertType.RESEARCH,
    RetrievalStrategy.GRAPH: ExpertType.GRAPH,
    RetrievalStrategy.ANALYTICS: ExpertType.ANALYTICS,
    RetrievalStrategy.FRESHNESS: ExpertType.FRESHNESS,
    RetrievalStrategy.WEB: ExpertType.SEARCH,
}

_EXPERT_REASONS: dict[ExpertType, str] = {
    ExpertType.POLICY: "Policy and document retrieval for structured knowledge bases.",
    ExpertType.RESEARCH: "Multi-hop retrieval across academic or technical documents.",
    ExpertType.ANALYTICS: "Structured data retrieval for metrics and dashboards.",
    ExpertType.GRAPH: "Graph traversal for relationship and dependency queries.",
    ExpertType.FRESHNESS: "Real-time retrieval for recent events and live data.",
    ExpertType.SEARCH: "Keyword and exact-match retrieval for IDs or legal phrases.",
}

_EXPERT_CONFIDENCE: dict[ExpertType, float] = {
    ExpertType.POLICY: 0.9,
    ExpertType.RESEARCH: 0.85,
    ExpertType.ANALYTICS: 0.9,
    ExpertType.GRAPH: 0.85,
    ExpertType.FRESHNESS: 0.8,
    ExpertType.SEARCH: 0.95,
}

# Fallback chain: if primary unavailable, try these in order
_FALLBACK: dict[ExpertType, ExpertType] = {
    ExpertType.FRESHNESS: ExpertType.SEARCH,
    ExpertType.GRAPH: ExpertType.RESEARCH,
    ExpertType.ANALYTICS: ExpertType.SEARCH,
}


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------


def route_to_experts(
    plan: RetrievalPlan,
    available: frozenset[ExpertType] = _DEFAULT_AVAILABLE,
) -> RoutingResult:
    """Select experts for a RetrievalPlan, respecting availability."""

    selections: list[ExpertSelection] = []
    unavailable: list[ExpertType] = []
    is_degraded = False

    for strategy in plan.strategies:
        desired = _STRATEGY_TO_EXPERT.get(strategy)
        if desired is None:
            continue

        if desired in available:
            selections.append(
                ExpertSelection(
                    expert=desired,
                    confidence=_EXPERT_CONFIDENCE[desired],
                    reason=_EXPERT_REASONS[desired],
                )
            )
        else:
            unavailable.append(desired)
            fallback = _FALLBACK.get(desired)
            if fallback and fallback in available:
                is_degraded = True
                selections.append(
                    ExpertSelection(
                        expert=fallback,
                        confidence=_EXPERT_CONFIDENCE[fallback] * 0.7,
                        reason=(
                            f"Fallback for unavailable {desired.value}: {_EXPERT_REASONS[fallback]}"
                        ),
                    )
                )
            else:
                raise ExpertUnavailableError(desired)

    # Deduplicate by expert, keeping highest-confidence entry
    seen: dict[ExpertType, ExpertSelection] = {}
    for sel in selections:
        if sel.expert not in seen or sel.confidence > seen[sel.expert].confidence:
            seen[sel.expert] = sel

    ordered = list(seen.values())

    if not ordered:
        # Final safety net — always return at least the policy expert
        ordered = [
            ExpertSelection(
                expert=ExpertType.POLICY,
                confidence=_EXPERT_CONFIDENCE[ExpertType.POLICY],
                reason=_EXPERT_REASONS[ExpertType.POLICY],
            )
        ]

    return RoutingResult(
        selections=tuple(ordered),
        is_degraded=is_degraded,
        unavailable_experts=tuple(unavailable),
    )
