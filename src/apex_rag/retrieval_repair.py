"""US-008 Retrieval Repair Loop — bounded iterative self-correction for poor retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.apex_rag.evidence_fusion import EvidenceBundle
from src.apex_rag.evidence_scoring import ScoredBundle, score_bundle
from src.apex_rag.expert_routing import ExpertType
from src.apex_rag.query_intelligence import QueryProfile


class FailureClass(str, Enum):
    NO_EVIDENCE = "no_evidence"
    LOW_RELEVANCE = "low_relevance"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    OUTDATED_EVIDENCE = "outdated_evidence"
    INCOMPLETE_COVERAGE = "incomplete_coverage"
    WRONG_EXPERT = "wrong_expert"


class RecoveryStrategy(str, Enum):
    EXPAND_QUERY = "expand_query"
    REWRITE_QUERY = "rewrite_query"
    ARBITRATE_CONFLICTS = "arbitrate_conflicts"
    FETCH_FRESH = "fetch_fresh"
    DECOMPOSE_CLAIMS = "decompose_claims"
    REROUTE_EXPERT = "reroute_expert"


_FAILURE_TO_RECOVERY: dict[FailureClass, RecoveryStrategy] = {
    FailureClass.NO_EVIDENCE: RecoveryStrategy.EXPAND_QUERY,
    FailureClass.LOW_RELEVANCE: RecoveryStrategy.REWRITE_QUERY,
    FailureClass.CONFLICTING_EVIDENCE: RecoveryStrategy.ARBITRATE_CONFLICTS,
    FailureClass.OUTDATED_EVIDENCE: RecoveryStrategy.FETCH_FRESH,
    FailureClass.INCOMPLETE_COVERAGE: RecoveryStrategy.DECOMPOSE_CLAIMS,
    FailureClass.WRONG_EXPERT: RecoveryStrategy.REROUTE_EXPERT,
}


@dataclass(frozen=True)
class RepairAttempt:
    iteration: int
    failure_class: FailureClass
    recovery_strategy: RecoveryStrategy
    confidence_before: float
    confidence_after: float
    outcome: str


@dataclass(frozen=True)
class RepairResult:
    succeeded: bool
    final_confidence: float
    iterations_used: int
    attempts: tuple[RepairAttempt, ...]
    final_bundle: EvidenceBundle
    failure_reason: str


# ---------------------------------------------------------------------------
# Failure detection
# ---------------------------------------------------------------------------

_CONFIDENCE_THRESHOLD_DEFAULT = 0.5
_LOW_RELEVANCE_THRESHOLD = 0.3
_MAX_CONFLICT_RATIO = 0.4
_STALENESS_CUTOFF = 0.25


def classify_failure(scored: ScoredBundle) -> FailureClass | None:
    """Return the primary failure class, or None when retrieval is acceptable."""
    scores = scored.scores

    if len(scored.bundle.items) == 0:
        return FailureClass.NO_EVIDENCE

    low_authority = scores.authority < _LOW_RELEVANCE_THRESHOLD
    low_completeness = scores.completeness < _LOW_RELEVANCE_THRESHOLD
    if low_authority and low_completeness:
        return FailureClass.LOW_RELEVANCE

    conflict_ratio = scored.bundle.conflict_count / max(len(scored.bundle.items), 1)
    if conflict_ratio > _MAX_CONFLICT_RATIO:
        return FailureClass.CONFLICTING_EVIDENCE

    if scores.freshness < _STALENESS_CUTOFF:
        return FailureClass.OUTDATED_EVIDENCE

    if scores.completeness < _LOW_RELEVANCE_THRESHOLD and len(scored.bundle.items) > 0:
        return FailureClass.INCOMPLETE_COVERAGE

    return None


# ---------------------------------------------------------------------------
# Recovery actions (deterministic, no side effects — simulate transformed query)
# ---------------------------------------------------------------------------


def _expand_query(profile: QueryProfile) -> str:
    """Broaden query scope by appending synonyms/context tokens."""
    return f"{profile.raw_query} overview introduction background"


def _rewrite_query(profile: QueryProfile) -> str:
    """Restate query with clearer phrasing and explicit scope markers."""
    return f"Explain in detail: {profile.raw_query}"


def _arbitrate_conflicts(bundle: EvidenceBundle) -> EvidenceBundle:
    """Keep only non-conflicting items."""
    from src.apex_rag.evidence_fusion import ConflictStatus

    kept = [item for item in bundle.items if item.conflict_status != ConflictStatus.CONFLICT]
    return EvidenceBundle(items=kept, query_id=bundle.query_id)


def _fetch_fresh_query(profile: QueryProfile) -> str:
    """Append a freshness marker to bias retrieval toward recent sources."""
    return f"{profile.raw_query} recent latest current"


def _decompose_for_coverage(profile: QueryProfile) -> list[str]:
    """Split the query into narrower sub-queries to improve coverage."""
    words = profile.raw_query.split()
    mid = max(1, len(words) // 2)
    part_a = " ".join(words[:mid])
    part_b = " ".join(words[mid:])
    return [part_a, part_b, profile.raw_query]


def _reroute_expert(profile: QueryProfile, current_expert: ExpertType) -> ExpertType:
    """Choose an alternative expert type."""
    fallback_order = [
        ExpertType.SEARCH,
        ExpertType.RESEARCH,
        ExpertType.ANALYTICS,
        ExpertType.POLICY,
        ExpertType.FRESHNESS,
        ExpertType.GRAPH,
    ]
    for expert in fallback_order:
        if expert != current_expert:
            return expert
    return ExpertType.SEARCH


# ---------------------------------------------------------------------------
# Repair loop
# ---------------------------------------------------------------------------

_MAX_ITERATIONS = 3


def run_repair_loop(
    profile: QueryProfile,
    bundle: EvidenceBundle,
    claim_ids: tuple[str, ...] = (),
    high_risk: bool = False,
    confidence_threshold: float = _CONFIDENCE_THRESHOLD_DEFAULT,
    max_iterations: int = _MAX_ITERATIONS,
    primary_expert: ExpertType = ExpertType.SEARCH,
) -> RepairResult:
    """Run the bounded repair loop until confidence threshold or max iterations is reached."""

    current_bundle = bundle
    current_expert = primary_expert
    attempts: list[RepairAttempt] = []

    for iteration in range(1, max_iterations + 1):
        scored = score_bundle(current_bundle, claim_ids, high_risk)

        if scored.sufficient:
            return RepairResult(
                succeeded=True,
                final_confidence=scored.scores.confidence,
                iterations_used=iteration - 1,
                attempts=tuple(attempts),
                final_bundle=current_bundle,
                failure_reason="",
            )

        failure = classify_failure(scored)
        if failure is None:
            return RepairResult(
                succeeded=True,
                final_confidence=scored.scores.confidence,
                iterations_used=iteration - 1,
                attempts=tuple(attempts),
                final_bundle=current_bundle,
                failure_reason="",
            )

        strategy = _FAILURE_TO_RECOVERY[failure]
        confidence_before = scored.scores.confidence

        if failure == FailureClass.CONFLICTING_EVIDENCE:
            current_bundle = _arbitrate_conflicts(current_bundle)
        elif failure == FailureClass.WRONG_EXPERT:
            current_expert = _reroute_expert(profile, current_expert)

        scored_after = score_bundle(current_bundle, claim_ids, high_risk)
        confidence_after = scored_after.scores.confidence

        outcome = "improved" if confidence_after > confidence_before else "unchanged"

        attempts.append(
            RepairAttempt(
                iteration=iteration,
                failure_class=failure,
                recovery_strategy=strategy,
                confidence_before=confidence_before,
                confidence_after=confidence_after,
                outcome=outcome,
            )
        )

        if confidence_after >= confidence_threshold:
            return RepairResult(
                succeeded=True,
                final_confidence=confidence_after,
                iterations_used=iteration,
                attempts=tuple(attempts),
                final_bundle=current_bundle,
                failure_reason="",
            )

    scored_final = score_bundle(current_bundle, claim_ids, high_risk)
    return RepairResult(
        succeeded=False,
        final_confidence=scored_final.scores.confidence,
        iterations_used=max_iterations,
        attempts=tuple(attempts),
        final_bundle=current_bundle,
        failure_reason=(
            f"Retrieval repair did not reach confidence threshold {confidence_threshold} "
            f"after {max_iterations} iterations."
        ),
    )
