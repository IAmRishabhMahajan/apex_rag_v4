"""US-007 Evidence Scoring and Gap Detection — score quality and detect coverage gaps."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.apex_rag.evidence_fusion import ConflictStatus, EvidenceBundle


class GapType(str, Enum):
    """Categories of evidence quality gap that trigger repair or escalation."""

    MISSING_EVIDENCE = "missing_evidence"
    LOW_RELEVANCE = "low_relevance"
    STALE_EVIDENCE = "stale_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    INCOMPLETE_COVERAGE = "incomplete_coverage"


@dataclass(frozen=True)
class EvidenceScores:
    """Five quality dimensions for an EvidenceBundle, each in [0.0, 1.0]."""

    authority: float
    freshness: float
    agreement: float
    completeness: float
    confidence: float
    reason: str

    def __post_init__(self) -> None:
        """Validate that all score dimensions are within the [0.0, 1.0] range."""
        for name, val in (
            ("authority", self.authority),
            ("freshness", self.freshness),
            ("agreement", self.agreement),
            ("completeness", self.completeness),
            ("confidence", self.confidence),
        ):
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"Score '{name}' must be in [0.0, 1.0], got {val}")


@dataclass(frozen=True)
class GapReport:
    """Describes a single evidence quality gap and how to address it."""

    gap_type: GapType
    description: str
    affected_claim_ids: tuple[str, ...]
    suggested_repair: str


@dataclass(frozen=True)
class ScoredBundle:
    """An EvidenceBundle annotated with quality scores, gap reports, and a sufficiency verdict."""

    bundle: EvidenceBundle
    scores: EvidenceScores
    gaps: tuple[GapReport, ...]
    sufficient: bool


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_NORMAL_CONFIDENCE_THRESHOLD = 0.5
_HIGH_RISK_CONFIDENCE_THRESHOLD = 0.75
_MIN_ITEMS_FOR_COMPLETENESS = 2
_STALE_FRESHNESS_THRESHOLD = 0.3
_LOW_AUTHORITY_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# Individual score calculations
# ---------------------------------------------------------------------------


def _score_authority(bundle: EvidenceBundle) -> float:
    """Proxy: presence of known-authority source patterns raises score."""
    if not bundle.items:
        return 0.0
    authority_signals = ("policy", "research", "analytics")
    high_auth = sum(
        1
        for item in bundle.items
        if any(sig in item.citation.retrieval_expert.lower() for sig in authority_signals)
    )
    return round(high_auth / len(bundle.items), 2)


def _score_freshness(bundle: EvidenceBundle) -> float:
    """Proxy: stale signals in content reduce freshness score."""
    if not bundle.items:
        return 0.0
    stale_signals = ("deprecated", "obsolete", "outdated", "legacy", "old version")
    stale = sum(
        1 for item in bundle.items if any(sig in item.content.lower() for sig in stale_signals)
    )
    return round(1.0 - (stale / len(bundle.items)), 2)


def _score_agreement(bundle: EvidenceBundle) -> float:
    """Conflict ratio directly lowers agreement."""
    if not bundle.items:
        return 0.0
    conflict_ratio = bundle.conflict_count / len(bundle.items)
    return round(1.0 - conflict_ratio, 2)


def _score_completeness(bundle: EvidenceBundle, claim_ids: tuple[str, ...]) -> float:
    """Fraction of requested claims that have at least one supporting item."""
    if not claim_ids:
        return 1.0 if len(bundle.items) >= _MIN_ITEMS_FOR_COMPLETENESS else 0.5
    covered = sum(1 for cid in claim_ids if bundle.by_claim(cid))
    return round(covered / len(claim_ids), 2)


def _composite_confidence(
    authority: float,
    freshness: float,
    agreement: float,
    completeness: float,
) -> float:
    """Compute a weighted composite confidence score from the four sub-scores."""
    return round((authority * 0.25 + freshness * 0.25 + agreement * 0.3 + completeness * 0.2), 2)


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------


def _detect_gaps(
    bundle: EvidenceBundle,
    scores: EvidenceScores,
    claim_ids: tuple[str, ...],
) -> tuple[GapReport, ...]:
    """Identify quality gaps in the bundle based on scores and coverage."""
    gaps: list[GapReport] = []

    if not bundle.items:
        gaps.append(
            GapReport(
                gap_type=GapType.MISSING_EVIDENCE,
                description="No evidence items present in the bundle.",
                affected_claim_ids=claim_ids,
                suggested_repair="Run retrieval with broader query expansions.",
            )
        )
        return tuple(gaps)

    if scores.authority < _LOW_AUTHORITY_THRESHOLD:
        gaps.append(
            GapReport(
                gap_type=GapType.LOW_RELEVANCE,
                description=f"Authority score {scores.authority:.2f} is below threshold.",
                affected_claim_ids=(),
                suggested_repair=(
                    "Re-route to policy or research expert for higher-authority sources."
                ),
            )
        )

    if scores.freshness < _STALE_FRESHNESS_THRESHOLD:
        gaps.append(
            GapReport(
                gap_type=GapType.STALE_EVIDENCE,
                description=f"Freshness score {scores.freshness:.2f} indicates stale evidence.",
                affected_claim_ids=(),
                suggested_repair="Re-run with freshness or web expert to get updated sources.",
            )
        )

    if bundle.conflict_count > 0:
        conflicted = tuple(
            str(i)
            for i, item in enumerate(bundle.items)
            if item.conflict_status == ConflictStatus.CONFLICT
        )
        gaps.append(
            GapReport(
                gap_type=GapType.CONFLICTING_EVIDENCE,
                description=f"{bundle.conflict_count} evidence items have conflicting signals.",
                affected_claim_ids=conflicted,
                suggested_repair="Resolve conflicts by authority-filtering or arbitration.",
            )
        )

    if claim_ids:
        uncovered = tuple(cid for cid in claim_ids if not bundle.by_claim(cid))
        if uncovered:
            gaps.append(
                GapReport(
                    gap_type=GapType.INCOMPLETE_COVERAGE,
                    description=f"Claims with no supporting evidence: {uncovered}",
                    affected_claim_ids=uncovered,
                    suggested_repair="Retrieve targeted evidence for each uncovered claim.",
                )
            )

    return tuple(gaps)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def score_bundle(
    bundle: EvidenceBundle,
    claim_ids: tuple[str, ...] = (),
    high_risk: bool = False,
) -> ScoredBundle:
    """Score an EvidenceBundle and return gaps and a sufficiency verdict."""

    authority = _score_authority(bundle)
    freshness = _score_freshness(bundle)
    agreement = _score_agreement(bundle)
    completeness = _score_completeness(bundle, claim_ids)
    confidence = _composite_confidence(authority, freshness, agreement, completeness)

    threshold = _HIGH_RISK_CONFIDENCE_THRESHOLD if high_risk else _NORMAL_CONFIDENCE_THRESHOLD
    reason_parts = [
        f"authority={authority:.2f}",
        f"freshness={freshness:.2f}",
        f"agreement={agreement:.2f}",
        f"completeness={completeness:.2f}",
        f"confidence={confidence:.2f}",
        f"threshold={'high-risk' if high_risk else 'normal'}({threshold})",
    ]

    scores = EvidenceScores(
        authority=authority,
        freshness=freshness,
        agreement=agreement,
        completeness=completeness,
        confidence=confidence,
        reason="; ".join(reason_parts),
    )

    gaps = _detect_gaps(bundle, scores, claim_ids)
    sufficient = confidence >= threshold and not any(
        g.gap_type in (GapType.MISSING_EVIDENCE, GapType.INCOMPLETE_COVERAGE) for g in gaps
    )

    return ScoredBundle(bundle=bundle, scores=scores, gaps=gaps, sufficient=sufficient)
