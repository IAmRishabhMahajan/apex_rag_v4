"""US-005 Validation Mesh — reusable validators for each pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.apex_rag.evidence_fusion import EvidenceBundle
from src.apex_rag.generation import GeneratedAnswer
from src.apex_rag.query_intelligence import Intent, QueryProfile


class ValidationStatus(str, Enum):
    """Outcome of a pipeline-stage validation check."""

    APPROVED = "approved"
    REJECTED = "rejected"
    REPAIR = "repair"
    ESCALATE = "escalate"


class Severity(str, Enum):
    """Severity level associated with a ValidationResult."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PipelineStage(str, Enum):
    """Pipeline stage that produced a ValidationResult."""

    QUERY = "query"
    RETRIEVAL = "retrieval"
    FUSION = "fusion"
    CLAIM = "claim"
    REASONING = "reasoning"
    GENERATION = "generation"


@dataclass(frozen=True)
class ValidationResult:
    """Structured outcome of a single validation check at a pipeline stage."""

    status: ValidationStatus
    severity: Severity
    stage: PipelineStage
    messages: tuple[str, ...]
    repair_hints: tuple[str, ...]
    affected_records: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """True when the validation outcome is APPROVED."""
        return self.status == ValidationStatus.APPROVED

    @property
    def blocks_downstream(self) -> bool:
        """True when the outcome is REJECTED or ESCALATE, halting downstream stages."""
        return self.status in (ValidationStatus.REJECTED, ValidationStatus.ESCALATE)


class ValidationBlockedError(Exception):
    """Raised when a rejected or escalated ValidationResult blocks downstream processing."""

    def __init__(self, result: ValidationResult) -> None:
        """Carry the full ValidationResult so upstream handlers can inspect details."""
        super().__init__(
            f"Stage '{result.stage.value}' blocked downstream: {'; '.join(result.messages)}"
        )
        self.result = result


def _approved(stage: PipelineStage, message: str = "Validation passed.") -> ValidationResult:
    """Build an APPROVED ValidationResult with INFO severity."""
    return ValidationResult(
        status=ValidationStatus.APPROVED,
        severity=Severity.INFO,
        stage=stage,
        messages=(message,),
        repair_hints=(),
        affected_records=(),
    )


def _rejected(
    stage: PipelineStage,
    messages: tuple[str, ...],
    repair_hints: tuple[str, ...] = (),
    affected: tuple[str, ...] = (),
    severity: Severity = Severity.ERROR,
) -> ValidationResult:
    """Build a REJECTED ValidationResult with the given severity and hints."""
    return ValidationResult(
        status=ValidationStatus.REJECTED,
        severity=severity,
        stage=stage,
        messages=messages,
        repair_hints=repair_hints,
        affected_records=affected,
    )


def _repair(
    stage: PipelineStage,
    messages: tuple[str, ...],
    repair_hints: tuple[str, ...],
    affected: tuple[str, ...] = (),
) -> ValidationResult:
    """Build a REPAIR ValidationResult with WARNING severity."""
    return ValidationResult(
        status=ValidationStatus.REPAIR,
        severity=Severity.WARNING,
        stage=stage,
        messages=messages,
        repair_hints=repair_hints,
        affected_records=affected,
    )


def _escalate(
    stage: PipelineStage,
    messages: tuple[str, ...],
    repair_hints: tuple[str, ...] = (),
    affected: tuple[str, ...] = (),
) -> ValidationResult:
    """Build an ESCALATE ValidationResult with CRITICAL severity."""
    return ValidationResult(
        status=ValidationStatus.ESCALATE,
        severity=Severity.CRITICAL,
        stage=stage,
        messages=messages,
        repair_hints=repair_hints,
        affected_records=affected,
    )


# ---------------------------------------------------------------------------
# Query validator
# ---------------------------------------------------------------------------


def validate_query(profile: QueryProfile) -> ValidationResult:
    """Check entity coverage, intent accuracy, and constraint completeness."""

    if not profile.is_valid:
        return _rejected(
            PipelineStage.QUERY,
            messages=tuple(profile.validation_errors),
            repair_hints=("Rephrase the query with more specific terms.",),
        )

    if profile.intent == Intent.UNKNOWN and not profile.entities:
        return _repair(
            PipelineStage.QUERY,
            messages=("Intent and entities are both absent; query may not route correctly.",),
            repair_hints=("Add explicit subject or action to clarify intent.",),
        )

    return _approved(PipelineStage.QUERY)


# ---------------------------------------------------------------------------
# Fusion validator
# ---------------------------------------------------------------------------

_MAX_CONFLICT_RATIO = 0.5
_MIN_EVIDENCE_ITEMS = 1


def validate_fusion(bundle: EvidenceBundle) -> ValidationResult:
    """Check for empty bundles, excessive conflicts, and missing provenance."""

    if not bundle.items:
        return _rejected(
            PipelineStage.FUSION,
            messages=("Evidence bundle is empty; retrieval produced no results.",),
            repair_hints=("Expand retrieval query or add more expert sources.",),
        )

    total = len(bundle.items)
    conflict_ratio = bundle.conflict_count / total if total else 0.0

    if conflict_ratio >= _MAX_CONFLICT_RATIO:
        return _escalate(
            PipelineStage.FUSION,
            messages=(
                f"{bundle.conflict_count} of {total} evidence items conflict "
                f"({conflict_ratio:.0%}); answer reliability is too low.",
            ),
            repair_hints=("Re-run retrieval with authority-filtered sources.",),
            affected=(f"conflict_ratio={conflict_ratio:.2f}",),
        )

    missing_provenance = [
        str(i)
        for i, item in enumerate(bundle.items)
        if not item.citation.source_id or not item.citation.retrieval_expert
    ]
    if missing_provenance:
        return _repair(
            PipelineStage.FUSION,
            messages=(f"Evidence items missing provenance: {missing_provenance}",),
            repair_hints=("Re-fetch evidence with full citation metadata.",),
            affected=tuple(missing_provenance),
        )

    if bundle.conflict_count > 0:
        return ValidationResult(
            status=ValidationStatus.REPAIR,
            severity=Severity.WARNING,
            stage=PipelineStage.FUSION,
            messages=(f"{bundle.conflict_count} conflicting evidence items detected.",),
            repair_hints=("Review conflicts before using evidence for generation.",),
            affected_records=(),
        )

    return _approved(PipelineStage.FUSION)


# ---------------------------------------------------------------------------
# Claim validator
# ---------------------------------------------------------------------------


def validate_claims(answer: GeneratedAnswer) -> ValidationResult:
    """Check claim-evidence alignment and unsupported claim count."""

    total = len(answer.approved_claims)
    if total == 0:
        return _rejected(
            PipelineStage.CLAIM,
            messages=("No claims were approved; generation output is empty.",),
            repair_hints=("Check that evidence covers the query topic.",),
        )

    unsupported = answer.unsupported_claim_count
    unsupported_ratio = unsupported / total

    if unsupported_ratio >= 0.5:
        return _rejected(
            PipelineStage.CLAIM,
            messages=(
                f"{unsupported} of {total} claims are unsupported "
                f"({unsupported_ratio:.0%}); answer is not sufficiently grounded.",
            ),
            repair_hints=("Retrieve more evidence or narrow the claim set.",),
            affected=(f"unsupported_ratio={unsupported_ratio:.2f}",),
        )

    if unsupported > 0:
        return _repair(
            PipelineStage.CLAIM,
            messages=(f"{unsupported} unsupported claim(s) were excluded from the answer.",),
            repair_hints=("Consider targeted retrieval for the missing claims.",),
        )

    return _approved(PipelineStage.CLAIM)


# ---------------------------------------------------------------------------
# Generation validator
# ---------------------------------------------------------------------------

_MIN_CITATION_LINKS = 1


def validate_generation(answer: GeneratedAnswer) -> ValidationResult:
    """Check citation completeness and unsupported content in final answer."""

    if not answer.text.strip():
        return _rejected(
            PipelineStage.GENERATION,
            messages=("Generated answer text is empty.",),
            repair_hints=("Ensure at least one supported claim feeds generation.",),
        )

    if len(answer.citation_links) < _MIN_CITATION_LINKS:
        return _rejected(
            PipelineStage.GENERATION,
            messages=("Final answer has no citation links; output cannot be verified.",),
            repair_hints=("Ensure supported claims map to evidence before generating.",),
            severity=Severity.CRITICAL,
        )

    if not answer.all_claims_supported:
        return _repair(
            PipelineStage.GENERATION,
            messages=("Answer contains claims without full evidence support.",),
            repair_hints=("Review unsupported claims before delivery.",),
        )

    return _approved(PipelineStage.GENERATION)


# ---------------------------------------------------------------------------
# Guard helper — blocks downstream when result demands it
# ---------------------------------------------------------------------------


def assert_passes(result: ValidationResult) -> None:
    """Raise ValidationBlockedError if the result blocks downstream processing."""
    if result.blocks_downstream:
        raise ValidationBlockedError(result)
