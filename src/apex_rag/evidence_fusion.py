"""US-004 Evidence Fusion — normalise, deduplicate, and conflict-detect evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum


class ConflictStatus(str, Enum):
    NONE = "none"
    CONFLICT = "conflict"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class CitationMetadata:
    source_id: str
    title: str
    url: str
    retrieval_expert: str
    retrieval_query: str


@dataclass
class EvidenceItem:
    content: str
    citation: CitationMetadata
    claim_ids: tuple[str, ...]
    conflict_status: ConflictStatus = ConflictStatus.NONE
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        self.content_hash = hashlib.sha256(self.content.strip().lower().encode()).hexdigest()

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.content.strip():
            errors.append("EvidenceItem content must not be empty.")
        if not self.citation.source_id:
            errors.append("EvidenceItem citation must include a source_id.")
        if not self.citation.retrieval_expert:
            errors.append("EvidenceItem citation must include retrieval_expert.")
        return errors


@dataclass
class EvidenceBundle:
    items: list[EvidenceItem]
    query_id: str

    def by_claim(self, claim_id: str) -> list[EvidenceItem]:
        return [item for item in self.items if claim_id in item.claim_ids]

    @property
    def conflict_count(self) -> int:
        return sum(1 for item in self.items if item.conflict_status == ConflictStatus.CONFLICT)

    @property
    def duplicate_count(self) -> int:
        return sum(1 for item in self.items if item.conflict_status == ConflictStatus.DUPLICATE)


class EvidenceValidationError(Exception):
    """Raised when an EvidenceItem fails its validation contract."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__(f"Evidence validation failed: {'; '.join(errors)}")
        self.errors = errors


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _deduplicate(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Mark later occurrences of identical content as DUPLICATE."""
    seen: set[str] = set()
    result: list[EvidenceItem] = []
    for item in items:
        if item.content_hash in seen:
            item.conflict_status = ConflictStatus.DUPLICATE
        else:
            seen.add(item.content_hash)
        result.append(item)
    return result


# ---------------------------------------------------------------------------
# Conflict detection (simple sentence-level negation heuristic)
# ---------------------------------------------------------------------------

_NEGATION_PAIRS = [
    ({"not ", "no ", "never ", "false", "incorrect"}, {"is ", "yes ", "true", "correct"}),
]


def _content_conflicts(a: str, b: str) -> bool:
    """Heuristic: texts conflict when one asserts a negation the other does not."""
    al, bl = a.lower(), b.lower()
    for neg_signals, pos_signals in _NEGATION_PAIRS:
        a_has_neg = any(s in al for s in neg_signals)
        b_has_neg = any(s in bl for s in neg_signals)
        a_has_pos = any(s in al for s in pos_signals)
        b_has_pos = any(s in bl for s in pos_signals)
        if (a_has_neg and b_has_pos and not b_has_neg) or (
            b_has_neg and a_has_pos and not a_has_neg
        ):
            return True
    return False


def _detect_conflicts(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Flag pairs of items that contain contradictory signals about the same topic."""
    for i, item_a in enumerate(items):
        if item_a.conflict_status != ConflictStatus.NONE:
            continue
        for item_b in items[i + 1 :]:
            if item_b.conflict_status != ConflictStatus.NONE:
                continue
            if _content_conflicts(item_a.content, item_b.content):
                item_a.conflict_status = ConflictStatus.CONFLICT
                item_b.conflict_status = ConflictStatus.CONFLICT
    return items


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fuse_evidence(
    raw_items: Sequence[EvidenceItem],
    query_id: str,
) -> EvidenceBundle:
    """Validate, deduplicate, and conflict-detect a batch of raw evidence items."""

    validated: list[EvidenceItem] = []
    for item in raw_items:
        errors = item.validate()
        if errors:
            raise EvidenceValidationError(errors)
        validated.append(item)

    deduped = _deduplicate(validated)
    processed = _detect_conflicts(deduped)

    return EvidenceBundle(items=processed, query_id=query_id)
