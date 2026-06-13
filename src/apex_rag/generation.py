"""US-009 Grounded Reasoning and Generation — answers built from approved evidence only."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.apex_rag.evidence_fusion import ConflictStatus, EvidenceBundle, EvidenceItem


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    PARTIAL = "partial"


@dataclass(frozen=True)
class CitationLink:
    claim_text: str
    evidence_source_id: str
    evidence_title: str
    evidence_url: str


@dataclass
class ApprovedClaim:
    text: str
    status: ClaimStatus
    evidence_items: tuple[EvidenceItem, ...]
    citation_links: tuple[CitationLink, ...]

    @property
    def is_supported(self) -> bool:
        return self.status == ClaimStatus.SUPPORTED


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    approved_claims: tuple[ApprovedClaim, ...]
    citation_links: tuple[CitationLink, ...]
    has_limitations: bool
    limitation_note: str

    @property
    def unsupported_claim_count(self) -> int:
        return sum(1 for c in self.approved_claims if c.status == ClaimStatus.UNSUPPORTED)

    @property
    def all_claims_supported(self) -> bool:
        return all(c.is_supported for c in self.approved_claims)


class GroundingError(Exception):
    """Raised when generation cannot be completed due to missing citation mappings."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Grounding failed: {reason}")
        self.reason = reason


# ---------------------------------------------------------------------------
# Claim approval
# ---------------------------------------------------------------------------


def _build_citation_link(claim_text: str, item: EvidenceItem) -> CitationLink:
    return CitationLink(
        claim_text=claim_text,
        evidence_source_id=item.citation.source_id,
        evidence_title=item.citation.title,
        evidence_url=item.citation.url,
    )


def approve_claims(
    candidate_claims: list[str],
    bundle: EvidenceBundle,
) -> list[ApprovedClaim]:
    """Match each candidate claim to supporting evidence items."""

    approved: list[ApprovedClaim] = []

    for claim_text in candidate_claims:
        claim_lower = claim_text.lower()
        supporting = [
            item
            for item in bundle.items
            if item.conflict_status == ConflictStatus.NONE
            and any(word in item.content.lower() for word in claim_lower.split() if len(word) > 4)
        ]

        if supporting:
            links = tuple(_build_citation_link(claim_text, item) for item in supporting)
            approved.append(
                ApprovedClaim(
                    text=claim_text,
                    status=ClaimStatus.SUPPORTED,
                    evidence_items=tuple(supporting),
                    citation_links=links,
                )
            )
        else:
            approved.append(
                ApprovedClaim(
                    text=claim_text,
                    status=ClaimStatus.UNSUPPORTED,
                    evidence_items=(),
                    citation_links=(),
                )
            )

    return approved


# ---------------------------------------------------------------------------
# Answer assembly
# ---------------------------------------------------------------------------

_UNABLE_TO_ANSWER = "This question cannot be answered because no approved evidence is available."
_LIMITATION_PARTIAL = (
    "Note: some claims could not be fully supported by available evidence "
    "and have been excluded from this answer."
)
_LIMITATION_CONFLICT = (
    "Note: conflicting evidence was detected; answer reflects only non-conflicting sources."
)


def generate_answer(
    candidate_claims: list[str],
    bundle: EvidenceBundle,
) -> GeneratedAnswer:
    """Build a grounded answer from approved claims and mapped evidence.

    Raises GroundingError when required citation mappings are missing entirely.
    """

    if not bundle.items:
        raise GroundingError("Evidence bundle is empty; cannot generate a grounded answer.")

    approved = approve_claims(candidate_claims, bundle)

    supported = [c for c in approved if c.is_supported]
    unsupported = [c for c in approved if not c.is_supported]

    if not supported:
        raise GroundingError(
            "No candidate claims could be matched to approved evidence; generation fails closed."
        )

    # Build answer text from supported claims only
    answer_parts = [c.text for c in supported]
    answer_text = " ".join(answer_parts)

    # Collect all citation links across supported claims
    all_links: list[CitationLink] = []
    seen_pairs: set[tuple[str, str]] = set()
    for claim in supported:
        for link in claim.citation_links:
            key = (link.claim_text, link.evidence_source_id)
            if key not in seen_pairs:
                seen_pairs.add(key)
                all_links.append(link)

    has_limitations = bool(unsupported) or bundle.conflict_count > 0
    if unsupported and bundle.conflict_count > 0:
        limitation_note = f"{_LIMITATION_PARTIAL} {_LIMITATION_CONFLICT}"
    elif unsupported:
        limitation_note = _LIMITATION_PARTIAL
    elif bundle.conflict_count > 0:
        limitation_note = _LIMITATION_CONFLICT
    else:
        limitation_note = ""

    return GeneratedAnswer(
        text=answer_text,
        approved_claims=tuple(approved),
        citation_links=tuple(all_links),
        has_limitations=has_limitations,
        limitation_note=limitation_note,
    )
