"""US-010 Risk Assessment, Critique, and Verification — safeguards for high-risk answers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from src.apex_rag.evidence_fusion import EvidenceBundle
from src.apex_rag.generation import GeneratedAnswer
from src.apex_rag.query_intelligence import QueryProfile


class RiskCategory(str, Enum):
    """Domain categories that require special handling and disclaimers."""

    NORMAL = "normal"
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCIAL = "financial"
    COMPLIANCE = "compliance"


@dataclass(frozen=True)
class RiskAssessment:
    """Result of classifying a query into a risk category."""

    category: RiskCategory
    is_high_risk: bool
    signals: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class CritiqueResult:
    """Outcome of critiquing a generated answer for unsupported or hedged content."""

    passed: bool
    issues: tuple[str, ...]
    removed_sentences: tuple[str, ...]
    disclaimer: str
    confidence_note: str


@dataclass(frozen=True)
class VerifiedAnswer:
    """Final answer after risk assessment and critique, with escalation flag."""

    text: str
    risk_assessment: RiskAssessment
    critique: CritiqueResult
    requires_escalation: bool


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------

_RISK_CATEGORY_PATTERNS: dict[RiskCategory, re.Pattern[str]] = {
    RiskCategory.MEDICAL: re.compile(
        r"\b(diagnos|treatment|medication|dose|symptom|disease|clinical|therapy|prescription)\b",
        re.IGNORECASE,
    ),
    RiskCategory.LEGAL: re.compile(
        r"\b(lawsuit|liable|sue|legal\s+action|attorney|contract|indemnity|jurisdiction|statute)\b",
        re.IGNORECASE,
    ),
    RiskCategory.FINANCIAL: re.compile(
        r"\b(invest|stock|buy|sell|portfolio|returns?|dividend|fund|securities|tax\s+advice)\b",
        re.IGNORECASE,
    ),
    RiskCategory.COMPLIANCE: re.compile(
        r"\b(gdpr|hipaa|sox|pci.?dss|regulation|compliance|audit|data\s+protection|breach)\b",
        re.IGNORECASE,
    ),
}

_DISCLAIMERS: dict[RiskCategory, str] = {
    RiskCategory.MEDICAL: (
        "This information is for general reference only and does not constitute medical advice. "
        "Consult a qualified healthcare professional before making any medical decisions."
    ),
    RiskCategory.LEGAL: (
        "This information is for general reference only and does not constitute legal advice. "
        "Consult a qualified legal professional for advice specific to your situation."
    ),
    RiskCategory.FINANCIAL: (
        "This information is for general reference only and does not constitute financial advice. "
        "Consult a qualified financial advisor before making any investment decisions."
    ),
    RiskCategory.COMPLIANCE: (
        "Compliance requirements vary by jurisdiction and context. "
        "Verify with a qualified compliance professional before acting on this information."
    ),
}


def assess_risk(profile: QueryProfile) -> RiskAssessment:
    """Classify a query into a risk category based on query content and risk signals."""

    query = profile.raw_query
    detected: list[RiskCategory] = []

    for category, pattern in _RISK_CATEGORY_PATTERNS.items():
        if pattern.search(query):
            detected.append(category)

    # Also honour risk signals already detected in the query profile
    for signal in profile.risk_signals:
        if signal == "medical_advice" and RiskCategory.MEDICAL not in detected:
            detected.append(RiskCategory.MEDICAL)
        elif signal == "legal_advice" and RiskCategory.LEGAL not in detected:
            detected.append(RiskCategory.LEGAL)
        elif signal == "financial_advice" and RiskCategory.FINANCIAL not in detected:
            detected.append(RiskCategory.FINANCIAL)

    if not detected:
        return RiskAssessment(
            category=RiskCategory.NORMAL,
            is_high_risk=False,
            signals=(),
            reason="No high-risk signals detected.",
        )

    primary = detected[0]
    return RiskAssessment(
        category=primary,
        is_high_risk=True,
        signals=tuple(c.value for c in detected),
        reason=f"High-risk category detected: {', '.join(c.value for c in detected)}.",
    )


# ---------------------------------------------------------------------------
# Answer critique
# ---------------------------------------------------------------------------

_UNSUPPORTED_HEDGE_PHRASES = (
    "it is widely believed",
    "many experts say",
    "some sources suggest",
    "it has been reported",
    "it is commonly known",
)

_CONTRADICTION_SIGNALS = ("however", "on the other hand", "but", "although", "nevertheless")


def _sentence_has_evidence(sentence: str, bundle: EvidenceBundle) -> bool:
    """Return True when at least one evidence item shares significant words with the sentence."""
    words = [w for w in sentence.lower().split() if len(w) > 4]
    if not words:
        return True  # very short sentences pass by default
    return any(any(word in item.content.lower() for word in words) for item in bundle.items)


def critique_answer(answer: GeneratedAnswer, bundle: EvidenceBundle) -> CritiqueResult:
    """Critique a generated answer against the evidence bundle."""

    issues: list[str] = []
    removed: list[str] = []

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.text) if s.strip()]
    kept: list[str] = []

    for sentence in sentences:
        # Remove sentences that contain unsupported hedge phrases
        hedged = any(phrase in sentence.lower() for phrase in _UNSUPPORTED_HEDGE_PHRASES)
        if hedged:
            removed.append(sentence)
            issues.append(f"Removed unsupported hedge: '{sentence[:60]}...'")
            continue

        # Remove sentences with no supporting evidence
        if not _sentence_has_evidence(sentence, bundle):
            removed.append(sentence)
            issues.append(f"Removed unsupported sentence: '{sentence[:60]}...'")
            continue

        kept.append(sentence)

    # Check for acknowledged contradictions
    contradictions = [s for s in kept if any(sig in s.lower() for sig in _CONTRADICTION_SIGNALS)]
    if contradictions:
        issues.append(
            f"{len(contradictions)} sentence(s) contain contradiction signals — "
            "verify they are appropriately acknowledged."
        )

    confidence_note = ""
    if answer.has_limitations:
        confidence_note = "Answer confidence is limited; some claims lack full evidence support."

    return CritiqueResult(
        passed=len(removed) == 0,
        issues=tuple(issues),
        removed_sentences=tuple(removed),
        disclaimer="",
        confidence_note=confidence_note,
    )


# ---------------------------------------------------------------------------
# Final verification
# ---------------------------------------------------------------------------


def verify_answer(
    answer: GeneratedAnswer,
    bundle: EvidenceBundle,
    profile: QueryProfile,
) -> VerifiedAnswer:
    """Run risk assessment, critique, and produce a verified answer."""

    assessment = assess_risk(profile)

    if not assessment.is_high_risk:
        return VerifiedAnswer(
            text=answer.text,
            risk_assessment=assessment,
            critique=CritiqueResult(
                passed=True,
                issues=(),
                removed_sentences=(),
                disclaimer="",
                confidence_note="",
            ),
            requires_escalation=False,
        )

    critique = critique_answer(answer, bundle)
    disclaimer = _DISCLAIMERS.get(assessment.category, "")

    kept_sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", answer.text)
        if s.strip() and s.strip() not in critique.removed_sentences
    ]
    verified_text = " ".join(kept_sentences)

    requires_escalation = not critique.passed and len(critique.removed_sentences) > 2

    return VerifiedAnswer(
        text=verified_text,
        risk_assessment=assessment,
        critique=CritiqueResult(
            passed=critique.passed,
            issues=critique.issues,
            removed_sentences=critique.removed_sentences,
            disclaimer=disclaimer,
            confidence_note=critique.confidence_note,
        ),
        requires_escalation=requires_escalation,
    )
