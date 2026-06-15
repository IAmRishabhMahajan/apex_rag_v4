"""US-011 APEX-Eval Framework — metrics and reporting for the full RAG pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from src.apex_rag.evidence_fusion import EvidenceBundle
from src.apex_rag.generation import GeneratedAnswer
from src.apex_rag.retrieval_repair import RepairResult


class MetricCategory(str, Enum):
    """High-level grouping for the different metric families."""

    RETRIEVAL = "retrieval"
    EVIDENCE = "evidence"
    CLAIM = "claim"
    RECOVERY = "recovery"
    REASONING = "reasoning"
    FINAL = "final"


@dataclass(frozen=True)
class RetrievalMetrics:
    """Standard IR metrics measured at rank k: Recall, Precision, MRR, NDCG."""

    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg: float
    k: int

    def __post_init__(self) -> None:
        """Validate that all metric values are in [0, 1]."""
        for attr in ("recall_at_k", "precision_at_k", "mrr", "ndcg"):
            val = getattr(self, attr)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{attr} must be in [0, 1], got {val}")


@dataclass(frozen=True)
class EvidenceMetrics:
    """Coverage, precision, and recall of retrieved evidence against a ground-truth set."""

    coverage: float
    precision: float
    recall: float

    def __post_init__(self) -> None:
        """Validate that all metric values are in [0, 1]."""
        for attr in ("coverage", "precision", "recall"):
            val = getattr(self, attr)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{attr} must be in [0, 1], got {val}")


@dataclass(frozen=True)
class ClaimMetrics:
    """Support and unsupported rates across all approved claims."""

    support_rate: float
    unsupported_rate: float
    total_claims: int

    def __post_init__(self) -> None:
        """Validate that support_rate + unsupported_rate sums to 1.0."""
        if abs(self.support_rate + self.unsupported_rate - 1.0) > 1e-6:
            raise ValueError("support_rate + unsupported_rate must equal 1.0")


@dataclass(frozen=True)
class RecoveryMetrics:
    """Metrics describing how well the repair loop diagnosed and fixed retrieval failures."""

    success_rate: float
    failure_detection_accuracy: float
    total_repair_attempts: int
    successful_repairs: int

    def __post_init__(self) -> None:
        """Validate that rate metrics are in [0, 1]."""
        for attr in ("success_rate", "failure_detection_accuracy"):
            val = getattr(self, attr)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{attr} must be in [0, 1], got {val}")


@dataclass(frozen=True)
class ReasoningMetrics:
    """Logical consistency and claim completeness of the reasoning path."""

    logical_consistency: float
    claim_completeness: float

    def __post_init__(self) -> None:
        """Validate that both metrics are in [0, 1]."""
        for attr in ("logical_consistency", "claim_completeness"):
            val = getattr(self, attr)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{attr} must be in [0, 1], got {val}")


@dataclass(frozen=True)
class FinalMetrics:
    """End-to-end answer quality: faithfulness, groundedness, relevance, and composite quality."""

    faithfulness: float
    groundedness: float
    relevance: float
    answer_quality: float

    def __post_init__(self) -> None:
        """Validate that all final metrics are in [0, 1]."""
        for attr in ("faithfulness", "groundedness", "relevance", "answer_quality"):
            val = getattr(self, attr)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{attr} must be in [0, 1], got {val}")


@dataclass
class QueryEvalResult:
    """All metric families for a single evaluated query."""

    query_id: str
    query_text: str
    retrieval: RetrievalMetrics
    evidence: EvidenceMetrics
    claims: ClaimMetrics
    recovery: RecoveryMetrics | None
    reasoning: ReasoningMetrics
    final: FinalMetrics
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AggregateReport:
    """Averaged metrics across all evaluated queries plus per-query breakdown."""

    total_queries: int
    avg_recall_at_k: float
    avg_precision_at_k: float
    avg_mrr: float
    avg_ndcg: float
    avg_evidence_coverage: float
    avg_claim_support_rate: float
    avg_unsupported_rate: float
    avg_faithfulness: float
    avg_groundedness: float
    avg_answer_quality: float
    recovery_success_rate: float
    per_query: tuple[QueryEvalResult, ...]


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------


def compute_recall_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k retrieved results."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(relevant_ids & top_k) / len(relevant_ids)


def compute_precision_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    """Fraction of top-k retrieved results that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(1 for d in top_k if d in relevant_ids) / k


def compute_mrr(relevant_ids: set[str], retrieved_ids: list[str]) -> float:
    """Mean Reciprocal Rank — reciprocal of the rank of the first relevant result."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_ndcg(relevant_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at k (binary relevance)."""
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1)
        if doc_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(
    relevant_ids: set[str],
    retrieved_ids: list[str],
    k: int = 5,
) -> RetrievalMetrics:
    """Bundle Recall@K, Precision@K, MRR, and NDCG into a RetrievalMetrics object."""
    return RetrievalMetrics(
        recall_at_k=compute_recall_at_k(relevant_ids, retrieved_ids, k),
        precision_at_k=compute_precision_at_k(relevant_ids, retrieved_ids, k),
        mrr=compute_mrr(relevant_ids, retrieved_ids),
        ndcg=compute_ndcg(relevant_ids, retrieved_ids, k),
        k=k,
    )


def compute_evidence_metrics(
    bundle: EvidenceBundle,
    relevant_source_ids: set[str],
) -> EvidenceMetrics:
    """Compute coverage, precision, and recall of the bundle against ground-truth source IDs."""
    retrieved_ids = {item.citation.source_id for item in bundle.items}
    if not retrieved_ids:
        return EvidenceMetrics(coverage=0.0, precision=0.0, recall=0.0)
    precision = (
        len(retrieved_ids & relevant_source_ids) / len(retrieved_ids) if retrieved_ids else 0.0
    )
    recall = (
        len(retrieved_ids & relevant_source_ids) / len(relevant_source_ids)
        if relevant_source_ids
        else 0.0
    )
    coverage = min(1.0, len(retrieved_ids) / max(1, len(relevant_source_ids)))
    return EvidenceMetrics(coverage=coverage, precision=precision, recall=recall)


def compute_claim_metrics(answer: GeneratedAnswer) -> ClaimMetrics:
    """Compute support and unsupported rates from a GeneratedAnswer."""
    total = len(answer.approved_claims)
    if total == 0:
        return ClaimMetrics(support_rate=1.0, unsupported_rate=0.0, total_claims=0)
    unsupported = answer.unsupported_claim_count
    supported = total - unsupported
    return ClaimMetrics(
        support_rate=supported / total,
        unsupported_rate=unsupported / total,
        total_claims=total,
    )


def compute_recovery_metrics(repair_result: RepairResult | None) -> RecoveryMetrics | None:
    """Compute recovery metrics from a RepairResult, or return None if no repair was run."""
    if repair_result is None:
        return None
    total = len(repair_result.attempts)
    successful = sum(1 for a in repair_result.attempts if a.confidence_after > a.confidence_before)
    success_rate = repair_result.final_confidence if repair_result.succeeded else 0.0
    detection_accuracy = successful / total if total > 0 else 0.0
    return RecoveryMetrics(
        success_rate=min(1.0, success_rate),
        failure_detection_accuracy=detection_accuracy,
        total_repair_attempts=total,
        successful_repairs=successful,
    )


def compute_reasoning_metrics(answer: GeneratedAnswer) -> ReasoningMetrics:
    """Estimate reasoning quality from answer properties (no LLM call)."""
    total_claims = len(answer.approved_claims)
    supported = total_claims - answer.unsupported_claim_count
    completeness = supported / total_claims if total_claims > 0 else 0.0
    consistency = 1.0 if answer.all_claims_supported else max(0.0, completeness - 0.1)
    return ReasoningMetrics(
        logical_consistency=round(consistency, 4),
        claim_completeness=round(completeness, 4),
    )


def compute_final_metrics(
    answer: GeneratedAnswer,
    bundle: EvidenceBundle,
    relevant_source_ids: set[str],
) -> FinalMetrics:
    """Compute final answer quality metrics."""
    em = compute_evidence_metrics(bundle, relevant_source_ids)
    cm = compute_claim_metrics(answer)
    faithfulness = cm.support_rate
    groundedness = em.coverage * faithfulness
    relevance = em.recall
    quality = (faithfulness + groundedness + relevance) / 3.0
    return FinalMetrics(
        faithfulness=round(faithfulness, 4),
        groundedness=round(min(1.0, groundedness), 4),
        relevance=round(relevance, 4),
        answer_quality=round(quality, 4),
    )


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def build_aggregate_report(results: list[QueryEvalResult]) -> AggregateReport:
    """Aggregate per-query results into a summary report."""
    n = len(results)
    if n == 0:
        return AggregateReport(
            total_queries=0,
            avg_recall_at_k=0.0,
            avg_precision_at_k=0.0,
            avg_mrr=0.0,
            avg_ndcg=0.0,
            avg_evidence_coverage=0.0,
            avg_claim_support_rate=0.0,
            avg_unsupported_rate=0.0,
            avg_faithfulness=0.0,
            avg_groundedness=0.0,
            avg_answer_quality=0.0,
            recovery_success_rate=0.0,
            per_query=(),
        )

    def _avg(values: list[float]) -> float:
        """Compute the arithmetic mean of a list, returning 0.0 for empty lists."""
        return sum(values) / len(values) if values else 0.0

    recovery_results = [r.recovery for r in results if r.recovery is not None]
    recovery_success = _avg([r.success_rate for r in recovery_results]) if recovery_results else 0.0

    return AggregateReport(
        total_queries=n,
        avg_recall_at_k=_avg([r.retrieval.recall_at_k for r in results]),
        avg_precision_at_k=_avg([r.retrieval.precision_at_k for r in results]),
        avg_mrr=_avg([r.retrieval.mrr for r in results]),
        avg_ndcg=_avg([r.retrieval.ndcg for r in results]),
        avg_evidence_coverage=_avg([r.evidence.coverage for r in results]),
        avg_claim_support_rate=_avg([r.claims.support_rate for r in results]),
        avg_unsupported_rate=_avg([r.claims.unsupported_rate for r in results]),
        avg_faithfulness=_avg([r.final.faithfulness for r in results]),
        avg_groundedness=_avg([r.final.groundedness for r in results]),
        avg_answer_quality=_avg([r.final.answer_quality for r in results]),
        recovery_success_rate=recovery_success,
        per_query=tuple(results),
    )


def format_report(report: AggregateReport) -> str:
    """Produce a human-readable plaintext evaluation report."""
    lines = [
        "=== APEX-Eval Report ===",
        f"Total queries evaluated : {report.total_queries}",
        "",
        "--- Retrieval ---",
        f"  Recall@K              : {report.avg_recall_at_k:.4f}",
        f"  Precision@K           : {report.avg_precision_at_k:.4f}",
        f"  MRR                   : {report.avg_mrr:.4f}",
        f"  NDCG                  : {report.avg_ndcg:.4f}",
        "",
        "--- Evidence ---",
        f"  Coverage              : {report.avg_evidence_coverage:.4f}",
        "",
        "--- Claims ---",
        f"  Support rate          : {report.avg_claim_support_rate:.4f}",
        f"  Unsupported rate      : {report.avg_unsupported_rate:.4f}",
        "",
        "--- Final ---",
        f"  Faithfulness          : {report.avg_faithfulness:.4f}",
        f"  Groundedness          : {report.avg_groundedness:.4f}",
        f"  Answer quality        : {report.avg_answer_quality:.4f}",
        "",
        "--- Recovery ---",
        f"  Recovery success rate : {report.recovery_success_rate:.4f}",
    ]
    return "\n".join(lines)
