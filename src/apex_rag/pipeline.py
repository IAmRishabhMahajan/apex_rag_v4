"""APEX-RAG v5 full pipeline — wires all modules into a single callable entry point."""

from __future__ import annotations

from dataclasses import dataclass

from src.apex_rag.apex_eval import (
    AggregateReport,
    QueryEvalResult,
    build_aggregate_report,
    compute_claim_metrics,
    compute_evidence_metrics,
    compute_final_metrics,
    compute_reasoning_metrics,
    compute_recovery_metrics,
    compute_retrieval_metrics,
)
from src.apex_rag.complex_reasoning import ComplexReasoningResult, run_complex_reasoning
from src.apex_rag.evidence_fusion import EvidenceBundle, fuse_evidence
from src.apex_rag.evidence_scoring import ScoredBundle, score_bundle
from src.apex_rag.generation import GeneratedAnswer, GroundingError, generate_answer
from src.apex_rag.query_intelligence import QueryProfile, build_query_profile
from src.apex_rag.retrieval_planning import RetrievalPlan, plan_retrieval
from src.apex_rag.retrieval_repair import RepairResult, run_repair_loop
from src.apex_rag.risk_verification import VerifiedAnswer, verify_answer
from src.apex_rag.validation_mesh import (
    ValidationResult,
    validate_claims,
    validate_fusion,
    validate_generation,
    validate_query,
)


@dataclass(frozen=True)
class PipelineResult:
    """Aggregated output of a single run_pipeline() call.

    Carries every intermediate stage result — profile, plan, bundle, scored
    evidence, optional repair, reasoning, answer, verified answer, all four
    validation results, and the APEX-Eval metrics — so callers can inspect
    any pipeline step.
    """

    query_profile: QueryProfile
    retrieval_plan: RetrievalPlan
    bundle: EvidenceBundle
    scored: ScoredBundle
    repair: RepairResult | None
    reasoning: ComplexReasoningResult
    answer: GeneratedAnswer
    verified: VerifiedAnswer
    query_validation: ValidationResult
    fusion_validation: ValidationResult
    claim_validation: ValidationResult
    generation_validation: ValidationResult
    eval_result: QueryEvalResult


def run_pipeline(
    raw_query: str,
    raw_evidence_items: list[dict[str, str]],
    *,
    query_id: str = "q1",
    candidate_claims: list[str] | None = None,
    relevant_source_ids: set[str] | None = None,
    high_risk: bool = False,
) -> PipelineResult:
    """Run the full APEX-RAG v5 pipeline end-to-end.

    Parameters
    ----------
    raw_query:
        The user's natural-language question.
    raw_evidence_items:
        List of dicts with keys: content, source_id, title, url,
        retrieval_expert, retrieval_query, claim_ids (optional).
    query_id:
        Identifier for this query run.
    candidate_claims:
        Candidate answer claims to ground and generate from.
        Defaults to a single claim echoing the query.
    relevant_source_ids:
        Ground-truth source IDs for eval metrics (optional).
    high_risk:
        Whether to apply high-risk evidence scoring thresholds.
    """
    import string

    if candidate_claims is None:
        candidate_claims = [raw_query.rstrip(string.punctuation)]
    if relevant_source_ids is None:
        relevant_source_ids = set()

    # --- Stage 1: Query Intelligence ---
    profile = build_query_profile(raw_query)
    q_val = validate_query(profile)

    # --- Stage 2: Retrieval Planning ---
    plan = plan_retrieval(profile)

    # --- Stage 3: Evidence Fusion ---
    from src.apex_rag.evidence_fusion import CitationMetadata, EvidenceItem

    items = []
    for raw in raw_evidence_items:
        citation = CitationMetadata(
            source_id=raw["source_id"],
            title=raw.get("title", ""),
            url=raw.get("url", ""),
            retrieval_expert=raw.get("retrieval_expert", "search"),
            retrieval_query=raw.get("retrieval_query", raw_query),
        )
        claim_ids_raw = raw.get("claim_ids", "")
        claim_ids = tuple(claim_ids_raw.split(",")) if claim_ids_raw else ()
        items.append(
            EvidenceItem(
                content=raw["content"],
                citation=citation,
                claim_ids=claim_ids,
            )
        )
    bundle = fuse_evidence(items, query_id)
    f_val = validate_fusion(bundle)

    # --- Stage 4: Evidence Scoring + Repair ---
    scored = score_bundle(bundle, (), high_risk)
    repair: RepairResult | None = None
    if not scored.sufficient:
        repair = run_repair_loop(profile, bundle, high_risk=high_risk)
        bundle = repair.final_bundle
        scored = score_bundle(bundle, (), high_risk)

    # --- Stage 5: Complex Reasoning (conditional) ---
    reasoning = run_complex_reasoning(profile, bundle)

    # --- Stage 6: Grounded Generation (fails closed on no evidence match) ---
    try:
        answer = generate_answer(candidate_claims, bundle)
    except GroundingError:
        answer = GeneratedAnswer(
            text="Unable to generate a grounded answer: no supported evidence found.",
            approved_claims=(),
            citation_links=(),
            has_limitations=True,
            limitation_note="All candidate claims lack supporting evidence.",
        )
    c_val = validate_claims(answer)

    # --- Stage 7: Risk Verification ---
    verified = verify_answer(answer, bundle, profile)

    # --- Stage 8: Generation Validation ---
    g_val = validate_generation(answer)

    # --- Stage 9: APEX-Eval ---
    retrieved_ids = [item.citation.source_id for item in bundle.items]
    retrieval_metrics = compute_retrieval_metrics(relevant_source_ids, retrieved_ids, k=5)
    evidence_metrics = compute_evidence_metrics(bundle, relevant_source_ids)
    claim_metrics = compute_claim_metrics(answer)
    recovery_metrics = compute_recovery_metrics(repair)
    reasoning_metrics = compute_reasoning_metrics(answer)
    final_metrics = compute_final_metrics(answer, bundle, relevant_source_ids)

    eval_result = QueryEvalResult(
        query_id=query_id,
        query_text=raw_query,
        retrieval=retrieval_metrics,
        evidence=evidence_metrics,
        claims=claim_metrics,
        recovery=recovery_metrics,
        reasoning=reasoning_metrics,
        final=final_metrics,
    )

    return PipelineResult(
        query_profile=profile,
        retrieval_plan=plan,
        bundle=bundle,
        scored=scored,
        repair=repair,
        reasoning=reasoning,
        answer=answer,
        verified=verified,
        query_validation=q_val,
        fusion_validation=f_val,
        claim_validation=c_val,
        generation_validation=g_val,
        eval_result=eval_result,
    )


def run_batch(
    queries: list[tuple[str, list[dict[str, str]]]],
    **pipeline_kwargs: object,
) -> AggregateReport:
    """Run the pipeline over a batch of queries and return an aggregate eval report."""
    results = []
    for raw_query, raw_items in queries:
        result = run_pipeline(raw_query, raw_items, **pipeline_kwargs)  # type: ignore[arg-type]
        results.append(result.eval_result)
    return build_aggregate_report(results)
