# US-011: APEX-Eval Framework

## User Story

As a system maintainer, I want a dedicated evaluation framework so that retrieval quality, evidence quality, reasoning, recovery, and final answers can be measured over time.

## Scope

Implement APEX-Eval metrics and reporting for the full RAG pipeline.

## Implementation Tasks

- Define evaluation datasets and expected outputs for representative query types.
- Implement retrieval metrics: Recall@K, Precision@K, MRR, and NDCG.
- Implement evidence metrics: evidence coverage, precision, and recall.
- Implement claim metrics: claim support rate and unsupported claim rate.
- Implement recovery metrics: recovery success rate, failure detection accuracy, and recovery latency.
- Implement reasoning metrics: logical consistency and claim completeness.
- Implement final metrics: faithfulness, groundedness, relevance, and answer quality.
- Produce machine-readable and human-readable evaluation reports.

## Acceptance Criteria

- Evaluation can run locally with a documented command.
- Reports include per-query and aggregate metrics.
- Unsupported claim rate is visible in the report.
- Recovery attempts are measured separately from initial retrieval.
- Evaluation failures provide enough detail to debug regressions.

## Testing Expectations

- Unit tests cover metric calculations.
- Snapshot or fixture tests cover report structure.
- Regression tests cover representative query examples.

## Documentation Updates

- Document evaluation commands, datasets, metrics, and report interpretation.

## Dependencies

- US-004 Evidence Fusion.
- US-008 Retrieval Repair Loop.
- US-009 Grounded Reasoning and Generation.

