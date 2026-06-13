# US-008: Retrieval Repair Loop

## User Story

As a user, I want APEX-RAG v5 to recover from poor retrieval by rewriting queries, switching experts, or expanding scope so that the system can self-correct before answering.

## Scope

Implement bounded iterative retrieval repair using failure classification and recovery strategies.

## Implementation Tasks

- Define retrieval failure classes for no evidence, low relevance, conflicting evidence, outdated evidence, incomplete coverage, and wrong expert selection.
- Map failure classes to recovery strategies.
- Implement query expansion for no-evidence failures.
- Implement query rewrite for low-relevance failures.
- Implement evidence arbitration for conflicts.
- Implement freshness retrieval for outdated evidence.
- Implement expanded claim decomposition for incomplete coverage.
- Implement expert rerouting for wrong-expert failures.
- Enforce bounded loop controls such as max iterations and confidence thresholds.

## Acceptance Criteria

- Retrieval repair stops when confidence reaches the configured threshold.
- Retrieval repair stops at the maximum iteration count.
- Each loop records the failure type, recovery action, and outcome.
- Wrong expert selection can trigger rerouting.
- Persistent failure returns a useful unable-to-answer state rather than unsupported output.

## Testing Expectations

- Unit tests cover each failure-to-recovery mapping.
- Tests verify loop bounds are enforced.
- Tests verify persistent failure produces a grounded fallback response.

## Documentation Updates

- Document failure classes, recovery strategies, and loop configuration.

## Dependencies

- US-003 Expert Retrieval Routing.
- US-006 Complex Query Reasoning Path.
- US-007 Evidence Scoring and Gap Detection.

