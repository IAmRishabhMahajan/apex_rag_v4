# US-007: Evidence Scoring and Gap Detection

## User Story

As a user, I want APEX-RAG v5 to judge whether the available evidence is strong enough before answering so that weak, stale, or incomplete evidence triggers repair instead of confident mistakes.

## Scope

Score evidence quality and detect evidence gaps at claim and answer levels.

## Implementation Tasks

- Add authority, freshness, agreement, completeness, and confidence scores to evidence items or bundles.
- Define score ranges and thresholds for normal and high-risk queries.
- Calculate claim-level evidence sufficiency.
- Detect missing evidence, low relevance, stale evidence, conflict, and incomplete coverage.
- Return gap reports that can drive retrieval repair.

## Acceptance Criteria

- Evidence bundles include quality scores with explainable reasons.
- Claims with insufficient evidence are flagged.
- Conflicting evidence lowers confidence or triggers arbitration.
- Freshness-sensitive queries penalize outdated evidence.
- Gap reports identify what evidence is missing.

## Testing Expectations

- Unit tests cover each evidence score.
- Tests cover threshold behavior for normal and high-risk queries.
- Tests verify stale or conflicting evidence produces gap reports.

## Documentation Updates

- Document scoring formulas, thresholds, and gap types.

## Dependencies

- US-004 Evidence Fusion.
- US-005 Validation Mesh.

