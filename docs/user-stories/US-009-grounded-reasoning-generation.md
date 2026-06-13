# US-009: Grounded Reasoning and Generation

## User Story

As a user, I want final answers generated only from approved claims and linked evidence so that each conclusion is grounded and citeable.

## Scope

Create the reasoning and constrained generation path that turns approved evidence into a final answer.

## Implementation Tasks

- Build a reasoning layer that maps conclusions to supporting claims and evidence.
- Define an approved-claims input contract for generation.
- Generate answers from approved claims, evidence links, and citation mappings only.
- Prevent unsupported claims from entering generated output.
- Include confidence calibration and limitations when evidence is incomplete.
- Produce citation mappings that can be inspected after answer generation.

## Acceptance Criteria

- Every final answer claim maps to approved evidence.
- Unsupported claims are omitted or explicitly marked as unavailable.
- Citations are present for evidence-backed statements.
- Answer limitations are surfaced when evidence coverage is incomplete.
- Generation fails closed when required citation mapping is missing.

## Testing Expectations

- Unit tests verify unsupported claims are blocked.
- Tests verify citations map to source evidence.
- Tests cover incomplete evidence and unable-to-answer behavior.

## Documentation Updates

- Document generation input contracts and citation mapping format.

## Dependencies

- US-004 Evidence Fusion.
- US-005 Validation Mesh.
- US-007 Evidence Scoring and Gap Detection.

