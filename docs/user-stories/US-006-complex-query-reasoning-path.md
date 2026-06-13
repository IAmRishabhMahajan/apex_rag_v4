# US-006: Complex Query Reasoning Path

## User Story

As a user asking a complex question, I want APEX-RAG v5 to decompose my request, reason over related claims, and compress noisy context so that the final answer covers the important parts without unnecessary retrieval noise.

## Scope

Add the optional complex-query path for investigations, multi-hop questions, research questions, and comparative analysis.

## Implementation Tasks

- Implement a complexity assessment gate that decides whether to activate the complex path.
- Define a `Claim` model with text, dependencies, evidence links, status, and confidence.
- Decompose complex questions into retrievable claims.
- Build a claim graph with support, cause, dependency, and contradiction relationships.
- Compress large retrieved contexts into high-signal summaries tied to source evidence.
- Keep the simple path available for queries that do not need decomposition.

## Acceptance Criteria

- Simple queries bypass the complex path.
- Complex queries produce multiple focused claims.
- Claim graph edges include relationship type and rationale.
- Compressed context retains source links.
- The final claim set covers the original user question.

## Testing Expectations

- Unit tests cover complexity gate decisions.
- Tests verify claim decomposition for investigation and comparison examples.
- Tests verify context compression preserves citation links.

## Documentation Updates

- Document complexity gate criteria.
- Document claim graph relationship types.

## Dependencies

- US-002 Adaptive Retrieval Planning.
- US-004 Evidence Fusion.
- US-005 Validation Mesh.

