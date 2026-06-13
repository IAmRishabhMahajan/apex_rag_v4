# US-004: Evidence Fusion

## User Story

As a user, I want retrieved chunks converted into structured evidence so that every later claim can be traced back to source material.

## Scope

Normalize retrieved content into evidence records with source mapping, deduplication, conflict signals, and citation metadata.

## Implementation Tasks

- Define `EvidenceItem` and `EvidenceBundle` models.
- Normalize chunks from all retrieval experts into a common evidence format.
- Deduplicate overlapping or repeated evidence.
- Detect direct conflicts between evidence items.
- Preserve source mapping, citation metadata, retrieval expert, and retrieval query.
- Link evidence items to candidate claims when possible.

## Acceptance Criteria

- Equivalent chunks from different retrieval runs are deduplicated.
- Conflicting evidence is flagged instead of silently merged.
- Every evidence item includes source and retrieval provenance.
- Evidence can be grouped by candidate claim.
- Invalid evidence records produce explicit validation errors.

## Testing Expectations

- Unit tests cover normalization from each expert output shape.
- Tests cover deduplication and conflict detection.
- Tests verify provenance fields are never dropped.

## Documentation Updates

- Document evidence schemas and provenance requirements.

## Dependencies

- US-003 Expert Retrieval Routing.

