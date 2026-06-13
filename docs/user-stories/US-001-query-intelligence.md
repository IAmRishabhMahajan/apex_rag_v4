# US-001: Query Intelligence

## User Story

As a user asking APEX-RAG v5 a question, I want the system to understand my intent, entities, constraints, and retrieval needs before searching so that retrieval starts with the right interpretation of my request.

## Scope

Build the first pipeline stage that turns a raw user query into a structured query profile.

## Implementation Tasks

- Define a `QueryProfile` model with fields for raw query, intent, entities, constraints, risk signals, and query expansions.
- Implement intent detection for fact lookup, investigation, analysis, comparison, summarization, and forecasting.
- Implement entity extraction for people, companies, products, technologies, locations, and dates.
- Implement constraint extraction for time ranges, jurisdictions, regions, departments, and categories.
- Generate retrieval-friendly query expansions while preserving the original user intent.
- Return explicit validation errors when the query is too ambiguous to route safely.

## Acceptance Criteria

- Given a simple fact lookup, the profile identifies the intent and key entity.
- Given a comparison query, the profile identifies compared entities and comparison dimensions.
- Given date or jurisdiction constraints, those constraints are preserved in structured output.
- Query expansions do not invent entities or constraints absent from the user query.
- Ambiguous queries include useful clarification or fallback messages.

## Testing Expectations

- Unit tests cover each supported intent class.
- Unit tests cover entity and constraint extraction edge cases.
- Regression tests verify query expansions remain grounded in the original query.

## Documentation Updates

- Document the `QueryProfile` schema.
- Document supported intent labels and constraint types.

## Dependencies

- None. This is the first pipeline capability.

