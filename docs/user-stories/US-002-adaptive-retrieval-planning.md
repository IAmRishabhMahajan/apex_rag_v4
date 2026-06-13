# US-002: Adaptive Retrieval Planning

## User Story

As a user, I want APEX-RAG v5 to choose a retrieval strategy that matches my query so that simple questions stay fast and complex questions receive deeper retrieval.

## Scope

Create a retrieval planner that maps a validated query profile to one or more retrieval strategies.

## Implementation Tasks

- Define a `RetrievalPlan` model with strategy, selected experts, required evidence types, freshness needs, and expected coverage.
- Support standard, multi-hop, graph, analytics, freshness, and web retrieval plans.
- Add planner rules that use intent, constraints, entity types, and risk signals.
- Include explainable planning reasons for observability and debugging.
- Add a fallback standard retrieval plan when no specialized plan is justified.

## Acceptance Criteria

- Simple lookups produce a standard retrieval plan.
- Multi-hop or investigation queries produce a multi-step plan.
- Freshness-sensitive queries request freshness or web retrieval.
- Analytics queries request analytics retrieval instead of document-only retrieval.
- The plan includes a plain-language reason for each selected strategy.

## Testing Expectations

- Unit tests verify planner output for representative query profiles.
- Tests cover fallback behavior when planner confidence is low.
- Tests verify freshness constraints affect the plan.

## Documentation Updates

- Document retrieval strategy types and planner decision rules.

## Dependencies

- US-001 Query Intelligence.

