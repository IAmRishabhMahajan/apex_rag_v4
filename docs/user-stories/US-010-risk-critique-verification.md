# US-010: Risk Assessment, Critique, and Verification

## User Story

As a user asking about medical, legal, financial, compliance, or similarly high-risk topics, I want extra critique and verification before delivery so that the system avoids unsafe or overconfident answers.

## Scope

Add risk classification, answer critique, and final verification safeguards.

## Implementation Tasks

- Implement a risk assessment gate that classifies normal and high-risk queries.
- Define high-risk categories such as medical, legal, financial, and compliance.
- Route high-risk answers through an answer critic.
- Check answer completeness, constraint satisfaction, confidence calibration, and contradiction handling.
- Verify each sentence against evidence before final delivery.
- Remove or revise unsupported content.
- Add high-risk disclaimers or escalation guidance where appropriate.

## Acceptance Criteria

- Normal-risk answers can use the direct final answer path.
- High-risk answers require critique and verification.
- Unsupported high-risk statements are removed before delivery.
- Contradictions are acknowledged instead of hidden.
- Final answers include calibrated confidence when evidence is limited.

## Testing Expectations

- Unit tests cover risk classification categories.
- Tests verify high-risk answers cannot bypass critique.
- Tests verify unsupported sentences are removed during verification.

## Documentation Updates

- Document risk categories, verification requirements, and escalation behavior.

## Dependencies

- US-005 Validation Mesh.
- US-009 Grounded Reasoning and Generation.

