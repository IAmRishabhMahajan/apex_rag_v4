"""Real unit tests for US-001 Query Intelligence implementation."""

from __future__ import annotations

import unittest

from src.apex_rag.query_intelligence import (
    ConstraintType,
    EntityType,
    Intent,
    build_query_profile,
    detect_intent,
    detect_risk_signals,
    extract_constraints,
    extract_entities,
    generate_query_expansions,
)


class TestIntentDetection(unittest.TestCase):
    """Tests for detect_intent() — verifies that each Intent variant is correctly classified.

    US-001 requires 6 intent classes: fact_lookup, investigation, analysis,
    comparison, summarization, forecasting, plus unknown as a fallback.
    """

    def test_fact_lookup(self) -> None:
        """A 'what is' query should be classified as FACT_LOOKUP."""
        self.assertEqual(detect_intent("What is Kubernetes?"), Intent.FACT_LOOKUP)

    def test_comparison(self) -> None:
        """A 'versus' query should be classified as COMPARISON."""
        self.assertEqual(
            detect_intent("Compare Python versus Java for data pipelines"),
            Intent.COMPARISON,
        )

    def test_forecasting(self) -> None:
        """A query containing 'forecast' should be classified as FORECASTING."""
        self.assertEqual(
            detect_intent("What is the forecast for cloud spending in 2025?"),
            Intent.FORECASTING,
        )

    def test_investigation(self) -> None:
        """A 'why did' query should be classified as INVESTIGATION."""
        self.assertEqual(
            detect_intent("Why did the deployment fail last night?"),
            Intent.INVESTIGATION,
        )

    def test_summarization(self) -> None:
        """A 'summarize' query should be classified as SUMMARIZATION."""
        self.assertEqual(detect_intent("Summarize the Q3 earnings report"), Intent.SUMMARIZATION)

    def test_analysis(self) -> None:
        """An 'analyze' query should be classified as ANALYSIS."""
        self.assertEqual(
            detect_intent("Analyze the performance breakdown of the API"),
            Intent.ANALYSIS,
        )

    def test_unknown_returns_unknown(self) -> None:
        """A gibberish query with no intent signals should return UNKNOWN."""
        self.assertEqual(detect_intent("xyzzy bloop florp"), Intent.UNKNOWN)

    def test_comparison_takes_priority_over_fact(self) -> None:
        """COMPARISON pattern should win over FACT_LOOKUP when both could match."""
        self.assertEqual(
            detect_intent("What is the difference between REST and GraphQL?"),
            Intent.COMPARISON,
        )


class TestEntityExtraction(unittest.TestCase):
    """Tests for extract_entities() — verifies DATE, TECHNOLOGY, and named-entity detection.

    US-001 requires entity extraction; the implementation uses keyword lists and
    capitalisation heuristics rather than an LLM.
    """

    def test_technology_entity(self) -> None:
        """Known technology keywords should be extracted as TECHNOLOGY entities."""
        entities = extract_entities("How does Kubernetes handle scaling?")
        types = [e.entity_type for e in entities]
        self.assertIn(EntityType.TECHNOLOGY, types)

    def test_date_entity(self) -> None:
        """Quarter/year tokens should be extracted as DATE entities."""
        entities = extract_entities("What happened in Q3 2024?")
        types = [e.entity_type for e in entities]
        self.assertIn(EntityType.DATE, types)

    def test_no_invented_entities(self) -> None:
        """Lowercase common words should not be extracted as PERSON entities."""
        entities = extract_entities("how many widgets were sold?")
        person_entities = [e for e in entities if e.entity_type == EntityType.PERSON]
        self.assertEqual(person_entities, [])

    def test_multiple_technologies(self) -> None:
        """Multiple technology keywords in one query should all be extracted."""
        entities = extract_entities("Compare Python and Docker deployment")
        tech_texts = {e.text.lower() for e in entities if e.entity_type == EntityType.TECHNOLOGY}
        self.assertIn("python", tech_texts)
        self.assertIn("docker", tech_texts)

    def test_deduplication(self) -> None:
        """Repeated occurrences of the same technology should produce a single entity."""
        entities = extract_entities("Python Python Python")
        tech = [e for e in entities if e.entity_type == EntityType.TECHNOLOGY]
        self.assertEqual(len(tech), 1)


class TestConstraintExtraction(unittest.TestCase):
    """Tests for extract_constraints() — verifies time-range, jurisdiction, and region detection.

    US-001 requires extracting constraints that narrow retrieval scope.
    """

    def test_time_range_since(self) -> None:
        """The word 'since' followed by a year should produce a TIME_RANGE constraint."""
        constraints = extract_constraints("Show me results since 2022")
        types = [c.constraint_type for c in constraints]
        self.assertIn(ConstraintType.TIME_RANGE, types)

    def test_jurisdiction_gdpr(self) -> None:
        """The term 'GDPR' should be extracted as a JURISDICTION constraint."""
        constraints = extract_constraints("What data must be retained under GDPR?")
        types = [c.constraint_type for c in constraints]
        self.assertIn(ConstraintType.JURISDICTION, types)

    def test_region_eu(self) -> None:
        """The phrase 'in the EU' should be extracted as a REGION constraint."""
        constraints = extract_constraints("Sales performance in the EU last quarter")
        types = [c.constraint_type for c in constraints]
        self.assertIn(ConstraintType.REGION, types)

    def test_no_false_constraints(self) -> None:
        """A plain factual query with no scope signals should yield no constraints."""
        constraints = extract_constraints("What is the capital of France?")
        self.assertEqual(constraints, ())


class TestRiskSignals(unittest.TestCase):
    """Tests for detect_risk_signals() — verifies that financial, medical, legal, and PII signals are found.

    US-001 requires risk detection so that downstream modules can apply
    appropriate safeguards (US-010 risk verification).
    """

    def test_financial_advice(self) -> None:
        """Investment-related keywords should trigger the 'financial_advice' signal."""
        signals = detect_risk_signals("Should I invest in this stock?")
        self.assertIn("financial_advice", signals)

    def test_medical_advice(self) -> None:
        """Medical keywords should trigger the 'medical_advice' signal."""
        signals = detect_risk_signals("What is the correct medication dose?")
        self.assertIn("medical_advice", signals)

    def test_no_false_risk(self) -> None:
        """A plain science query should produce no risk signals."""
        signals = detect_risk_signals("What is the speed of light?")
        self.assertEqual(signals, ())

    def test_pii_detection(self) -> None:
        """References to social security numbers should trigger the 'pii' signal."""
        signals = detect_risk_signals("Find records by social security number")
        self.assertIn("pii", signals)


class TestQueryExpansions(unittest.TestCase):
    """Tests for generate_query_expansions() — verifies synonym-based query rewriting.

    US-001 requires query expansions to broaden retrieval coverage.
    Expansions must be grounded in the original query and not invent new entities.
    """

    def test_comparison_expansion(self) -> None:
        """A comparison query should produce at least one synonym expansion."""
        expansions = generate_query_expansions("Compare REST vs GraphQL", Intent.COMPARISON)
        self.assertTrue(len(expansions) > 0)

    def test_expansions_grounded_in_original(self) -> None:
        """Every expansion should share at least one word with the original query."""
        query = "Summarize the incident report"
        expansions = generate_query_expansions(query, Intent.SUMMARIZATION)
        for expansion in expansions:
            original_words = set(query.lower().split())
            expansion_words = set(expansion.lower().split())
            self.assertTrue(
                original_words & expansion_words,
                f"Expansion '{expansion}' shares no words with original query",
            )

    def test_no_invented_entities_in_expansions(self) -> None:
        """Expansions must not introduce entity names absent from the original query."""
        query = "What is Docker?"
        expansions = generate_query_expansions(query, Intent.FACT_LOOKUP)
        for exp in expansions:
            self.assertNotIn("Kubernetes", exp)
            self.assertNotIn("AWS", exp)


class TestBuildQueryProfile(unittest.TestCase):
    """Tests for build_query_profile() — the main US-001 public entry point.

    Verifies that all profile fields (intent, entities, constraints, risk signals,
    expansions, validation errors) are correctly populated end-to-end.
    """

    def test_simple_fact_lookup(self) -> None:
        """A fact-lookup query should produce a valid profile with a TECHNOLOGY entity."""
        profile = build_query_profile("What is Kubernetes?")
        self.assertEqual(profile.intent, Intent.FACT_LOOKUP)
        self.assertTrue(profile.is_valid)
        tech_entities = [e for e in profile.entities if e.entity_type == EntityType.TECHNOLOGY]
        self.assertTrue(len(tech_entities) > 0)

    def test_comparison_query_identifies_entities(self) -> None:
        """A comparison query should extract both technology entities mentioned."""
        profile = build_query_profile("Compare Python and Docker for CI/CD pipelines")
        self.assertEqual(profile.intent, Intent.COMPARISON)
        tech_texts = {
            e.text.lower() for e in profile.entities if e.entity_type == EntityType.TECHNOLOGY
        }
        self.assertIn("python", tech_texts)
        self.assertIn("docker", tech_texts)

    def test_date_constraint_preserved(self) -> None:
        """A query with 'since YYYY' should include a TIME_RANGE constraint in the profile."""
        profile = build_query_profile("Show incidents since 2023 in the EU")
        time_constraints = [
            c for c in profile.constraints if c.constraint_type == ConstraintType.TIME_RANGE
        ]
        self.assertTrue(len(time_constraints) > 0)

    def test_ambiguous_query_has_error(self) -> None:
        """A query shorter than the minimum length should produce validation errors."""
        profile = build_query_profile("aaa")
        self.assertFalse(profile.is_valid)
        self.assertTrue(len(profile.validation_errors) > 0)

    def test_gdpr_jurisdiction_preserved(self) -> None:
        """A GDPR query should include a JURISDICTION constraint in the profile."""
        profile = build_query_profile("What retention rules apply under GDPR compliance?")
        jurisdiction = [
            c for c in profile.constraints if c.constraint_type == ConstraintType.JURISDICTION
        ]
        self.assertTrue(len(jurisdiction) > 0)


if __name__ == "__main__":
    unittest.main()
