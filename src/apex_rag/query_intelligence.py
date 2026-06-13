"""US-001 Query Intelligence — turns a raw query into a structured QueryProfile."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    FACT_LOOKUP = "fact_lookup"
    INVESTIGATION = "investigation"
    ANALYSIS = "analysis"
    COMPARISON = "comparison"
    SUMMARIZATION = "summarization"
    FORECASTING = "forecasting"
    UNKNOWN = "unknown"


class EntityType(str, Enum):
    PERSON = "person"
    COMPANY = "company"
    PRODUCT = "product"
    TECHNOLOGY = "technology"
    LOCATION = "location"
    DATE = "date"


class ConstraintType(str, Enum):
    TIME_RANGE = "time_range"
    JURISDICTION = "jurisdiction"
    REGION = "region"
    DEPARTMENT = "department"
    CATEGORY = "category"


@dataclass(frozen=True)
class Entity:
    text: str
    entity_type: EntityType


@dataclass(frozen=True)
class Constraint:
    text: str
    constraint_type: ConstraintType


@dataclass(frozen=True)
class QueryProfile:
    raw_query: str
    intent: Intent
    entities: tuple[Entity, ...]
    constraints: tuple[Constraint, ...]
    risk_signals: tuple[str, ...]
    query_expansions: tuple[str, ...]
    validation_errors: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return len(self.validation_errors) == 0


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

_COMPARISON_PATTERNS = re.compile(
    r"\b(compare|versus|vs\.?|difference between|contrast|which is better|pros and cons)\b",
    re.IGNORECASE,
)
_FORECASTING_PATTERNS = re.compile(
    r"\b(forecast|predict|will|future|outlook|projection|trend)\b",
    re.IGNORECASE,
)
_INVESTIGATION_PATTERNS = re.compile(
    r"\b(why|how did|investigate|root cause|explain|what happened|cause of)\b",
    re.IGNORECASE,
)
_SUMMARIZATION_PATTERNS = re.compile(
    r"\b(summarize|summary|overview|recap|brief|tldr)\b",
    re.IGNORECASE,
)
_ANALYSIS_PATTERNS = re.compile(
    r"\b(analyze|analyse|breakdown|deep.?dive|assess|evaluate|review)\b",
    re.IGNORECASE,
)
_FACT_PATTERNS = re.compile(
    r"\b(what is|who is|when was|where is|how many|how much|define|list)\b",
    re.IGNORECASE,
)


def detect_intent(query: str) -> Intent:
    if _COMPARISON_PATTERNS.search(query):
        return Intent.COMPARISON
    if _FORECASTING_PATTERNS.search(query):
        return Intent.FORECASTING
    if _INVESTIGATION_PATTERNS.search(query):
        return Intent.INVESTIGATION
    if _SUMMARIZATION_PATTERNS.search(query):
        return Intent.SUMMARIZATION
    if _ANALYSIS_PATTERNS.search(query):
        return Intent.ANALYSIS
    if _FACT_PATTERNS.search(query):
        return Intent.FACT_LOOKUP
    return Intent.UNKNOWN


# ---------------------------------------------------------------------------
# Entity extraction (keyword/pattern-based; LLM-free for now)
# ---------------------------------------------------------------------------

_DATE_PATTERN = re.compile(
    r"\b(\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|january|february|march|april|may|june|july|august|"
    r"september|october|november|december|Q[1-4]\s*\d{4}|last\s+year|this\s+year|"
    r"last\s+quarter|this\s+quarter)\b",
    re.IGNORECASE,
)

_TECH_KEYWORDS = frozenset(
    {
        "python",
        "java",
        "javascript",
        "typescript",
        "kubernetes",
        "docker",
        "aws",
        "gcp",
        "azure",
        "llm",
        "rag",
        "api",
        "sql",
        "nosql",
        "mongodb",
        "postgresql",
        "redis",
        "kafka",
        "spark",
        "tensorflow",
        "pytorch",
        "bert",
        "gpt",
        "transformer",
        "langchain",
        "openai",
        "anthropic",
        "claude",
    }
)


def extract_entities(query: str) -> tuple[Entity, ...]:
    entities: list[Entity] = []

    for match in _DATE_PATTERN.finditer(query):
        entities.append(Entity(text=match.group(), entity_type=EntityType.DATE))

    words = query.split()
    for word in words:
        clean = word.strip(".,;:?!\"'()").lower()
        if clean in _TECH_KEYWORDS:
            entities.append(
                Entity(text=word.strip(".,;:?!\"'()"), entity_type=EntityType.TECHNOLOGY)
            )

    # Capitalised words (not at sentence start) treated as named entities
    for i, word in enumerate(words):
        clean = word.strip(".,;:?!\"'()")
        if (
            i > 0
            and clean
            and clean[0].isupper()
            and clean.lower() not in _TECH_KEYWORDS
            and not _DATE_PATTERN.match(clean)
        ):
            entities.append(Entity(text=clean, entity_type=EntityType.PERSON))

    # Deduplicate preserving order
    seen: set[tuple[str, EntityType]] = set()
    unique: list[Entity] = []
    for e in entities:
        key = (e.text.lower(), e.entity_type)
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return tuple(unique)


# ---------------------------------------------------------------------------
# Constraint extraction
# ---------------------------------------------------------------------------

_TIME_RANGE_PATTERN = re.compile(
    r"\b(since|before|after|between|from\s+\d{4}|until|in\s+\d{4}|"
    r"last\s+\d+\s+(days?|weeks?|months?|years?)|past\s+\d+\s+(days?|weeks?|months?|years?))\b",
    re.IGNORECASE,
)
_JURISDICTION_PATTERN = re.compile(
    r"\b(under\s+\w+\s+law|gdpr|hipaa|sox|pci.?dss|regulation|compliance|"
    r"legal\s+jurisdiction|governed\s+by)\b",
    re.IGNORECASE,
)
_REGION_PATTERN = re.compile(
    r"\b(in\s+(the\s+)?(US|UK|EU|APAC|EMEA|north america|europe|asia|india|china|japan))\b",
    re.IGNORECASE,
)
_DEPARTMENT_PATTERN = re.compile(
    r"\b(finance|engineering|marketing|sales|hr|legal|compliance|operations|"
    r"product|design|data\s+science)\s+(team|department|org)?\b",
    re.IGNORECASE,
)
_CATEGORY_PATTERN = re.compile(
    r"\b(category|type|kind|class|segment|tier):\s*\w+",
    re.IGNORECASE,
)


def extract_constraints(query: str) -> tuple[Constraint, ...]:
    constraints: list[Constraint] = []

    for match in _TIME_RANGE_PATTERN.finditer(query):
        constraints.append(
            Constraint(text=match.group().strip(), constraint_type=ConstraintType.TIME_RANGE)
        )
    for match in _JURISDICTION_PATTERN.finditer(query):
        constraints.append(
            Constraint(text=match.group().strip(), constraint_type=ConstraintType.JURISDICTION)
        )
    for match in _REGION_PATTERN.finditer(query):
        constraints.append(
            Constraint(text=match.group().strip(), constraint_type=ConstraintType.REGION)
        )
    for match in _DEPARTMENT_PATTERN.finditer(query):
        constraints.append(
            Constraint(text=match.group().strip(), constraint_type=ConstraintType.DEPARTMENT)
        )
    for match in _CATEGORY_PATTERN.finditer(query):
        constraints.append(
            Constraint(text=match.group().strip(), constraint_type=ConstraintType.CATEGORY)
        )

    seen: set[tuple[str, ConstraintType]] = set()
    unique: list[Constraint] = []
    for c in constraints:
        key = (c.text.lower(), c.constraint_type)
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return tuple(unique)


# ---------------------------------------------------------------------------
# Risk signals
# ---------------------------------------------------------------------------

_RISK_PATTERNS = {
    "financial_advice": re.compile(
        r"\b(invest|stock|buy|sell|portfolio|returns?)\b", re.IGNORECASE
    ),
    "medical_advice": re.compile(
        r"\b(diagnos|treatment|medication|dose|symptom|disease)\b", re.IGNORECASE
    ),
    "legal_advice": re.compile(r"\b(lawsuit|liable|sue|legal\s+action|attorney)\b", re.IGNORECASE),
    "pii": re.compile(
        r"\b(ssn|social security|credit card|password|personal data)\b", re.IGNORECASE
    ),
}


def detect_risk_signals(query: str) -> tuple[str, ...]:
    return tuple(label for label, pattern in _RISK_PATTERNS.items() if pattern.search(query))


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

_EXPANSION_SYNONYMS: dict[str, list[str]] = {
    "compare": ["contrast", "differentiate", "evaluate differences"],
    "summarize": ["overview", "key points", "brief summary"],
    "analyze": ["examine", "break down", "assess"],
    "investigate": ["explore", "diagnose", "identify root cause"],
    "predict": ["forecast", "estimate future", "project"],
    "what is": ["definition of", "explain"],
    "how does": ["mechanism of", "process behind"],
}


def generate_query_expansions(query: str, intent: Intent) -> tuple[str, ...]:
    expansions: list[str] = []
    query_lower = query.lower()

    for trigger, synonyms in _EXPANSION_SYNONYMS.items():
        if trigger in query_lower:
            for syn in synonyms:
                variant = re.sub(re.escape(trigger), syn, query, flags=re.IGNORECASE, count=1)
                if variant.lower() != query_lower:
                    expansions.append(variant)

    # Intent-level expansions that don't duplicate what's already in expansions
    if intent == Intent.COMPARISON and not expansions:
        expansions.append(f"differences and similarities: {query}")
    elif intent == Intent.SUMMARIZATION and not expansions:
        expansions.append(f"key points from: {query}")
    elif intent == Intent.FORECASTING and not expansions:
        expansions.append(f"historical trends for: {query}")

    return tuple(dict.fromkeys(expansions))  # deduplicate, preserve order


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_MIN_QUERY_LENGTH = 3
_MAX_QUERY_LENGTH = 2000


def validate_query(query: str, entities: tuple[Entity, ...], intent: Intent) -> tuple[str, ...]:
    errors: list[str] = []
    stripped = query.strip()

    if len(stripped) < _MIN_QUERY_LENGTH:
        errors.append("Query is too short to route safely.")
    if len(stripped) > _MAX_QUERY_LENGTH:
        errors.append("Query exceeds maximum supported length.")
    if intent == Intent.UNKNOWN and not entities:
        errors.append(
            "Query intent could not be determined and no entities were found; "
            "please provide more context."
        )

    return tuple(errors)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_query_profile(raw_query: str) -> QueryProfile:
    """Convert a raw user query into a structured QueryProfile."""

    intent = detect_intent(raw_query)
    entities = extract_entities(raw_query)
    constraints = extract_constraints(raw_query)
    risk_signals = detect_risk_signals(raw_query)
    expansions = generate_query_expansions(raw_query, intent)
    errors = validate_query(raw_query, entities, intent)

    return QueryProfile(
        raw_query=raw_query,
        intent=intent,
        entities=entities,
        constraints=constraints,
        risk_signals=risk_signals,
        query_expansions=expansions,
        validation_errors=errors,
    )
