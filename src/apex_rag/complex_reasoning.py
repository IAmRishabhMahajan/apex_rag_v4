"""US-006 Complex Query Reasoning Path — decompose, graph, and compress for hard queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.apex_rag.evidence_fusion import EvidenceBundle, EvidenceItem
from src.apex_rag.query_intelligence import Intent, QueryProfile


class ClaimStatus(str, Enum):
    """Lifecycle state of a claim as evidence is linked to it."""

    PENDING = "pending"
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class EdgeRelationship(str, Enum):
    """Semantic relationship between two claims in the claim graph."""

    SUPPORT = "support"
    CAUSE = "cause"
    DEPENDENCY = "dependency"
    CONTRADICTION = "contradiction"


@dataclass
class Claim:
    """A single retrievable sub-question decomposed from the original complex query."""

    text: str
    status: ClaimStatus = ClaimStatus.PENDING
    dependencies: list[str] = field(default_factory=list)
    evidence_links: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def link_evidence(self, source_id: str) -> None:
        """Append a source ID to this claim's evidence links if not already present."""
        if source_id not in self.evidence_links:
            self.evidence_links.append(source_id)


@dataclass(frozen=True)
class ClaimEdge:
    """A directed edge between two claims with a typed relationship and rationale."""

    from_claim: str
    to_claim: str
    relationship: EdgeRelationship
    rationale: str


@dataclass
class ClaimGraph:
    """Directed graph of claims and typed edges for a complex query."""

    claims: list[Claim]
    edges: list[ClaimEdge]

    def add_edge(
        self,
        from_text: str,
        to_text: str,
        relationship: EdgeRelationship,
        rationale: str,
    ) -> None:
        """Append a new typed edge to the graph."""
        self.edges.append(
            ClaimEdge(
                from_claim=from_text,
                to_claim=to_text,
                relationship=relationship,
                rationale=rationale,
            )
        )

    def covers_query(self, raw_query: str) -> bool:
        """True when at least one claim shares significant words with the original query."""
        query_words = {w.lower() for w in raw_query.split() if len(w) > 4}
        for claim in self.claims:
            claim_words = {w.lower() for w in claim.text.split() if len(w) > 4}
            if query_words & claim_words:
                return True
        return False


@dataclass(frozen=True)
class CompressedContext:
    """A prioritised summary of the most claim-relevant evidence with source citations."""

    summary: str
    source_links: tuple[str, ...]
    retained_item_count: int


# ---------------------------------------------------------------------------
# Complexity gate
# ---------------------------------------------------------------------------

_COMPLEX_INTENTS = frozenset({Intent.INVESTIGATION, Intent.ANALYSIS, Intent.COMPARISON})
_COMPLEXITY_KEYWORDS = frozenset(
    {
        "why",
        "how did",
        "root cause",
        "compare",
        "versus",
        "difference between",
        "multi-step",
        "relationship",
        "sequence",
        "timeline",
        "explain",
    }
)
_MIN_ENTITIES_FOR_COMPLEXITY = 2


def is_complex(profile: QueryProfile) -> bool:
    """Return True when the query warrants the complex reasoning path."""
    if profile.intent in _COMPLEX_INTENTS:
        return True
    query_lower = profile.raw_query.lower()
    if any(kw in query_lower for kw in _COMPLEXITY_KEYWORDS):
        return True
    non_date_entities = [e for e in profile.entities if e.entity_type.value != "date"]
    return len(non_date_entities) >= _MIN_ENTITIES_FOR_COMPLEXITY


# ---------------------------------------------------------------------------
# Claim decomposition
# ---------------------------------------------------------------------------

_DECOMPOSITION_TEMPLATES: dict[Intent, list[str]] = {
    Intent.INVESTIGATION: [
        "What was the state before the event described in: {query}",
        "What caused the change described in: {query}",
        "What were the consequences of: {query}",
    ],
    Intent.ANALYSIS: [
        "What are the components involved in: {query}",
        "How do the components interact in: {query}",
        "What are the outcomes or implications of: {query}",
    ],
    Intent.COMPARISON: [
        "What are the key properties of the first subject in: {query}",
        "What are the key properties of the second subject in: {query}",
        "What are the main differences and similarities in: {query}",
    ],
}

_DEFAULT_TEMPLATES = [
    "What is the main claim in: {query}",
    "What evidence supports or refutes: {query}",
    "What are the limitations or caveats of: {query}",
]


def decompose_query(profile: QueryProfile) -> list[Claim]:
    """Break a complex query into focused sub-claims."""
    templates = _DECOMPOSITION_TEMPLATES.get(profile.intent, _DEFAULT_TEMPLATES)
    return [
        Claim(text=t.format(query=profile.raw_query), status=ClaimStatus.PENDING) for t in templates
    ]


# ---------------------------------------------------------------------------
# Claim graph construction
# ---------------------------------------------------------------------------


def build_claim_graph(claims: list[Claim], profile: QueryProfile) -> ClaimGraph:
    """Wire basic structural edges between decomposed claims."""
    graph = ClaimGraph(claims=claims, edges=[])

    if len(claims) >= 2:
        graph.add_edge(
            claims[0].text,
            claims[1].text,
            EdgeRelationship.DEPENDENCY,
            "Second claim depends on the context established by the first.",
        )

    if len(claims) >= 3:
        graph.add_edge(
            claims[1].text,
            claims[2].text,
            EdgeRelationship.CAUSE,
            "Third claim explores the causal or comparative outcome of the second.",
        )

    if profile.intent == Intent.COMPARISON and len(claims) >= 3:
        graph.add_edge(
            claims[0].text,
            claims[1].text,
            EdgeRelationship.CONTRADICTION,
            "Comparison subjects may hold contradictory properties.",
        )

    return graph


# ---------------------------------------------------------------------------
# Evidence linkage and context compression
# ---------------------------------------------------------------------------

_MAX_SUMMARY_ITEMS = 5
_MIN_WORD_OVERLAP = 2


def link_evidence_to_claims(graph: ClaimGraph, bundle: EvidenceBundle) -> ClaimGraph:
    """Match evidence items to claims by word overlap and update claim status."""
    for claim in graph.claims:
        claim_words = {w.lower() for w in claim.text.split() if len(w) > 4}
        for item in bundle.items:
            item_words = {w.lower() for w in item.content.split() if len(w) > 4}
            if len(claim_words & item_words) >= _MIN_WORD_OVERLAP:
                claim.link_evidence(item.citation.source_id)
                claim.status = ClaimStatus.SUPPORTED
                claim.confidence = min(1.0, claim.confidence + 0.3)

        if claim.status == ClaimStatus.PENDING:
            claim.status = ClaimStatus.UNSUPPORTED

    return graph


def compress_context(
    bundle: EvidenceBundle,
    graph: ClaimGraph,
) -> CompressedContext:
    """Summarise the most claim-relevant evidence, preserving source links."""
    linked_ids: set[str] = set()
    for claim in graph.claims:
        linked_ids.update(claim.evidence_links)

    prioritised: list[EvidenceItem] = [
        item for item in bundle.items if item.citation.source_id in linked_ids
    ]
    remaining = [item for item in bundle.items if item.citation.source_id not in linked_ids]
    selected = (prioritised + remaining)[:_MAX_SUMMARY_ITEMS]

    summary_parts = [item.content[:120] for item in selected]
    summary = " | ".join(summary_parts)
    source_links = tuple(item.citation.source_id for item in selected)

    return CompressedContext(
        summary=summary,
        source_links=source_links,
        retained_item_count=len(selected),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComplexReasoningResult:
    """Output of run_complex_reasoning(): flags whether the complex path was used."""

    used_complex_path: bool
    graph: ClaimGraph | None
    compressed_context: CompressedContext | None


def run_complex_reasoning(
    profile: QueryProfile,
    bundle: EvidenceBundle,
) -> ComplexReasoningResult:
    """Run the full complex reasoning path if the query warrants it."""
    if not is_complex(profile):
        return ComplexReasoningResult(
            used_complex_path=False,
            graph=None,
            compressed_context=None,
        )

    claims = decompose_query(profile)
    graph = build_claim_graph(claims, profile)
    graph = link_evidence_to_claims(graph, bundle)
    compressed = compress_context(bundle, graph)

    return ComplexReasoningResult(
        used_complex_path=True,
        graph=graph,
        compressed_context=compressed,
    )
