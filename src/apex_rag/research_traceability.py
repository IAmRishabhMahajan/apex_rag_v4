"""US-012 Research Foundation Traceability — registry linking components to research papers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ImplementationStatus(str, Enum):
    """Implementation state of a component relative to its research foundation."""

    IMPLEMENTED = "implemented"
    DEFERRED = "deferred"
    SIMPLIFIED = "simplified"


@dataclass(frozen=True)
class PaperReference:
    """A research paper that underpins one or more APEX-RAG components."""

    key: str
    title: str
    url: str
    contribution_summary: str

    def __post_init__(self) -> None:
        """Validate that key, URL, and contribution_summary are non-empty and well-formed."""
        if not self.key:
            raise ValueError("key must not be empty")
        if not self.url.startswith(("https://", "http://")):
            raise ValueError(f"url must start with https:// or http://, got: {self.url}")
        if not self.contribution_summary:
            raise ValueError("contribution_summary must not be empty")


@dataclass(frozen=True)
class ComponentMapping:
    """Maps a pipeline component to one or more research papers that motivated it."""

    component_name: str
    paper_keys: tuple[str, ...]
    status: ImplementationStatus
    notes: str

    def __post_init__(self) -> None:
        """Validate that component_name is non-empty and at least one paper key is provided."""
        if not self.component_name:
            raise ValueError("component_name must not be empty")
        if not self.paper_keys:
            raise ValueError("component_mapping must reference at least one paper")


@dataclass
class ResearchRegistry:
    """Holds all registered papers and component mappings for the project."""

    papers: dict[str, PaperReference] = field(default_factory=dict)
    mappings: list[ComponentMapping] = field(default_factory=list)

    def add_paper(self, paper: PaperReference) -> None:
        """Register a paper by its key; overwrites any existing entry with the same key."""
        self.papers[paper.key] = paper

    def add_mapping(self, mapping: ComponentMapping) -> None:
        """Add a component mapping after verifying all referenced paper keys exist."""
        unknown = [k for k in mapping.paper_keys if k not in self.papers]
        if unknown:
            raise ValueError(f"Mapping references unknown paper keys: {unknown}")
        self.mappings.append(mapping)

    def papers_for_component(self, component_name: str) -> list[PaperReference]:
        """Return all papers linked to the named component, or an empty list if unknown."""
        for mapping in self.mappings:
            if mapping.component_name == component_name:
                return [self.papers[k] for k in mapping.paper_keys if k in self.papers]
        return []

    def components_for_paper(self, paper_key: str) -> list[ComponentMapping]:
        """Return all component mappings that reference the given paper key."""
        return [m for m in self.mappings if paper_key in m.paper_keys]

    def deferred_ideas(self) -> list[ComponentMapping]:
        """Return all mappings whose status is DEFERRED."""
        return [m for m in self.mappings if m.status == ImplementationStatus.DEFERRED]

    def validate(self) -> list[str]:
        """Return a list of validation errors; empty list means registry is valid."""
        errors: list[str] = []
        for mapping in self.mappings:
            for key in mapping.paper_keys:
                if key not in self.papers:
                    errors.append(
                        f"Component '{mapping.component_name}' references unknown paper '{key}'"
                    )
        return errors


# ---------------------------------------------------------------------------
# Built-in registry
# ---------------------------------------------------------------------------

_PAPERS = [
    PaperReference(
        key="CRAG",
        title="CRAG: Comprehensive RAG",
        url="https://arxiv.org/abs/2401.15884",
        contribution_summary=(
            "Introduces corrective RAG that evaluates retrieved documents and triggers "
            "web-search fallback for low-confidence evidence — foundation for the "
            "retrieval repair loop (US-008)."
        ),
    ),
    PaperReference(
        key="Self-RAG",
        title="Self-RAG: Learning to Retrieve, Generate, and Critique",
        url="https://arxiv.org/abs/2310.11511",
        contribution_summary=(
            "Self-reflective tokens for retrieval necessity and critique — informs "
            "evidence scoring (US-007) and validation mesh (US-005)."
        ),
    ),
    PaperReference(
        key="GraphRAG",
        title="From Local to Global: A Graph RAG Approach",
        url="https://arxiv.org/abs/2404.16130",
        contribution_summary=(
            "Community-based graph retrieval over knowledge graphs — inspires expert "
            "routing (US-003) GRAPH expert and complex reasoning graph structure (US-006)."
        ),
    ),
    PaperReference(
        key="FLARE",
        title="Active Retrieval Augmented Generation",
        url="https://arxiv.org/abs/2305.06983",
        contribution_summary=(
            "Forward-looking active retrieval triggered when generation confidence is low — "
            "informs adaptive retrieval planning (US-002) and repair loop trigger conditions."
        ),
    ),
    PaperReference(
        key="RAGTruth",
        title="RAGTruth: A Hallucination Corpus for Developing Trustworthy RAG",
        url="https://arxiv.org/abs/2401.00396",
        contribution_summary=(
            "Hallucination taxonomy and detection for RAG outputs — informs grounded "
            "generation (US-009) claim-blocking and risk verification (US-010)."
        ),
    ),
    PaperReference(
        key="LongRAG",
        title="LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs",
        url="https://arxiv.org/abs/2406.15319",
        contribution_summary=(
            "Long-context evidence aggregation — informs evidence fusion (US-004) and "
            "context compression in complex reasoning (US-006)."
        ),
    ),
    PaperReference(
        key="RECOMP",
        title="RECOMP: Improving Retrieval-Augmented LMs with Context Compression",
        url="https://arxiv.org/abs/2310.04408",
        contribution_summary=(
            "Abstractive and extractive context compressors — directly informs "
            "compress_context() in the complex reasoning path (US-006)."
        ),
    ),
    PaperReference(
        key="DSPy",
        title="DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines",
        url="https://arxiv.org/abs/2310.03714",
        contribution_summary=(
            "Declarative pipeline optimisation — inspires the modular pipeline design "
            "across all implementation stories."
        ),
    ),
    PaperReference(
        key="RAGAS",
        title="RAGAS: Automated Evaluation of Retrieval Augmented Generation",
        url="https://arxiv.org/abs/2309.15217",
        contribution_summary=(
            "Faithfulness, answer relevance, and context precision metrics — directly "
            "maps to FinalMetrics in APEX-Eval (US-011)."
        ),
    ),
    PaperReference(
        key="ARES",
        title="ARES: An Automated Evaluation Framework for RAG",
        url="https://arxiv.org/abs/2311.09476",
        contribution_summary=(
            "LLM judge-based evaluation with in-domain fine-tuning — informs evaluation "
            "framework design (US-011) and regression testing approach."
        ),
    ),
    PaperReference(
        key="BEIR",
        title="BEIR: A Heterogeneous Benchmark for Zero-Shot Evaluation",
        url="https://arxiv.org/abs/2104.08663",
        contribution_summary=(
            "Zero-shot retrieval benchmark covering diverse domains — informs "
            "Recall@K, Precision@K, NDCG metrics in APEX-Eval (US-011)."
        ),
    ),
    PaperReference(
        key="KILT",
        title="KILT: A Benchmark for Knowledge-Intensive Language Tasks",
        url="https://arxiv.org/abs/2009.02252",
        contribution_summary=(
            "Knowledge-intensive task benchmark linking answers to Wikipedia provenance — "
            "informs research traceability (US-012) and answer grounding (US-009)."
        ),
    ),
    PaperReference(
        key="RAGBench",
        title="RAGBench: Explainable Benchmark for Retrieval-Augmented Generation",
        url="https://arxiv.org/abs/2407.11005",
        contribution_summary=(
            "Fine-grained faithfulness and relevance benchmarking — informs "
            "evidence scoring (US-007) and APEX-Eval metrics (US-011)."
        ),
    ),
]

_COMPONENT_MAPPINGS = [
    ComponentMapping(
        component_name="query_intelligence",
        paper_keys=("DSPy", "FLARE"),
        status=ImplementationStatus.IMPLEMENTED,
        notes="Intent detection and query expansion implemented via regex heuristics (no LLM).",
    ),
    ComponentMapping(
        component_name="retrieval_planning",
        paper_keys=("FLARE", "Self-RAG"),
        status=ImplementationStatus.IMPLEMENTED,
        notes="Adaptive plan selection based on intent and query content.",
    ),
    ComponentMapping(
        component_name="expert_routing",
        paper_keys=("GraphRAG", "CRAG"),
        status=ImplementationStatus.IMPLEMENTED,
        notes="Multi-expert routing with fallback chains.",
    ),
    ComponentMapping(
        component_name="evidence_fusion",
        paper_keys=("LongRAG", "RAGTruth"),
        status=ImplementationStatus.IMPLEMENTED,
        notes="SHA-256 deduplication, negation-based conflict detection.",
    ),
    ComponentMapping(
        component_name="validation_mesh",
        paper_keys=("Self-RAG", "RAGTruth"),
        status=ImplementationStatus.IMPLEMENTED,
        notes="Pipeline-stage validators with approve/reject/repair/escalate outcomes.",
    ),
    ComponentMapping(
        component_name="complex_reasoning",
        paper_keys=("GraphRAG", "RECOMP", "LongRAG"),
        status=ImplementationStatus.IMPLEMENTED,
        notes="Claim decomposition, graph edges, context compression.",
    ),
    ComponentMapping(
        component_name="evidence_scoring",
        paper_keys=("Self-RAG", "RAGAS", "RAGBench"),
        status=ImplementationStatus.IMPLEMENTED,
        notes="Composite authority/freshness/agreement/completeness scoring.",
    ),
    ComponentMapping(
        component_name="retrieval_repair",
        paper_keys=("CRAG", "FLARE"),
        status=ImplementationStatus.IMPLEMENTED,
        notes="Bounded repair loop with six failure classes and recovery strategies.",
    ),
    ComponentMapping(
        component_name="generation",
        paper_keys=("RAGTruth", "RAGAS"),
        status=ImplementationStatus.IMPLEMENTED,
        notes="Claim-blocking grounded generation; fails closed without evidence.",
    ),
    ComponentMapping(
        component_name="risk_verification",
        paper_keys=("RAGTruth", "Self-RAG"),
        status=ImplementationStatus.IMPLEMENTED,
        notes="Risk category classification, hedge removal, sentence-level critique.",
    ),
    ComponentMapping(
        component_name="apex_eval",
        paper_keys=("RAGAS", "ARES", "BEIR", "RAGBench"),
        status=ImplementationStatus.IMPLEMENTED,
        notes="Recall@K, Precision@K, MRR, NDCG, faithfulness, groundedness metrics.",
    ),
    ComponentMapping(
        component_name="llm_judge_evaluation",
        paper_keys=("ARES", "RAGAS"),
        status=ImplementationStatus.DEFERRED,
        notes="LLM-as-judge scoring deferred; current eval is fully rule-based.",
    ),
    ComponentMapping(
        component_name="wikipedia_provenance_grounding",
        paper_keys=("KILT",),
        status=ImplementationStatus.DEFERRED,
        notes="KILT-style wiki-provenance links deferred; citation metadata used instead.",
    ),
    ComponentMapping(
        component_name="dspy_pipeline_optimisation",
        paper_keys=("DSPy",),
        status=ImplementationStatus.DEFERRED,
        notes="DSPy teleprompter optimisation of prompts deferred; no LLM in current impl.",
    ),
]


def build_default_registry() -> ResearchRegistry:
    """Return the pre-populated registry for APEX-RAG v5."""
    registry = ResearchRegistry()
    for paper in _PAPERS:
        registry.add_paper(paper)
    for mapping in _COMPONENT_MAPPINGS:
        registry.add_mapping(mapping)
    return registry
