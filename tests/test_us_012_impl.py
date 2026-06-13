"""Real unit tests for US-012 Research Foundation Traceability implementation."""

from __future__ import annotations

import unittest

from src.apex_rag.research_traceability import (
    ComponentMapping,
    ImplementationStatus,
    PaperReference,
    ResearchRegistry,
    build_default_registry,
)


def _paper(key: str = "TESTPAPER") -> PaperReference:
    return PaperReference(
        key=key,
        title=f"Test paper {key}",
        url="https://arxiv.org/abs/0000.00000",
        contribution_summary="A test contribution.",
    )


def _mapping(
    component: str = "test_component",
    paper_keys: tuple[str, ...] = ("TESTPAPER",),
    status: ImplementationStatus = ImplementationStatus.IMPLEMENTED,
) -> ComponentMapping:
    return ComponentMapping(
        component_name=component,
        paper_keys=paper_keys,
        status=status,
        notes="Test mapping.",
    )


class TestPaperReference(unittest.TestCase):
    def test_valid_paper_constructs_correctly(self) -> None:
        paper = _paper("RAGAS")
        self.assertEqual(paper.key, "RAGAS")
        self.assertTrue(paper.url.startswith("https://"))

    def test_empty_key_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            PaperReference(
                key="",
                title="Test",
                url="https://arxiv.org",
                contribution_summary="Something.",
            )

    def test_invalid_url_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            PaperReference(
                key="TESTPAPER",
                title="Test",
                url="arxiv.org/abs/0000",
                contribution_summary="Something.",
            )

    def test_empty_contribution_summary_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            PaperReference(
                key="TESTPAPER",
                title="Test",
                url="https://arxiv.org",
                contribution_summary="",
            )


class TestComponentMapping(unittest.TestCase):
    def test_valid_mapping_constructs_correctly(self) -> None:
        m = _mapping("query_intelligence", ("CRAG",))
        self.assertEqual(m.component_name, "query_intelligence")
        self.assertIn("CRAG", m.paper_keys)

    def test_empty_component_name_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            ComponentMapping(
                component_name="",
                paper_keys=("CRAG",),
                status=ImplementationStatus.IMPLEMENTED,
                notes="",
            )

    def test_empty_paper_keys_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            ComponentMapping(
                component_name="some_component",
                paper_keys=(),
                status=ImplementationStatus.IMPLEMENTED,
                notes="",
            )


class TestResearchRegistry(unittest.TestCase):
    def test_add_paper_and_retrieve(self) -> None:
        registry = ResearchRegistry()
        registry.add_paper(_paper("CRAG"))
        self.assertIn("CRAG", registry.papers)

    def test_add_mapping_with_valid_paper_succeeds(self) -> None:
        registry = ResearchRegistry()
        registry.add_paper(_paper("TESTPAPER"))
        registry.add_mapping(_mapping("component_a", ("TESTPAPER",)))
        self.assertEqual(len(registry.mappings), 1)

    def test_add_mapping_with_unknown_paper_raises(self) -> None:
        registry = ResearchRegistry()
        with self.assertRaises(ValueError):
            registry.add_mapping(_mapping("component_a", ("UNKNOWN_KEY",)))

    def test_papers_for_component_returns_correct_papers(self) -> None:
        registry = ResearchRegistry()
        registry.add_paper(_paper("CRAG"))
        registry.add_mapping(_mapping("retrieval_repair", ("CRAG",)))
        papers = registry.papers_for_component("retrieval_repair")
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].key, "CRAG")

    def test_papers_for_unknown_component_returns_empty(self) -> None:
        registry = ResearchRegistry()
        result = registry.papers_for_component("nonexistent_component")
        self.assertEqual(result, [])

    def test_components_for_paper_returns_correct_mappings(self) -> None:
        registry = ResearchRegistry()
        registry.add_paper(_paper("CRAG"))
        registry.add_mapping(_mapping("retrieval_repair", ("CRAG",)))
        registry.add_mapping(_mapping("expert_routing", ("CRAG",)))
        components = registry.components_for_paper("CRAG")
        names = [c.component_name for c in components]
        self.assertIn("retrieval_repair", names)
        self.assertIn("expert_routing", names)

    def test_deferred_ideas_returns_only_deferred(self) -> None:
        registry = ResearchRegistry()
        registry.add_paper(_paper("CRAG"))
        registry.add_mapping(
            _mapping("implemented_comp", ("CRAG",), ImplementationStatus.IMPLEMENTED)
        )
        registry.add_mapping(_mapping("deferred_comp", ("CRAG",), ImplementationStatus.DEFERRED))
        deferred = registry.deferred_ideas()
        names = [m.component_name for m in deferred]
        self.assertIn("deferred_comp", names)
        self.assertNotIn("implemented_comp", names)

    def test_validate_returns_empty_for_valid_registry(self) -> None:
        registry = ResearchRegistry()
        registry.add_paper(_paper("CRAG"))
        registry.add_mapping(_mapping("repair", ("CRAG",)))
        self.assertEqual(registry.validate(), [])


class TestDefaultRegistry(unittest.TestCase):
    def test_all_known_papers_present(self) -> None:
        registry = build_default_registry()
        expected_keys = [
            "CRAG",
            "Self-RAG",
            "GraphRAG",
            "FLARE",
            "RAGTruth",
            "LongRAG",
            "RECOMP",
            "DSPy",
            "RAGAS",
            "ARES",
            "BEIR",
            "KILT",
            "RAGBench",
        ]
        for key in expected_keys:
            self.assertIn(key, registry.papers, f"Missing paper key: {key}")

    def test_all_paper_urls_are_well_formed(self) -> None:
        registry = build_default_registry()
        for paper in registry.papers.values():
            self.assertTrue(
                paper.url.startswith(("https://", "http://")),
                f"Bad URL for paper {paper.key}: {paper.url}",
            )

    def test_all_papers_have_contribution_summary(self) -> None:
        registry = build_default_registry()
        for paper in registry.papers.values():
            self.assertTrue(
                len(paper.contribution_summary) > 0,
                f"Missing contribution summary for {paper.key}",
            )

    def test_no_mappings_reference_unknown_papers(self) -> None:
        registry = build_default_registry()
        errors = registry.validate()
        self.assertEqual(errors, [], f"Validation errors: {errors}")

    def test_apex_eval_component_has_mappings(self) -> None:
        registry = build_default_registry()
        papers = registry.papers_for_component("apex_eval")
        self.assertTrue(len(papers) > 0)

    def test_deferred_ideas_are_tracked(self) -> None:
        registry = build_default_registry()
        deferred = registry.deferred_ideas()
        self.assertTrue(len(deferred) > 0)

    def test_each_component_maps_to_real_papers(self) -> None:
        registry = build_default_registry()
        for mapping in registry.mappings:
            for key in mapping.paper_keys:
                self.assertIn(
                    key,
                    registry.papers,
                    f"Component '{mapping.component_name}' references unknown paper '{key}'",
                )

    def test_retrieval_repair_maps_to_crag(self) -> None:
        registry = build_default_registry()
        papers = registry.papers_for_component("retrieval_repair")
        keys = [p.key for p in papers]
        self.assertIn("CRAG", keys)


if __name__ == "__main__":
    unittest.main()
