"""Shared assertions for user-story documentation tests.

The project currently starts from planning artifacts, so these helpers make the
story files executable specifications until production modules exist.
"""

from __future__ import annotations

import re
import unittest
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORIES_DIR = ROOT / "docs" / "user-stories"
REQUIRED_SECTIONS = (
    "User Story",
    "Scope",
    "Implementation Tasks",
    "Acceptance Criteria",
    "Testing Expectations",
    "Documentation Updates",
    "Dependencies",
)


@dataclass(frozen=True)
class StoryExpectation:
    """Describes the minimum contract a story document must satisfy."""

    story_id: str
    slug: str
    title: str
    required_terms: tuple[str, ...]
    dependency_terms: tuple[str, ...] = ()

    @property
    def path(self) -> Path:
        """Resolve from the repository root so tests stay stable across shells."""

        return STORIES_DIR / f"{self.story_id}-{self.slug}.md"


def read_story(expectation: StoryExpectation) -> str:
    """Load a story file with a helpful failure when the roadmap drifts."""

    try:
        return expectation.path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AssertionError(f"Missing user story: {expectation.path}") from exc


def assert_story_contract(test_case: unittest.TestCase, expectation: StoryExpectation) -> None:
    """Verify a story remains detailed enough to drive implementation."""

    text = read_story(expectation)
    test_case.assertIn(f"# {expectation.story_id}: {expectation.title}", text)

    for section in REQUIRED_SECTIONS:
        test_case.assertRegex(text, rf"(?m)^## {re.escape(section)}$")

    for term in expectation.required_terms:
        test_case.assertIn(term, text)

    for dependency in expectation.dependency_terms:
        test_case.assertIn(dependency, text)

    acceptance = _section_body(text, "Acceptance Criteria")
    implementation = _section_body(text, "Implementation Tasks")
    testing = _section_body(text, "Testing Expectations")

    test_case.assertGreaterEqual(_bullet_count(implementation), 5)
    test_case.assertGreaterEqual(_bullet_count(acceptance), 5)
    test_case.assertGreaterEqual(_bullet_count(testing), 3)


def _section_body(text: str, section: str) -> str:
    pattern = rf"(?ms)^## {re.escape(section)}\n\n(?P<body>.*?)(?=^## |\Z)"
    match = re.search(pattern, text)
    if match is None:
        raise AssertionError(f"Missing section body: {section}")
    return match.group("body")


def _bullet_count(text: str) -> int:
    return len(re.findall(r"(?m)^- ", text))
