"""Executable checks for US-005 Validation Mesh."""

from __future__ import annotations

import unittest

from tests.story_checks import StoryExpectation, assert_story_contract


class TestUS005ValidationMesh(unittest.TestCase):
    """Keep validation gates present across the pipeline stages."""

    def test_story_contract(self) -> None:
        expectation = StoryExpectation(
            story_id="US-005",
            slug="validation-mesh",
            title="Validation Mesh",
            required_terms=(
                "ValidationResult",
                "approve",
                "reject",
                "repair",
                "escalate",
                "generation validation",
            ),
            dependency_terms=("US-001 Query Intelligence", "US-004 Evidence Fusion"),
        )

        assert_story_contract(self, expectation)


if __name__ == "__main__":
    unittest.main()
