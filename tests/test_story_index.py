"""Checks that the user-story index remains aligned with the story files."""

from __future__ import annotations

import unittest

from tests.story_checks import STORIES_DIR


class TestStoryIndex(unittest.TestCase):
    """Protect the roadmap index because it is the entry point for planning."""

    def test_index_links_all_story_files(self) -> None:
        """The README.md index must link to all US-*.md story files by filename."""
        index_text = (STORIES_DIR / "README.md").read_text(encoding="utf-8")
        story_files = sorted(STORIES_DIR.glob("US-*.md"))

        self.assertEqual(23, len(story_files))
        for story_file in story_files:
            self.assertIn(f"]({story_file.name})", index_text)


if __name__ == "__main__":
    unittest.main()
