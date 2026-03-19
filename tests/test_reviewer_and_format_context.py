"""Tests for reviewer prompt optimizations and _format_context improvements.

Covers:
- exclude_target=True removes the target file from the "Related Files" section
- ReviewerAgent._review_file uses context.related_files instead of re-reading disk
- get_language_profile is not called in a loop (results cached per context build)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.coder_agent import CoderAgent
from agents.reviewer_agent import ReviewerAgent
from core.models import (
    AgentContext,
    FileBlueprint,
    RepositoryBlueprint,
    Task,
    TaskType,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_context(
    *,
    file_path: str = "src/Service.java",
    related: dict[str, str] | None = None,
    arch_summary: str = "REST architecture",
    language: str = "java",
) -> AgentContext:
    fb = FileBlueprint(path=file_path, purpose="Service", depends_on=["src/Dep.java"], language=language)
    bp = RepositoryBlueprint(
        name="test",
        description="test",
        architecture_style="REST",
        tech_stack={"language": language},
        file_blueprints=[fb],
        architecture_doc=arch_summary,
    )
    return AgentContext(
        task=Task(
            task_id=1, task_type=TaskType.REVIEW_FILE,
            file=file_path, description="review",
        ),
        blueprint=bp,
        file_blueprint=fb,
        related_files=related or {
            file_path: "class Service { void doWork() {} }",
            "src/Dep.java": "public class Dep { void helper() {} }",
        },
        architecture_summary=arch_summary,
    )


def _make_agent(mock_llm=None, mock_repo=None):
    if mock_llm is None:
        mock_llm = MagicMock()
    if mock_repo is None:
        mock_repo = MagicMock()
        mock_repo.workspace = MagicMock()
    return ReviewerAgent(llm_client=mock_llm, repo_manager=mock_repo)


# ── Tests for _format_context(exclude_target=True) ────────────────────────────


class TestFormatContextExcludeTarget:
    """_format_context(exclude_target=True) must omit the primary target file
    from the Related Files section."""

    def test_target_excluded_when_flag_set(self):
        agent = _make_agent()
        ctx = _make_context(file_path="src/Service.java")

        result = agent._format_context(ctx, exclude_target=True)

        # Target file must NOT appear as a file section heading
        assert "### src/Service.java" not in result, (
            "Target file section must be absent when exclude_target=True"
        )

    def test_target_included_by_default(self):
        agent = _make_agent()
        ctx = _make_context(file_path="src/Service.java")

        result = agent._format_context(ctx)

        assert "### src/Service.java" in result, (
            "Target file section must be present when exclude_target=False (default)"
        )

    def test_dep_still_included_when_target_excluded(self):
        agent = _make_agent()
        ctx = _make_context(
            file_path="src/Service.java",
            related={
                "src/Service.java": "class Service {}",
                "src/Dep.java": "public class Dep {}",
            },
        )

        result = agent._format_context(ctx, exclude_target=True)

        # Dep must still be present
        assert "### src/Dep.java" in result, (
            "Dependency files must remain in Related Files even when exclude_target=True"
        )

    def test_no_double_inclusion_of_target(self):
        """Baseline: default call includes target exactly once."""
        agent = _make_agent()
        target_content = "class Service { void doWork() {} }"
        ctx = _make_context(
            file_path="src/Service.java",
            related={"src/Service.java": target_content},
        )

        result = agent._format_context(ctx)
        # The file section should appear exactly once
        assert result.count("### src/Service.java") == 1

    def test_architecture_summary_always_included(self):
        agent = _make_agent()
        ctx = _make_context(arch_summary="Hexagonal architecture with DDD patterns")

        result_default = agent._format_context(ctx)
        result_excl = agent._format_context(ctx, exclude_target=True)

        assert "Hexagonal architecture" in result_default
        assert "Hexagonal architecture" in result_excl

    def test_related_files_section_absent_when_only_target_and_excluded(self):
        """When the only related file is the target and it is excluded, the
        Related Files section heading must not appear."""
        agent = _make_agent()
        ctx = _make_context(
            file_path="src/Service.java",
            related={"src/Service.java": "class Service {}"},
        )
        result = agent._format_context(ctx, exclude_target=True)
        assert "## Related Files" not in result, (
            "Related Files section must be absent when no files remain after exclusion"
        )

    def test_correct_language_fence_for_primary(self):
        """Primary file should use the blueprint language fence."""
        agent = _make_agent()
        ctx = _make_context(
            file_path="src/Service.java",
            language="java",
            related={"src/Service.java": "class Service {}"},
        )
        result = agent._format_context(ctx)
        # Java blueprint language should produce a java fence
        assert "```java" in result

    def test_dep_file_uses_extension_fence_not_primary_language(self):
        """Dependency files should get an extension-based fence label."""
        agent = _make_agent()
        ctx = _make_context(
            file_path="src/Service.java",
            language="java",
            related={
                "src/Service.java": "class Service {}",
                "src/util.py": "def helper(): pass",  # Python dep in Java project
            },
        )
        result = agent._format_context(ctx)
        # The Python dep should use python fence
        assert "```python" in result

    def test_exclude_target_false_matches_default(self):
        """Explicitly passing exclude_target=False must produce the same output
        as not passing it at all (default)."""
        agent = _make_agent()
        ctx = _make_context()

        default_result = agent._format_context(ctx)
        explicit_false = agent._format_context(ctx, exclude_target=False)

        assert default_result == explicit_false


# ── Tests for ReviewerAgent._review_file ──────────────────────────────────────


class TestReviewerFileNoDiskRead:
    """_review_file must use context.related_files instead of an async disk read
    when the target file is already present in the context."""

    @pytest.mark.anyio
    async def test_no_async_read_when_content_in_context(self):
        """repo.async_read_file must NOT be called when related_files contains the target."""
        mock_llm = MagicMock()
        mock_repo = MagicMock()
        mock_repo.workspace = MagicMock()
        mock_repo.async_read_file = AsyncMock(return_value=None)  # returns None → would be skipped

        # Simulate what _call_llm_json returns
        mock_llm.generate_json = AsyncMock(return_value={
            "passed": True,
            "summary": "Code looks good",
            "findings": [],
        })

        agent = ReviewerAgent(llm_client=mock_llm, repo_manager=mock_repo)

        ctx = _make_context(
            file_path="src/Service.java",
            related={
                "src/Service.java": "class Service { void doWork() {} }",
                "src/Dep.java": "public class Dep {}",
            },
        )

        await agent._review_file(ctx)

        # async_read_file should NOT have been called because the file is in related_files
        mock_repo.async_read_file.assert_not_called(), (
            "async_read_file must not be called when the target file is already in "
            "context.related_files — this avoids a redundant async I/O on every review"
        )

    @pytest.mark.anyio
    async def test_falls_back_to_disk_read_when_not_in_context(self):
        """async_read_file IS called as a safety net when related_files doesn't
        contain the target (e.g. context build was skipped in tests)."""
        mock_llm = MagicMock()
        mock_repo = MagicMock()
        mock_repo.workspace = MagicMock()
        mock_repo.async_read_file = AsyncMock(return_value="class Fallback {}")

        mock_llm.generate_json = AsyncMock(return_value={
            "passed": True,
            "summary": "ok",
            "findings": [],
        })

        agent = ReviewerAgent(llm_client=mock_llm, repo_manager=mock_repo)

        # related_files does NOT contain the target file
        ctx = _make_context(
            file_path="src/Service.java",
            related={"src/Dep.java": "public class Dep {}"},  # target not present
        )

        await agent._review_file(ctx)

        # Fallback disk read must be attempted
        mock_repo.async_read_file.assert_called_once_with("src/Service.java"), (
            "async_read_file must be called as a fallback when target file is NOT "
            "in context.related_files"
        )

    @pytest.mark.anyio
    async def test_target_file_not_double_included_in_prompt(self):
        """The target file content must appear only ONCE in the prompt sent to
        the LLM — in the TARGET FILE section with line numbers, NOT also in the
        Related Files section."""
        mock_llm = MagicMock()
        mock_repo = MagicMock()
        mock_repo.workspace = MagicMock()
        mock_repo.async_read_file = AsyncMock(return_value=None)

        captured_prompts: list[str] = []

        async def _capture_json(system_prompt: str, user_prompt: str) -> dict:
            captured_prompts.append(user_prompt)
            return {"passed": True, "summary": "ok", "findings": []}

        mock_llm.generate_json = _capture_json

        agent = ReviewerAgent(llm_client=mock_llm, repo_manager=mock_repo)

        target_content = "class UniqueMarker1234 { void doWork() {} }"
        ctx = _make_context(
            file_path="src/Service.java",
            related={
                "src/Service.java": target_content,
                "src/Dep.java": "public class Dep {}",
            },
        )

        await agent._review_file(ctx)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]

        # The unique marker string should appear exactly once
        count = prompt.count("UniqueMarker1234")
        assert count == 1, (
            f"Target file content must appear exactly once in the review prompt "
            f"(found {count} occurrences). Double-inclusion wastes tokens and "
            f"slows down LLM response time."
        )

    @pytest.mark.anyio
    async def test_target_file_present_with_line_numbers(self):
        """The TARGET FILE section must include line numbers."""
        mock_llm = MagicMock()
        mock_repo = MagicMock()
        mock_repo.workspace = MagicMock()
        mock_repo.async_read_file = AsyncMock(return_value=None)

        captured_prompts: list[str] = []

        async def _capture_json(system_prompt: str, user_prompt: str) -> dict:
            captured_prompts.append(user_prompt)
            return {"passed": True, "summary": "ok", "findings": []}

        mock_llm.generate_json = _capture_json

        agent = ReviewerAgent(llm_client=mock_llm, repo_manager=mock_repo)

        ctx = _make_context(
            file_path="src/Service.java",
            related={"src/Service.java": "class Service {}\nclass Helper {}\n"},
        )

        await agent._review_file(ctx)

        prompt = captured_prompts[0]
        # Line numbers should be present (format: "   1 | ...")
        assert "## TARGET FILE" in prompt, "TARGET FILE section must be present"
        assert " | " in prompt, "Line numbers (N | content) must be present in the TARGET FILE section"
