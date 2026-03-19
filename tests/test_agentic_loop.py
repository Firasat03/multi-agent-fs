"""Tests for BaseAgent.execute_agentic() loop behaviour.

Covers the budget-guard fix (tool calls must be executed before the nudge
message is appended) and the _call_llm continuation prompt cap.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.base_agent import BaseAgent
from agents.coder_agent import CoderAgent
from core.llm_client import ToolCall
from core.models import (
    AgentContext,
    FileBlueprint,
    RepositoryBlueprint,
    Task,
    TaskType,
)

# Named constant for the large-prompt size used in continuation tests.
_LARGE_PROMPT_SIZE = 10_000


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_context(file_path: str = "src/Service.java") -> AgentContext:
    fb = FileBlueprint(path=file_path, purpose="Test", depends_on=[])
    bp = RepositoryBlueprint(
        name="test",
        description="test project",
        architecture_style="REST",
        tech_stack={"language": "java"},
        file_blueprints=[fb],
    )
    return AgentContext(
        task=Task(
            task_id=1,
            task_type=TaskType.GENERATE_FILE,
            file=file_path,
            description="generate",
        ),
        blueprint=bp,
        file_blueprint=fb,
    )


def _make_response(
    *,
    stop_reason: str = "tool_use",
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    usage: dict | None = None,
) -> MagicMock:
    r = MagicMock()
    r.stop_reason = stop_reason
    r.content = content
    r.raw_content = content or []
    r.tool_calls = tool_calls or []
    r.usage = usage or {"input_tokens": 10, "output_tokens": 20}
    return r


def _make_agent(tmp_path, *, llm_responses: list[MagicMock]) -> CoderAgent:
    mock_llm = MagicMock()
    # generate_with_tools returns responses in sequence
    mock_llm.generate_with_tools = AsyncMock(side_effect=llm_responses)

    mock_repo = MagicMock()
    mock_repo.workspace = tmp_path
    mock_repo.async_write_file = AsyncMock(return_value=tmp_path / "Service.java")
    mock_repo.async_read_file = AsyncMock(return_value="existing content")
    mock_repo._last_broken_imports = []
    mock_repo.get_repo_index = MagicMock(return_value=MagicMock(files=[]))

    return CoderAgent(llm_client=mock_llm, repo_manager=mock_repo)


# ── Budget guard tests ─────────────────────────────────────────────────────────


class TestBudgetGuardExecutesToolsFirst:
    """Budget guard must execute pending tool calls before appending the nudge.

    Previously the guard sent dummy 'skipped' results instead of executing
    the tool call, which meant read_file calls were never answered and the
    LLM generated broken code.
    """

    @pytest.mark.anyio
    async def test_read_file_executed_when_budget_guard_triggers(self, tmp_path):
        """read_file tool must be called even when budget guard is active."""
        ctx = _make_context()
        file_path = ctx.file_blueprint.path

        read_call = ToolCall(
            tool_use_id="tc-read",
            name="read_file",
            input={"path": "src/Dep.java"},
        )
        write_call = ToolCall(
            tool_use_id="tc-write",
            name="write_file",
            input={"path": file_path, "content": "class Service {}"},
        )

        # max_iterations=10 → budget guard fires at iteration >= 5
        # Turn 0-4: LLM keeps reading (no write_file)
        # Turn 5 (≥ max//2): budget guard should fire but still execute read_file
        # Turn 6: LLM calls write_file
        read_responses = [
            _make_response(tool_calls=[read_call]) for _ in range(5)
        ]
        write_response = _make_response(tool_calls=[write_call])
        end_response = _make_response(stop_reason="end_turn", content="done")

        agent = _make_agent(tmp_path, llm_responses=read_responses + [write_response, end_response])
        agent.max_iterations = 10

        # Track whether _tool_read_file was actually called
        real_read_calls: list[str] = []

        async def _tracking_read(inp: dict) -> str:
            real_read_calls.append(inp.get("path", ""))
            return "class Dep { void doThing() {} }"

        agent._tool_read_file = _tracking_read
        agent._tool_write_file = AsyncMock(return_value=f"Written 14 bytes to {file_path}")

        result = await agent.execute_agentic(ctx)

        # read_file must have been called on turns 0-5 (including the budget-guard turn)
        assert len(real_read_calls) >= 5, (
            f"read_file should have been called ≥5 times (including budget-guard iterations), "
            f"got {len(real_read_calls)}"
        )

    @pytest.mark.anyio
    async def test_budget_guard_nudge_content_has_warning(self, tmp_path):
        """The budget guard nudge message must include the warning text and target path."""
        ctx = _make_context("src/MyService.java")
        file_path = ctx.file_blueprint.path

        read_call = ToolCall(
            tool_use_id="tc-read",
            name="read_file",
            input={"path": "src/Dep.java"},
        )

        # 6 read-only iterations, then a write_file to finish
        read_response = _make_response(tool_calls=[read_call])
        write_call = ToolCall(
            tool_use_id="tc-write",
            name="write_file",
            input={"path": file_path, "content": "class MyService {}"},
        )
        write_response = _make_response(tool_calls=[write_call])

        agent = _make_agent(
            tmp_path,
            llm_responses=[read_response] * 6 + [write_response],
        )
        agent.max_iterations = 10

        # Capture all messages sent to generate_with_tools
        all_messages: list[list[dict]] = []

        async def _capture_generate(**kwargs):
            all_messages.append(kwargs.get("messages", []))
            idx = len(all_messages) - 1
            responses = [read_response] * 6 + [write_response]
            return responses[min(idx, len(responses) - 1)]

        agent.llm.generate_with_tools = _capture_generate
        agent._tool_read_file = AsyncMock(return_value="// dep content")
        agent._tool_write_file = AsyncMock(
            return_value=f"Written 18 bytes to {file_path}"
        )

        await agent.execute_agentic(ctx)

        # Find a user message that contains the budget-guard nudge text
        nudge_found = False
        for msgs in all_messages:
            for msg in msgs:
                if msg.get("role") != "user":
                    continue
                content = msg.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                            and "WARNING" in block.get("text", "")
                            and file_path in block.get("text", "")
                        ):
                            nudge_found = True
                            break
                if nudge_found:
                    break
            if nudge_found:
                break

        assert nudge_found, (
            "Budget guard must append a 'WARNING' text block containing the target file path "
            "to the user message when iterations ≥ max_iterations // 2 and file not written"
        )

    @pytest.mark.anyio
    async def test_no_dummy_results_in_budget_guard(self, tmp_path):
        """Tool results must NEVER contain '[skipped — budget guard]' text."""
        ctx = _make_context("src/Svc.java")
        file_path = ctx.file_blueprint.path

        read_call = ToolCall(
            tool_use_id="tc-read",
            name="read_file",
            input={"path": "src/A.java"},
        )
        write_call = ToolCall(
            tool_use_id="tc-write",
            name="write_file",
            input={"path": file_path, "content": "class Svc {}"},
        )

        read_response = _make_response(tool_calls=[read_call])
        write_response = _make_response(tool_calls=[write_call])

        agent = _make_agent(
            tmp_path,
            llm_responses=[read_response] * 8 + [write_response],
        )
        agent.max_iterations = 10

        all_messages: list[list] = []

        async def _capture(**kwargs):
            all_messages.append(list(kwargs.get("messages", [])))
            idx = len(all_messages) - 1
            responses = [read_response] * 8 + [write_response]
            return responses[min(idx, len(responses) - 1)]

        agent.llm.generate_with_tools = _capture
        agent._tool_read_file = AsyncMock(return_value="real file content")
        agent._tool_write_file = AsyncMock(
            return_value=f"Written 12 bytes to {file_path}"
        )

        await agent.execute_agentic(ctx)

        # Search every message for dummy-result text
        for msgs in all_messages:
            for msg in msgs:
                content = msg.get("content", "")
                if isinstance(content, str):
                    assert "skipped — budget guard" not in content, (
                        "Dummy '[skipped — budget guard]' results must not appear in messages"
                    )
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            block_content = block.get("content", "")
                            assert "skipped — budget guard" not in str(block_content), (
                                "Dummy '[skipped — budget guard]' results must not appear in messages"
                            )


# ── _call_llm continuation prompt cap tests ────────────────────────────────────


class TestCallLlmContinuationPromptCap:
    """Continuation prompts must cap the re-sent user_prompt at 2000 chars."""

    @pytest.mark.anyio
    async def test_large_prompt_is_capped_in_continuation(self, tmp_path):
        """When user_prompt > 2000 chars, the continuation prompt must be < 3000 chars total."""
        mock_llm = MagicMock()
        mock_repo = MagicMock()
        mock_repo.workspace = tmp_path

        agent = CoderAgent(llm_client=mock_llm, repo_manager=mock_repo)

        large_prompt = "X" * _LARGE_PROMPT_SIZE  # 10k char prompt (typical for fix tasks with file content)

        # First call: truncated (hits max_tokens)
        first_resp = MagicMock()
        first_resp.stop_reason = "max_tokens"
        first_resp.content = "partial code..."
        first_resp.usage = {"input_tokens": 100, "output_tokens": 200}

        # Second call: complete
        second_resp = MagicMock()
        second_resp.stop_reason = "end_turn"
        second_resp.content = " // rest of code"
        second_resp.usage = {"input_tokens": 50, "output_tokens": 100}

        captured_prompts: list[str] = []

        async def _mock_generate(**kwargs):
            captured_prompts.append(kwargs.get("user_prompt", ""))
            return first_resp if len(captured_prompts) == 1 else second_resp

        mock_llm.generate = _mock_generate

        result = await agent._call_llm(large_prompt)

        assert len(captured_prompts) == 2, "Expected exactly 2 LLM calls (initial + continuation)"

        continuation_prompt = captured_prompts[1]
        # The continuation must NOT re-send the full 10k-char prompt
        # It must be capped at 2000 (context) + ~800 (tail + instruction) = ~2800 chars max
        assert len(continuation_prompt) < 5_000, (
            f"Continuation prompt ({len(continuation_prompt)} chars) re-sent too much context. "
            f"Expected < 5000 chars (capped context [{BaseAgent._CONTINUATION_PROMPT_CTX_CHARS} chars] + tail + instruction)"
        )
        # Must still contain the continuation instruction
        assert "cut off" in continuation_prompt.lower(), (
            "Continuation prompt must explain the output was cut off"
        )
        # Must contain the trimmed context marker
        assert "context trimmed" in continuation_prompt, (
            "Continuation prompt must indicate that context was trimmed"
        )

    @pytest.mark.anyio
    async def test_small_prompt_not_trimmed_in_continuation(self, tmp_path):
        """When user_prompt ≤ 2000 chars, the full prompt must be included in the continuation."""
        mock_llm = MagicMock()
        mock_repo = MagicMock()
        mock_repo.workspace = tmp_path

        agent = CoderAgent(llm_client=mock_llm, repo_manager=mock_repo)

        small_prompt = "Generate a short config file for project X."  # 46 chars

        first_resp = MagicMock()
        first_resp.stop_reason = "max_tokens"
        first_resp.content = "key=value"
        first_resp.usage = {"input_tokens": 10, "output_tokens": 20}

        second_resp = MagicMock()
        second_resp.stop_reason = "end_turn"
        second_resp.content = "\nmore=stuff"
        second_resp.usage = {"input_tokens": 10, "output_tokens": 20}

        captured: list[str] = []

        async def _mock_generate(**kwargs):
            captured.append(kwargs.get("user_prompt", ""))
            return first_resp if len(captured) == 1 else second_resp

        mock_llm.generate = _mock_generate

        await agent._call_llm(small_prompt)

        assert len(captured) == 2
        assert small_prompt in captured[1], (
            "When user_prompt is short (≤ 2000 chars), it must be fully included in the continuation"
        )
        assert "context trimmed" not in captured[1], (
            "Short prompts must NOT be trimmed"
        )
