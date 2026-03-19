"""Reviewer agent - performs hierarchical code review."""

from __future__ import annotations

import logging
from typing import Any

from agents.base_agent import BaseAgent
from core.models import (
    AgentContext,
    AgentRole,
    ReviewFinding,
    ReviewLevel,
    ReviewResult,
    TaskResult,
    TaskType,
)

logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    role = AgentRole.REVIEWER

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior code reviewer agent. You perform structured, deterministic "
            "code reviews.\n\n"

            "You MUST respond with a JSON object (no markdown fences, no prose outside JSON):\n"
            "{\n"
            '  "passed": true or false,\n'
            '  "summary": "One-sentence overall assessment",\n'
            '  "findings": [\n'
            "    {\n"
            '      "severity": "critical" | "warning" | "info",\n'
            '      "file": "path/to/file",\n'
            '      "line": <line-number-or-null>,\n'
            '      "message": "What is wrong",\n'
            '      "suggestion": "How to fix it"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"

            "SEVERITY DEFINITIONS (follow strictly):\n"
            "- critical: Code will not compile, will crash at runtime, has a security "
            "vulnerability (injection, XSS, SSRF), or violates the architecture blueprint "
            "(wrong imports, missing exports, layer violation). ONLY use critical for "
            "issues that MUST be fixed for the code to work correctly.\n"
            "- warning: Code works but has a significant quality issue — missing error "
            "handling for likely failure cases, resource leaks, race conditions, missing "
            "input validation on public APIs.\n"
            "- info: Style suggestions, minor improvements, optional optimizations. "
            "These do NOT affect correctness.\n\n"

            "PASS/FAIL RULE:\n"
            "- Set \"passed\": false if there is AT LEAST ONE critical finding\n"
            "- Set \"passed\": true if there are ZERO critical findings "
            "(warnings and info are acceptable)\n\n"

            "REVIEW CHECKLIST:\n"
            "1. Does the file compile/parse without syntax errors?\n"
            "2. Are all imports valid and do they match the dependency graph?\n"
            "3. Are all exports from the blueprint actually defined?\n"
            "4. Are there any runtime errors (null dereference, type mismatch, "
            "missing return statements)?\n"
            "5. Is error handling present for I/O operations and external calls?\n"
            "6. Are there security issues (injection, hardcoded secrets, path traversal)?\n"
            "7. Does the code follow the architectural layer constraints?\n\n"

            "IMPORTANT:\n"
            "- Do NOT flag style preferences as critical or warning\n"
            "- Do NOT hallucinate issues — only report problems you can identify in the "
            "actual code shown to you\n"
            "- If the code is correct and follows the blueprint, return passed: true "
            "with an empty findings array\n"
            "- Be specific: reference exact line numbers and exact variable/method names"
        )

    async def execute(self, context: AgentContext) -> TaskResult:
        """Execute the appropriate review level."""
        from pydantic import ValidationError

        task_type = context.task.task_type

        try:
            if task_type == TaskType.REVIEW_FILE:
                review = await self._review_file(context)
            elif task_type == TaskType.REVIEW_MODULE:
                review = await self._review_module(context)
            elif task_type == TaskType.REVIEW_ARCHITECTURE:
                review = await self._review_architecture(context)
            else:
                return TaskResult(success=False, errors=[f"Unknown review type: {task_type}"])
        except ValidationError as e:
            return TaskResult(
                success=False,
                errors=[f"LLM output validation failed: {e}"],
            )

        critical_count = sum(1 for f in review.findings if f.severity == "critical")
        status = "PASSED" if review.passed else f"NEEDS WORK ({critical_count} critical)"

        return TaskResult(
            # Review completing (even with findings) is a successful task execution.
            # Findings are reported as output/warnings, not task failures.
            success=True,
            output=f"[{status}] {review.summary}",
            errors=[f.message for f in review.findings if f.severity == "critical"],
            metrics={
                "review_level": review.level.value,
                "findings_count": len(review.findings),
                "critical_count": critical_count,
                "passed": review.passed,
            },
        )

    async def _review_file(self, context: AgentContext) -> ReviewResult:
        """Review a single file."""
        # Build the context string, but exclude the target file from the generic
        # "Related Files" section so it is only rendered once — below, with line
        # numbers — rather than twice at double the token cost.
        formatted = self._format_context(context, exclude_target=True)

        # Use the file content already assembled by ContextBuilder to avoid a
        # redundant async disk read on every review call.  Fall back to a direct
        # read only when the file was not included in the context (safety net).
        target_file_content: str = context.related_files.get(context.task.file, "")
        if not target_file_content:
            target_file_content = await self.repo.async_read_file(context.task.file) or ""

        if target_file_content:
            numbered_lines = []
            for i, line in enumerate(target_file_content.splitlines(), start=1):
                numbered_lines.append(f"{i:4d} | {line}")
            numbered_content = "\n".join(numbered_lines)

            # Extract language name for code fence
            lang_name = ""
            if context.file_blueprint and context.file_blueprint.language:
                from core.language import get_language_profile
                lang_name = get_language_profile(context.file_blueprint.language).code_fence_name
            else:
                lang_name = "code"

            formatted += (
                f"\n\n## TARGET FILE ({context.task.file})\n"
                f"```{lang_name}\n"
                f"{numbered_content}\n"
                f"```"
            )

        fb = context.file_blueprint

        blueprint_checklist = ""
        if fb:
            exports_check = ", ".join(fb.exports) if fb.exports else "none specified"
            deps_check = ", ".join(fb.depends_on) if fb.depends_on else "none"
            blueprint_checklist = (
                f"\n\nBLUEPRINT REQUIREMENTS TO VERIFY:\n"
                f"- File MUST export: {exports_check}\n"
                f"- File depends on: {deps_check}\n"
                f"- File purpose: {fb.purpose}\n"
                f"- File layer: {fb.layer}\n"
                f"- Mark as critical if any listed export is missing or if imports "
                f"reference files not in depends_on"
            )

        prompt = (
            f"{formatted}\n\n"
            f"Review the file: {context.task.file}\n"
            f"{blueprint_checklist}\n\n"
            "Check each item in the review checklist from your instructions. "
            "For each issue found, provide the exact line number and a specific "
            "fix suggestion."
        )

        data = await self._call_llm_json(prompt)
        return self._parse_review(data, ReviewLevel.FILE)

    async def _review_module(self, context: AgentContext) -> ReviewResult:
        """Review module-level consistency."""
        formatted = self._format_context(context)
        prompt = (
            f"{formatted}\n\n"
            "Perform a MODULE-LEVEL review. Check the following across all files shown:\n"
            "1. Do controllers call service methods that actually exist with correct signatures?\n"
            "2. Do services call repository methods that actually exist with correct signatures?\n"
            "3. Are return types consistent across layer boundaries?\n"
            "4. Are all imports between files valid (no references to non-existent classes)?\n"
            "5. Is error handling consistent (same pattern for similar error types)?\n"
            "6. Is dependency injection used consistently across the module?\n\n"
            "IMPORTANT — FULLSTACK AWARENESS:\n"
            "- This may be a fullstack project with SEPARATE backend and frontend modules.\n"
            "- Backend files (e.g. .java, .py, .go) and frontend files (e.g. .tsx, .vue)\n"
            "  are in DIFFERENT directories and are INDEPENDENT codebases.\n"
            "- Do NOT flag a Java file for being 'in a TypeScript project' or vice versa.\n"
            "- Only check cross-file consistency WITHIN the same language/module.\n\n"
            "For each issue, specify the EXACT file and line where the mismatch occurs "
            "and which other file it conflicts with."
        )

        data = await self._call_llm_json(prompt)
        return self._parse_review(data, ReviewLevel.MODULE)

    async def _review_architecture(self, context: AgentContext) -> ReviewResult:
        """Review architecture-level concerns."""
        formatted = self._format_context(context)
        prompt = (
            f"{formatted}\n\n"
            "Perform an ARCHITECTURE-LEVEL review. Check specifically:\n"
            "1. Layer violations: Does any controller directly import a repository "
            "(skipping the service layer)?\n"
            "2. Dependency cycles: Does file A import file B which imports file A?\n"
            "3. Missing layers: Is there a controller without a corresponding service?\n"
            "4. Cross-cutting concerns: Is authentication/authorization applied consistently?\n"
            "5. Configuration: Are secrets hardcoded instead of externalized?\n\n"
            "IMPORTANT — FULLSTACK AWARENESS:\n"
            "- This may be a fullstack project with SEPARATE backend and frontend modules.\n"
            "- Backend (e.g. .java, .py) and frontend (e.g. .tsx, .vue) files are\n"
            "  independent codebases in different directories.\n"
            "- Do NOT flag language mismatches between backend and frontend modules.\n"
            "- Check layer violations WITHIN each module only.\n\n"
            "ONLY report issues you can specifically identify in the code shown. "
            "Do NOT report vague concerns like 'consider scalability' without "
            "pointing to a specific file and line."
        )

        data = await self._call_llm_json(prompt)
        return self._parse_review(data, ReviewLevel.ARCHITECTURE)

    def _parse_review(self, data: dict[str, Any], level: ReviewLevel) -> ReviewResult:
        from core.llm_schema import validate_review_response

        # Raises pydantic.ValidationError if required fields are missing —
        # caught by execute() which returns TaskResult(success=False).
        validated = validate_review_response(data)

        findings = [
            ReviewFinding(
                level=level,
                severity=f.severity,
                file=f.file,
                line=f.line,
                message=f.message,
                suggestion=f.suggestion,
            )
            for f in validated.findings
        ]
        return ReviewResult(
            level=level,
            passed=validated.passed,
            findings=findings,
            summary=validated.summary,
        )
