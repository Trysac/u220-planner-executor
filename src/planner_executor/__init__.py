from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict

__all__ = [
    "FakeLLM",
    "LLMPort",
    "Message",
    "Plan",
    "PlanError",
    "RecordedLLM",
    "RecordingExhaustedError",
    "RunResult",
    "Step",
    "StepRecord",
    "ToolFailure",
    "__version__",
    "parse_plan",
    "run",
]

__version__ = "0.1.0"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_REFERENCE_PATTERN = re.compile(r"^\$([A-Za-z0-9_-]{1,32})$")
_MAX_REPLY_CHARS = 100_000
_MAX_JSON_DEPTH = 16
_MAX_STEPS = 20
_MAX_OUTPUT_CHARS = 4000
_MAX_TOOL_FAILURE_MSG = 200


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMPort(Protocol):
    def complete(self, messages: Sequence[Message], *, max_tokens: int) -> str: ...


class RecordingExhaustedError(Exception):
    """Raised by :class:`RecordedLLM` when all recorded replies have been consumed."""


class RecordedLLM:
    """An LLM port that returns a fixed sequence of replies in order."""

    def __init__(self, replies: Sequence[str]) -> None:
        self._replies = list(replies)
        self._index = 0

    def complete(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        if self._index >= len(self._replies):
            raise RecordingExhaustedError("no more recorded replies")
        reply = self._replies[self._index]
        self._index += 1
        return reply


class FakeLLM:
    """An LLM port that delegates to a callable."""

    def __init__(self, fn: Callable[[Sequence[Message]], str]) -> None:
        self._fn = fn

    def complete(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        return self._fn(messages)


class PlanError(Exception):
    """Raised when a plan is invalid."""

    def __init__(
        self,
        code: Literal[
            "invalid_json",
            "invalid_plan",
            "unknown_tool",
            "bad_reference",
            "too_many_steps",
            "missing_finish",
        ],
        message: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code


class ToolFailure(Exception):
    """Raised by tools to report an expected failure."""


@dataclass(frozen=True)
class Step:
    id: str
    tool: str
    args: Mapping[str, Any]


@dataclass(frozen=True)
class Plan:
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class StepRecord:
    id: str
    tool: str
    ok: bool
    output: str


@dataclass(frozen=True)
class RunResult:
    status: Literal["done", "failed"]
    answer: str | None
    steps: tuple[StepRecord, ...]
    replans: int
    reason: Literal["replans_exhausted", "step_budget_exhausted"] | None


def _check_json_depth(value: Any, current_depth: int = 0) -> int:
    """Return the maximum nesting depth of a JSON-decoded value."""
    if isinstance(value, dict):
        if not value:
            return current_depth + 1
        return max(_check_json_depth(v, current_depth + 1) for v in value.values())
    if isinstance(value, list):
        if not value:
            return current_depth + 1
        return max(_check_json_depth(item, current_depth + 1) for item in value)
    return current_depth + 1


def _iter_strings(obj: Any) -> Any:
    """Yield all string values in a JSON-like structure."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_strings(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_strings(item)


def _substitute_string(value: str, outputs: Mapping[str, str]) -> str:
    """Substitute a whole-string reference ``$id`` with the recorded output if present."""
    match = _REFERENCE_PATTERN.fullmatch(value)
    if match:
        ref_id = match.group(1)
        if ref_id in outputs:
            return outputs[ref_id]
    return value


def _substitute_value(value: Any, outputs: Mapping[str, str]) -> Any:
    """Recursively substitute whole-string references in a JSON-like value."""
    if isinstance(value, str):
        return _substitute_string(value, outputs)
    if isinstance(value, dict):
        return {k: _substitute_value(v, outputs) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_value(v, outputs) for v in value]
    return value


def _substitute_mapping(args: Mapping[str, Any], outputs: Mapping[str, str]) -> dict[str, Any]:
    """Deep-copy args with references substituted."""
    return {key: _substitute_value(value, outputs) for key, value in args.items()}


def _escape_output(output: str) -> str:
    """Escape ``&``, ``<`` and ``>`` for inclusion in prompt text."""
    return output.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _format_result_lines(completed_outputs: Mapping[str, str]) -> list[str]:
    """Build the result lines for the replan prompt."""
    return [
        f'<result id="{step_id}">{_escape_output(output)}</result>'
        for step_id, output in completed_outputs.items()
    ]


def parse_plan(
    text: str,
    tools: Collection[str] = (),
    completed: Collection[str] = (),
) -> Plan:
    """Parse and validate a planner reply into a :class:`Plan`.

    Args:
        text: The LLM reply (may be surrounded by one code fence).
        tools: Names of available tools (excluding ``finish``).
        completed: IDs of steps that have already been completed.

    Raises:
        PlanError: If the reply is not a valid plan.
    """
    if not isinstance(text, str):
        raise PlanError("invalid_json", "reply is not a string")
    if len(text) > _MAX_REPLY_CHARS:
        raise PlanError("invalid_json", "reply exceeds 100000 characters")

    # Strip one surrounding code fence, if present.
    stripped = text
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped[3:-3]
        # Remove optional language tag before first newline
        if "\n" in stripped:
            first_line_end = stripped.index("\n")
            if stripped[:first_line_end].strip() and not stripped[:first_line_end].startswith("{"):
                stripped = stripped[first_line_end + 1 :]
        stripped = stripped.strip()

    try:
        data = json.loads(stripped)
        depth = _check_json_depth(data)
    except json.JSONDecodeError as exc:
        raise PlanError("invalid_json", "invalid JSON") from exc
    except RecursionError as exc:
        raise PlanError("invalid_plan", "JSON depth exceeds 16") from exc

    if depth > _MAX_JSON_DEPTH:
        raise PlanError("invalid_plan", "JSON depth exceeds 16")

    if not isinstance(data, dict):
        raise PlanError("invalid_plan", "top-level JSON value must be an object")

    steps_raw = data.get("steps")
    if not isinstance(steps_raw, list):
        raise PlanError("invalid_plan", "plan must contain a 'steps' list")

    if len(steps_raw) == 0:
        raise PlanError("missing_finish", "plan has no steps")
    if len(steps_raw) > _MAX_STEPS:
        raise PlanError("too_many_steps", "plan has more than 20 steps")

    tool_names = set(tools)
    completed_ids = set(completed)
    previous_ids: set[str] = set(completed_ids)
    seen_ids: set[str] = set(completed_ids)
    finish_count = 0
    finish_index = None
    parsed_steps: list[Step] = []

    for index, raw_step in enumerate(steps_raw):
        if not isinstance(raw_step, dict):
            raise PlanError("invalid_plan", "each step must be an object")

        step_id = raw_step.get("id")
        if not isinstance(step_id, str) or not _ID_PATTERN.fullmatch(step_id):
            raise PlanError("invalid_plan", "step id is missing or invalid")
        if step_id in seen_ids:
            raise PlanError(
                "invalid_plan",
                f"duplicate or reused step id: {step_id}",
            )
        seen_ids.add(step_id)

        tool = raw_step.get("tool")
        if not isinstance(tool, str):
            raise PlanError("invalid_plan", "step tool must be a string")

        args = raw_step.get("args")
        if not isinstance(args, dict):
            raise PlanError("invalid_plan", "step args must be an object")

        if tool == "finish":
            # Validate finish step: must have string answer and be the last.
            if "answer" not in args:
                raise PlanError("invalid_plan", "finish step requires an 'answer' argument")
            answer = args["answer"]
            if not isinstance(answer, str):
                raise PlanError("invalid_plan", "finish answer must be a string")
            finish_count += 1
            finish_index = index
            if index != len(steps_raw) - 1:
                raise PlanError("invalid_plan", "finish step must be last")
        else:
            if tool not in tool_names:
                raise PlanError("unknown_tool", f"unknown tool: {tool}")

        # Validate references in args.
        for value in _iter_strings(args):
            match = _REFERENCE_PATTERN.fullmatch(value)
            if match:
                ref_id = match.group(1)
                if ref_id not in previous_ids and ref_id not in completed_ids:
                    raise PlanError(
                        "bad_reference", f"reference to unknown or future step: {ref_id}"
                    )

        previous_ids.add(step_id)
        parsed_steps.append(Step(id=step_id, tool=tool, args=args))

    if finish_count != 1:
        raise PlanError("missing_finish", "plan must contain exactly one finish step")
    if finish_index is None or finish_index != len(steps_raw) - 1:
        raise PlanError("invalid_plan", "finish step must be the last step")

    return Plan(steps=tuple(parsed_steps))


def _failure_result(records: list[StepRecord], replans: int) -> RunResult:
    """Build a failed :class:`RunResult` after replans are exhausted."""
    return RunResult(
        status="failed",
        answer=None,
        steps=tuple(records),
        replans=replans,
        reason="replans_exhausted",
    )


def run(
    goal: str,
    llm: LLMPort,
    tools: Mapping[str, Callable[[Mapping[str, Any]], str]],
    *,
    max_steps: int = _MAX_STEPS,
    max_replans: int = 2,
    max_tokens: int = 1024,
    on_event: Callable[[dict[str, object]], None] | None = None,
) -> RunResult:
    """Execute a goal using the planner-executor pattern.

    Args:
        goal: The objective to achieve.
        llm: The LLM port used for planning.
        tools: Mapping of tool names to callables that accept a mapping of arguments.
        max_steps: Maximum number of tool steps across all plans.
        max_replans: Maximum number of replans allowed after the initial plan.
        max_tokens: Passed to the LLM `complete` method.
        on_event: Optional callback for step and replan events.

    Returns:
        A :class:`RunResult` indicating success or failure.
    """
    tool_names = sorted(tools.keys())
    system_prompt = (
        "You are a planner. Reply with JSON only: "
        '{"steps": [{"id": "s1", "tool": "<tool>", "args": {}}]}.'
        ' The last step must be the tool finish with args {"answer": "..."}.'
        ' A string argument "$<id>" is replaced by the output of that earlier step.'
        " Tools: " + ", ".join(tool_names) + ", finish."
    )

    messages: list[Message] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Goal: " + goal},
    ]

    completed_ids: set[str] = set()
    completed_outputs: dict[str, str] = {}  # preserves insertion order
    all_records: list[StepRecord] = []
    replans = 0
    executed_tool_steps = 0

    for _attempt in range(max_replans + 1):
        reply = llm.complete(messages, max_tokens=max_tokens)

        try:
            plan = parse_plan(reply, tools=set(tools.keys()), completed=completed_ids)
        except PlanError as exc:
            if on_event:
                on_event({"event": "replan", "reason": "invalid_plan"})
            if replans >= max_replans:
                return _failure_result(all_records, replans)
            messages.append({"role": "assistant", "content": reply})
            completed_str = ", ".join(sorted(completed_ids)) if completed_ids else "none"
            result_lines = _format_result_lines(completed_outputs)
            content = (
                f"Plan rejected: {exc.code}. Completed: {completed_str}. "
                "Return a new plan for the remaining work."
            )
            if result_lines:
                content += "\n" + "\n".join(result_lines)
            messages.append({"role": "user", "content": content})
            replans += 1
            continue

        # Execute plan steps
        for step in plan.steps:
            if step.tool == "finish":
                # Finish step: answer may contain reference.
                answer = step.args["answer"]
                answer_substituted = _substitute_string(answer, completed_outputs)
                all_records.append(
                    StepRecord(id=step.id, tool="finish", ok=True, output=answer_substituted)
                )
                if on_event:
                    on_event({"event": "step", "id": step.id, "tool": "finish", "ok": True})
                return RunResult(
                    status="done",
                    answer=answer_substituted,
                    steps=tuple(all_records),
                    replans=replans,
                    reason=None,
                )

            # Tool step
            if executed_tool_steps >= max_steps:
                return RunResult(
                    status="failed",
                    answer=None,
                    steps=tuple(all_records),
                    replans=replans,
                    reason="step_budget_exhausted",
                )
            executed_tool_steps += 1

            try:
                substituted_args = _substitute_mapping(step.args, completed_outputs)
                tool_fn = tools[step.tool]
                result = tool_fn(substituted_args)
            except ToolFailure as tf:
                reason = str(tf)[:_MAX_TOOL_FAILURE_MSG]
                all_records.append(StepRecord(id=step.id, tool=step.tool, ok=False, output=""))
                if on_event:
                    on_event({"event": "step", "id": step.id, "tool": step.tool, "ok": False})
                    on_event({"event": "replan", "reason": "tool_failure"})
                if replans >= max_replans:
                    return _failure_result(all_records, replans)
                _append_failure_prompt(
                    messages,
                    reply,
                    f"Step {step.id} failed: {reason}",
                    completed_ids,
                    completed_outputs,
                )
                replans += 1
                break
            except Exception:
                all_records.append(StepRecord(id=step.id, tool=step.tool, ok=False, output=""))
                if on_event:
                    on_event({"event": "step", "id": step.id, "tool": step.tool, "ok": False})
                    on_event({"event": "replan", "reason": "tool_error"})
                if replans >= max_replans:
                    return _failure_result(all_records, replans)
                _append_failure_prompt(
                    messages,
                    reply,
                    f"Step {step.id} failed: tool_error",
                    completed_ids,
                    completed_outputs,
                )
                replans += 1
                break
            else:
                if not isinstance(result, str):
                    all_records.append(StepRecord(id=step.id, tool=step.tool, ok=False, output=""))
                    if on_event:
                        on_event({"event": "step", "id": step.id, "tool": step.tool, "ok": False})
                        on_event({"event": "replan", "reason": "tool_error"})
                    if replans >= max_replans:
                        return _failure_result(all_records, replans)
                    _append_failure_prompt(
                        messages,
                        reply,
                        f"Step {step.id} failed: tool_error",
                        completed_ids,
                        completed_outputs,
                    )
                    replans += 1
                    break

                truncated = result[:_MAX_OUTPUT_CHARS]
                completed_ids.add(step.id)
                completed_outputs[step.id] = truncated
                all_records.append(
                    StepRecord(id=step.id, tool=step.tool, ok=True, output=truncated)
                )
                if on_event:
                    on_event({"event": "step", "id": step.id, "tool": step.tool, "ok": True})

        # Reaching here means the plan ended without a finish (failure) and there
        # are replans left, so continue to the next attempt.

    # Exhausted all attempts without success
    return _failure_result(all_records, replans)


def _append_failure_prompt(
    messages: list[Message],
    assistant_reply: str,
    failure_text: str,
    completed_ids: set[str],
    completed_outputs: Mapping[str, str],
) -> None:
    """Append assistant reply and the failure-triggered user message to the conversation."""
    messages.append({"role": "assistant", "content": assistant_reply})
    completed_str = ", ".join(sorted(completed_ids)) if completed_ids else "none"
    result_lines = _format_result_lines(completed_outputs)
    content = (
        f"{failure_text}. Completed: {completed_str}. Return a new plan for the remaining work."
    )
    if result_lines:
        content += "\n" + "\n".join(result_lines)
    messages.append({"role": "user", "content": content})
