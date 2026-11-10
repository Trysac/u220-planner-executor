from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from planner_executor import (
    FakeLLM,
    Message,
    PlanError,
    RecordedLLM,
    RecordingExhaustedError,
    ToolFailure,
    parse_plan,
    run,
)


def _make_plan(reply: str) -> str:
    return reply


class TestParsePlan:
    def test_valid_plan(self) -> None:
        reply = '{"steps":[{"id":"s1","tool":"add","args":{"a":2,"b":3}},{"id":"fin","tool":"finish","args":{"answer":"$s1"}}]}'
        plan = parse_plan(reply, tools=["add"])
        assert len(plan.steps) == 2
        assert plan.steps[0].id == "s1"
        assert plan.steps[0].tool == "add"
        assert plan.steps[0].args == {"a": 2, "b": 3}
        assert plan.steps[1].tool == "finish"
        assert plan.steps[1].args["answer"] == "$s1"

    def test_plan_with_only_finish(self) -> None:
        reply = '{"steps":[{"id":"fin","tool":"finish","args":{"answer":"done"}}]}'
        plan = parse_plan(reply)
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "finish"

    def test_invalid_json(self) -> None:
        with pytest.raises(PlanError) as exc_info:
            parse_plan("not json")
        assert exc_info.value.code == "invalid_json"

    def test_missing_finish_empty_steps(self) -> None:
        with pytest.raises(PlanError) as exc_info:
            parse_plan('{"steps":[]}')
        assert exc_info.value.code == "missing_finish"

    def test_unknown_tool(self) -> None:
        reply = '{"steps":[{"id":"s1","tool":"rm","args":{}},{"id":"fin","tool":"finish","args":{"answer":"x"}}]}'
        with pytest.raises(PlanError) as exc_info:
            parse_plan(reply, tools=["add"])
        assert exc_info.value.code == "unknown_tool"

    def test_bad_reference_later_step(self) -> None:
        reply = '{"steps":[{"id":"s1","tool":"noop","args":{"x":"$s2"}},{"id":"s2","tool":"noop","args":{}},{"id":"fin","tool":"finish","args":{"answer":""}}]}'
        with pytest.raises(PlanError) as exc_info:
            parse_plan(reply, tools=["noop"])
        assert exc_info.value.code == "bad_reference"

    def test_bad_reference_unknown(self) -> None:
        reply = '{"steps":[{"id":"s1","tool":"noop","args":{"x":"$unknown"}},{"id":"fin","tool":"finish","args":{"answer":""}}]}'
        with pytest.raises(PlanError) as exc_info:
            parse_plan(reply, tools=["noop"])
        assert exc_info.value.code == "bad_reference"

    def test_too_many_steps_21(self) -> None:
        steps = [{"id": f"s{i}", "tool": "noop", "args": {}} for i in range(21)]
        steps[-1] = {"id": "fin", "tool": "finish", "args": {"answer": ""}}
        reply = json.dumps({"steps": steps})
        with pytest.raises(PlanError) as exc_info:
            parse_plan(reply, tools=["noop"])
        assert exc_info.value.code == "too_many_steps"

    def test_duplicate_id(self) -> None:
        reply = '{"steps":[{"id":"s1","tool":"noop","args":{}},{"id":"s1","tool":"noop","args":{}},{"id":"fin","tool":"finish","args":{"answer":""}}]}'
        with pytest.raises(PlanError) as exc_info:
            parse_plan(reply, tools=["noop"])
        assert exc_info.value.code == "invalid_plan"

    def test_reuse_completed_id(self) -> None:
        reply = '{"steps":[{"id":"s1","tool":"noop","args":{}},{"id":"fin","tool":"finish","args":{"answer":""}}]}'
        with pytest.raises(PlanError) as exc_info:
            parse_plan(reply, tools=["noop"], completed=["s1"])
        assert exc_info.value.code == "invalid_plan"

    def test_invalid_id_format(self) -> None:
        reply = '{"steps":[{"id":"s1!","tool":"noop","args":{}},{"id":"fin","tool":"finish","args":{"answer":""}}]}'
        with pytest.raises(PlanError) as exc_info:
            parse_plan(reply, tools=["noop"])
        assert exc_info.value.code == "invalid_plan"

    def test_finish_not_last(self) -> None:
        reply = '{"steps":[{"id":"fin","tool":"finish","args":{"answer":""}},{"id":"s1","tool":"noop","args":{}}]}'
        with pytest.raises(PlanError) as exc_info:
            parse_plan(reply, tools=["noop"])
        assert exc_info.value.code == "invalid_plan"

    def test_finish_missing_answer(self) -> None:
        reply = '{"steps":[{"id":"fin","tool":"finish","args":{}}]}'
        with pytest.raises(PlanError) as exc_info:
            parse_plan(reply)
        assert exc_info.value.code == "invalid_plan"

    def test_finish_answer_not_string(self) -> None:
        reply = '{"steps":[{"id":"fin","tool":"finish","args":{"answer":123}}]}'
        with pytest.raises(PlanError) as exc_info:
            parse_plan(reply)
        assert exc_info.value.code == "invalid_plan"

    def test_args_not_object(self) -> None:
        reply = '{"steps":[{"id":"s1","tool":"noop","args":[]},{"id":"fin","tool":"finish","args":{"answer":""}}]}'
        with pytest.raises(PlanError) as exc_info:
            parse_plan(reply, tools=["noop"])
        assert exc_info.value.code == "invalid_plan"

    def test_code_fence_stripping(self) -> None:
        reply = '```json\n{"steps":[{"id":"s1","tool":"noop","args":{}},{"id":"fin","tool":"finish","args":{"answer":""}}]}\n```'
        plan = parse_plan(reply, tools=["noop"])
        assert len(plan.steps) == 2

    def test_reply_too_long(self) -> None:
        long_text = "x" * 100001
        with pytest.raises(PlanError) as exc_info:
            parse_plan(long_text)
        assert exc_info.value.code == "invalid_json"

    def test_depth_exceeds_16(self) -> None:
        # Build a deeply nested object inside args
        deep_obj: Any = {}
        for _ in range(17):
            deep_obj = {"nested": deep_obj}
        reply_data = {
            "steps": [
                {"id": "s1", "tool": "noop", "args": deep_obj},
                {"id": "fin", "tool": "finish", "args": {"answer": ""}},
            ]
        }
        reply = json.dumps(reply_data)
        with pytest.raises(PlanError) as exc_info:
            parse_plan(reply, tools=["noop"])
        assert exc_info.value.code == "invalid_plan"

    def test_reference_nested_in_args(self) -> None:
        reply = '{"steps":[{"id":"s1","tool":"noop","args":{}},{"id":"s2","tool":"noop","args":{"x":{"y":"$s1"}}},{"id":"fin","tool":"finish","args":{"answer":""}}]}'
        # This should pass because $s1 is earlier.
        plan = parse_plan(reply, tools=["noop"])
        assert plan.steps[1].args == {"x": {"y": "$s1"}}

    def test_reference_with_extra_text_not_validated(self) -> None:
        # "$s1 apples" is literal, not a reference, so should pass even if s1 not earlier?
        # Actually if s1 is later, then it's not a whole-string ref, so no validation.
        reply = '{"steps":[{"id":"s1","tool":"noop","args":{"x":"$s2 apples"}},{"id":"s2","tool":"noop","args":{}},{"id":"fin","tool":"finish","args":{"answer":""}}]}'
        plan = parse_plan(reply, tools=["noop"])
        assert plan.steps[0].args["x"] == "$s2 apples"

    def test_deeply_nested_json_is_invalid_plan_not_recursion_error(self) -> None:
        # 2000 nested arrays overflow json's decoder: it must surface as PlanError.
        reply = '{"steps":[{"id":"s1","tool":"t","args":{"x":' + "[" * 2000 + "]" * 2000 + "}}]}"
        with pytest.raises(PlanError) as exc_info:
            parse_plan(reply, tools=["t"])
        assert exc_info.value.code == "invalid_plan"


class TestLLMPorts:
    def test_recorded_llm_returns_in_order(self) -> None:
        llm = RecordedLLM(["a", "b"])
        assert llm.complete([], max_tokens=10) == "a"
        assert llm.complete([], max_tokens=10) == "b"

    def test_recorded_llm_raises_when_exhausted(self) -> None:
        llm = RecordedLLM(["a"])
        llm.complete([], max_tokens=10)
        with pytest.raises(RecordingExhaustedError):
            llm.complete([], max_tokens=10)

    def test_fake_llm_calls_fn(self) -> None:
        def fn(messages: Sequence[Message]) -> str:
            return "fake"

        llm = FakeLLM(fn)
        assert llm.complete([], max_tokens=10) == "fake"


class TestRunAcceptance:
    def test_basic_success(self) -> None:
        reply = '{"steps":[{"id":"s1","tool":"add","args":{"a":2,"b":3}},{"id":"s2","tool":"finish","args":{"answer":"$s1"}}]}'
        llm = RecordedLLM([reply])
        tools = {"add": lambda args: str(args["a"] + args["b"])}
        result = run("add 2 and 3", llm, tools)
        assert result.status == "done"
        assert result.answer == "5"
        assert result.replans == 0
        assert result.reason is None
        assert [sr.id for sr in result.steps] == ["s1", "s2"]
        assert result.steps[0].ok is True
        assert result.steps[0].output == "5"
        assert result.steps[1].ok is True
        assert result.steps[1].output == "5"

    def test_tool_failure_replan_success(self) -> None:
        first_reply = '{"steps":[{"id":"s1","tool":"divide","args":{"a":1,"b":0}},{"id":"fin","tool":"finish","args":{"answer":"$s1"}}]}'
        second_reply = '{"steps":[{"id":"s2","tool":"noop","args":{}},{"id":"fin2","tool":"finish","args":{"answer":"ok"}}]}'
        llm = RecordedLLM([first_reply, second_reply])
        tools = {
            "divide": lambda args: (_ for _ in ()).throw(ToolFailure("division by zero")),
            "noop": lambda args: "done",
        }
        events: list[dict[str, object]] = []
        result = run("divide", llm, tools, on_event=lambda e: events.append(e))
        assert result.status == "done"
        assert result.answer == "ok"
        assert result.replans == 1
        assert result.reason is None
        # Check messages sent to LLM
        # We can use FakeLLM to capture messages

    def test_max_replans_exhausted(self) -> None:
        failing_reply = '{"steps":[{"id":"s1","tool":"fail","args":{}},{"id":"fin","tool":"finish","args":{"answer":""}}]}'
        llm = RecordedLLM([failing_reply, failing_reply, failing_reply])
        tools = {"fail": lambda args: (_ for _ in ()).throw(ToolFailure("always fails"))}
        result = run("goal", llm, tools, max_replans=2)
        assert result.status == "failed"
        assert result.reason == "replans_exhausted"
        assert result.replans == 2  # initial + 2 replans
        # Ensure LLM called exactly 3 times
        # RecordedLLM would raise if called 4th time; no need.

    def test_invalid_first_plan_then_valid(self) -> None:
        invalid_reply = "not json"
        valid_reply = '{"steps":[{"id":"fin","tool":"finish","args":{"answer":"ok"}}]}'
        llm = RecordedLLM([invalid_reply, valid_reply])
        result = run("goal", llm, {})
        assert result.status == "done"
        assert result.answer == "ok"
        assert result.replans == 1

    def test_tool_returning_non_str_is_tool_error(self) -> None:
        first_reply = '{"steps":[{"id":"s1","tool":"bad","args":{}},{"id":"fin","tool":"finish","args":{"answer":""}}]}'
        second_reply = '{"steps":[{"id":"fin2","tool":"finish","args":{"answer":"recovered"}}]}'
        llm = RecordedLLM([first_reply, second_reply])
        tools = {"bad": lambda args: 123}  # returns int, not str
        result = run("goal", llm, tools, max_replans=1)
        assert result.status == "done"
        assert result.answer == "recovered"
        assert result.replans == 1
        # Check that replan reason was tool_error: can capture events

    def test_output_truncation_and_reference(self) -> None:
        long_output = "x" * 5000
        first_reply = json.dumps(
            {
                "steps": [
                    {"id": "s1", "tool": "long", "args": {}},
                    {"id": "fin", "tool": "finish", "args": {"answer": "$s1"}},
                ]
            }
        )
        llm = RecordedLLM([first_reply])
        tools = {"long": lambda args: long_output}
        result = run("goal", llm, tools)
        assert result.status == "done"
        assert result.answer == long_output[:4000]
        assert result.steps[0].output == long_output[:4000]

    def test_whole_string_reference_substitution_only(self) -> None:
        received: list[Mapping[str, Any]] = []
        reply = json.dumps(
            {
                "steps": [
                    {"id": "s1", "tool": "echo", "args": {}},
                    {
                        "id": "s2",
                        "tool": "echo",
                        "args": {"lit": "$s1 apples", "ref": "$s1"},
                    },
                    {"id": "fin", "tool": "finish", "args": {"answer": "$s2"}},
                ]
            }
        )

        def echo(args: Mapping[str, Any]) -> str:
            received.append(dict(args))
            if not args:
                return "out1"
            return f"lit={args['lit']};ref={args['ref']}"

        result = run("goal", RecordedLLM([reply]), {"echo": echo})
        assert result.status == "done"
        assert received[1] == {"lit": "$s1 apples", "ref": "out1"}
        assert result.answer == "lit=$s1 apples;ref=out1"

    def test_list_of_scalar_arguments_passed_to_tool(self) -> None:
        reply = json.dumps(
            {
                "steps": [
                    {"id": "s1", "tool": "sum", "args": {"nums": [1, 2, 3]}},
                    {"id": "fin", "tool": "finish", "args": {"answer": "$s1"}},
                ]
            }
        )
        tools = {"sum": lambda args: str(sum(args["nums"]))}
        result = run("goal", RecordedLLM([reply]), tools)
        assert result.status == "done"
        assert result.answer == "6"

    def test_unexpected_tool_exception_is_tool_error_without_message(self) -> None:
        captured: list[Sequence[Message]] = []

        def fake_complete(messages: Sequence[Message]) -> str:
            captured.append(list(messages))
            if len(captured) == 1:
                return '{"steps":[{"id":"s1","tool":"boom","args":{}},{"id":"fin","tool":"finish","args":{"answer":""}}]}'
            return '{"steps":[{"id":"fin2","tool":"finish","args":{"answer":"ok"}}]}'

        def boom(args: Mapping[str, Any]) -> str:
            raise ValueError("secret-token-abc")

        events: list[dict[str, object]] = []
        result = run("goal", FakeLLM(fake_complete), {"boom": boom}, on_event=events.append)

        assert result.status == "done"
        assert result.replans == 1
        assert result.steps[0].ok is False
        assert result.steps[0].output == ""
        replan_events = [e for e in events if e["event"] == "replan"]
        assert replan_events[0]["reason"] == "tool_error"
        assert captured[1][-1] == {
            "role": "user",
            "content": "Step s1 failed: tool_error. Completed: none. Return a new plan for the remaining work.",
        }
        for messages in captured:
            for message in messages:
                assert "secret-token-abc" not in message["content"]

    def test_max_steps_exhausted(self) -> None:
        # Plan with many tool steps but no finish? Need finish last.
        steps = [{"id": f"s{i}", "tool": "noop", "args": {}} for i in range(10)]
        steps.append({"id": "fin", "tool": "finish", "args": {"answer": "done"}})
        reply = json.dumps({"steps": steps})
        llm = RecordedLLM([reply])
        tools = {"noop": lambda args: "ok"}
        result = run("goal", llm, tools, max_steps=5)
        assert result.status == "failed"
        assert result.reason == "step_budget_exhausted"
        assert len(result.steps) == 5  # 5 tool steps executed before budget

    def test_llm_exception_propagates(self) -> None:
        def fn(messages: Sequence[Message]) -> str:
            raise RuntimeError("boom")

        llm = FakeLLM(fn)
        with pytest.raises(RuntimeError, match="boom"):
            run("goal", llm, {})

    def test_on_event_contains_no_goal_args_or_outputs(self) -> None:
        events: list[dict[str, object]] = []
        reply = '{"steps":[{"id":"s1","tool":"add","args":{"a":2,"b":3}},{"id":"fin","tool":"finish","args":{"answer":"$s1"}}]}'
        llm = RecordedLLM([reply])
        tools = {"add": lambda args: str(args["a"] + args["b"])}
        run("add 2 and 3", llm, tools, on_event=lambda e: events.append(e))
        step_events = [e for e in events if e["event"] == "step"]
        assert len(step_events) == 2
        for e in step_events:
            assert set(e.keys()) <= {"event", "id", "tool", "ok"}
            assert "args" not in e
            assert "output" not in e
            assert "goal" not in e
        # No replan events in this success case
        assert all(e["event"] != "replan" for e in events)

    def test_on_event_replan_tool_failure(self) -> None:
        events: list[dict[str, object]] = []
        first_reply = '{"steps":[{"id":"s1","tool":"fail","args":{}},{"id":"fin","tool":"finish","args":{"answer":""}}]}'
        second_reply = '{"steps":[{"id":"fin2","tool":"finish","args":{"answer":"ok"}}]}'
        llm = RecordedLLM([first_reply, second_reply])
        tools = {"fail": lambda args: (_ for _ in ()).throw(ToolFailure("bad"))}
        run("goal", llm, tools, on_event=lambda e: events.append(e))
        replan_events = [e for e in events if e["event"] == "replan"]
        assert len(replan_events) == 1
        assert replan_events[0]["reason"] == "tool_failure"
        assert "goal" not in replan_events[0]
        assert "args" not in replan_events[0]

    def test_on_event_replan_tool_error(self) -> None:
        events: list[dict[str, object]] = []
        first_reply = '{"steps":[{"id":"s1","tool":"bad","args":{}},{"id":"fin","tool":"finish","args":{"answer":""}}]}'
        second_reply = '{"steps":[{"id":"fin2","tool":"finish","args":{"answer":"ok"}}]}'
        llm = RecordedLLM([first_reply, second_reply])
        tools = {"bad": lambda args: 123}  # non-str
        run("goal", llm, tools, on_event=lambda e: events.append(e))
        replan_events = [e for e in events if e["event"] == "replan"]
        assert len(replan_events) == 1
        assert replan_events[0]["reason"] == "tool_error"

    def test_on_event_replan_invalid_plan(self) -> None:
        events: list[dict[str, object]] = []
        invalid_reply = "not json"
        valid_reply = '{"steps":[{"id":"fin","tool":"finish","args":{"answer":"ok"}}]}'
        llm = RecordedLLM([invalid_reply, valid_reply])
        run("goal", llm, {}, on_event=lambda e: events.append(e))
        replan_events = [e for e in events if e["event"] == "replan"]
        assert len(replan_events) == 1
        assert replan_events[0]["reason"] == "invalid_plan"

    def test_msg_prompts_exact_and_results_escaping(self) -> None:
        """Verify system prompt and replay user message content."""
        captured_messages: list[Sequence[Message]] = []

        def fake_complete(messages: Sequence[Message]) -> str:
            captured_messages.append(messages)
            if len(captured_messages) == 1:
                return '{"steps":[{"id":"s1","tool":"fail","args":{}},{"id":"fin","tool":"finish","args":{"answer":""}}]}'
            else:
                return '{"steps":[{"id":"fin2","tool":"finish","args":{"answer":"ok"}}]}'

        llm = FakeLLM(fake_complete)
        tools = {"fail": lambda args: (_ for _ in ()).throw(ToolFailure("bad &< >"))}
        run("goal", llm, tools)
        # First call messages
        first_messages = captured_messages[0]
        assert first_messages[0] == {
            "role": "system",
            "content": 'You are a planner. Reply with JSON only: {"steps": [{"id": "s1", "tool": "<tool>", "args": {}}]}. The last step must be the tool finish with args {"answer": "..."}. A string argument "$<id>" is replaced by the output of that earlier step. Tools: fail, finish.',
        }
        assert first_messages[1] == {"role": "user", "content": "Goal: goal"}
        # Second call messages should include assistant reply and user message with failure info and results lines.
        second_messages = captured_messages[1]
        # after the first failure, messages = [sys, user, assistant, user]
        # second call is with messages = [sys, user, assistant, user]
        assert second_messages[2] == {
            "role": "assistant",
            "content": '{"steps":[{"id":"s1","tool":"fail","args":{}},{"id":"fin","tool":"finish","args":{"answer":""}}]}',
        }
        expected_user = (
            "Step s1 failed: bad &< >. Completed: none. Return a new plan for the remaining work."
        )
        assert second_messages[3] == {"role": "user", "content": expected_user}

    def test_reference_to_completed_step_in_later_plan(self) -> None:
        steps = [
            {"id": "s1", "tool": "get", "args": {"key": "value"}},
            {"id": "s2", "tool": "fail", "args": {}},
            {"id": "fin1", "tool": "finish", "args": {"answer": ""}},
        ]
        first_reply = json.dumps({"steps": steps})
        second_reply = json.dumps(
            {
                "steps": [
                    {"id": "s3", "tool": "use", "args": {"input": "$s1"}},
                    {"id": "fin2", "tool": "finish", "args": {"answer": "$s3"}},
                ]
            }
        )
        llm = RecordedLLM([first_reply, second_reply])
        tools = {
            "get": lambda args: "stored",
            "fail": lambda args: (_ for _ in ()).throw(ToolFailure("fail")),
            "use": lambda args: "used " + args["input"],
        }
        result = run("goal", llm, tools)
        assert result.status == "done"
        assert result.answer == "used stored"
        assert result.replans == 1

    def test_tool_failure_replan_message_exact(self) -> None:
        """The first plan's step s1 calls a tool that raises ToolFailure(
        'division by zero'), so the second LLM call gets the exact replan prompt."""
        captured: list[Sequence[Message]] = []

        def fake_complete(messages: Sequence[Message]) -> str:
            captured.append(list(messages))
            if len(captured) == 1:
                return '{"steps":[{"id":"s1","tool":"divide","args":{}},{"id":"fin","tool":"finish","args":{"answer":""}}]}'
            return '{"steps":[{"id":"fin2","tool":"finish","args":{"answer":"ok"}}]}'

        llm = FakeLLM(fake_complete)
        tools = {"divide": lambda args: (_ for _ in ()).throw(ToolFailure("division by zero"))}
        result = run("goal", llm, tools)

        assert result.status == "done"
        assert result.answer == "ok"
        assert result.replans == 1
        assert captured[1][-1] == {
            "role": "user",
            "content": "Step s1 failed: division by zero. Completed: none. Return a new plan for the remaining work.",
        }
