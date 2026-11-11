# Architecture

`planner_executor` implements the planner-executor agent pattern over a text-only LLM port. The
package is a single module, `src/planner_executor/__init__.py`, and depends only on the standard
library.

## Plan format

The planner replies with one JSON object:

```json
{"steps": [{"id": "s1", "tool": "add", "args": {"a": 2, "b": 3}},
           {"id": "s2", "tool": "finish", "args": {"answer": "$s1"}}]}
```

Validation performed by `parse_plan` before anything executes:

- one surrounding code fence is stripped; the reply is at most 100 000 characters;
- JSON depth is at most 16 and the top level is an object with a `steps` list;
- 1-20 steps; ids match `^[A-Za-z0-9_-]{1,32}$`, are unique and do not reuse completed ids;
- `tool` is a registered tool or `finish`, and `args` is a JSON object;
- exactly one `finish` step, last, with a string `answer` argument;
- a string argument that is entirely `$` + id names an earlier or completed step.

Each failure raises `PlanError` with a code: `invalid_json`, `invalid_plan`, `unknown_tool`,
`bad_reference`, `too_many_steps` or `missing_finish`.

## Execution and references

Tool steps execute in order. A string argument equal to `$<id>` is replaced by that step's recorded
output; any other string, including `"$s1 apples"`, is passed literally. Substitution walks nested
objects and arrays and leaves scalars untouched. Outputs are truncated to 4000 characters before
being recorded and before appearing in prompts.

`finish` ends the run with status `done`; its `answer` may also be a whole-string reference. The run
fails with `reason="step_budget_exhausted"` once more than `max_steps` tool steps (default 20) have
executed across all plans.

## Replanning

One LLM call produces one plan. When a step raises `ToolFailure`, the message (at most 200
characters) is shown to the planner; any other exception is reported only as `tool_error`, without
its message. The assistant reply is appended, followed by a user message:

```
Step <id> failed: <reason>. Completed: <ids or none>. Return a new plan for the remaining work.
```

Invalid plans use `Plan rejected: <PlanError code>.` followed by the same completed-work sentence.
Completed steps appear as `<result id="<id>">escaped output</result>` lines and are never
re-executed. Each replan is one more LLM call and increments `replans`; after `max_replans` the run
fails with `reason="replans_exhausted"`. LLM exceptions propagate unchanged.

## Events

`on_event` receives content-free payloads: `{"event": "step", "id": ..., "tool": ..., "ok": ...}`
and `{"event": "replan", "reason": "tool_failure" | "tool_error" | "invalid_plan"}`. They never
contain arguments, outputs or the goal.

## Public API

`run`, `parse_plan`, the dataclasses `Plan`, `Step`, `StepRecord`, `RunResult`, the errors `PlanError`
and `ToolFailure`, and the port helpers `LLMPort`, `Message`, `RecordedLLM`, `FakeLLM`,
`RecordingExhaustedError`.
