# 0001. Standard library only and bounded replanning

## Status

Accepted.

## Context

Agent frameworks typically pull in a dependency tree, decide one tool call at a time, and can loop
without a hard bound. This library targets the planner-executor pattern, where one LLM call yields a
whole plan and code executes it.

## Decision

- Implement everything with the Python standard library and no runtime dependencies.
- Validate each plan before execution and reject it with a `PlanError` code instead of repairing it
  silently.
- Bound the run: at most `max_steps` tool steps across all plans and at most `max_replans` extra LLM
  calls. Exceeding either ends with a `failed` result and an explicit reason.
- Expose the model through the `LLMPort` protocol so callers bring their own client and tests use
  `RecordedLLM` or `FakeLLM`.

## Consequences

The package stays small, auditable and deterministic. Retries, memory across runs, parallelism and
provider-specific clients are out of scope and belong to callers.
