# Use planner-executor pattern with strict plan validation

Status: accepted (generated, pending human review)

## Context

The utility must implement a bounded, deterministic agent that avoids per-step LLM calls and drift. The specification requires JSON plans, references, replanning on failure, and limits.

## Decision

Implement a pure-Python library with no runtime dependencies. The plan is parsed into frozen dataclasses with rigorous validation (structure, tool existence, references, finish constraints). The run loop executes plans sequentially, maintaining completed steps, and replans by appending failure prompts. All limits (reply length, JSON depth, step count, replans, output truncation) are enforced.

## Consequences

The library is simple and auditable, with clear bounds. It does not support parallel execution or memory across runs, as out of scope. All errors are typed and messages are sanitized to avoid leaking sensitive data.
