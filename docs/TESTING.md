# Testing

Tests live in `tests/` and exercise the public API only.

```sh
uv run pytest --cov=planner_executor
```

The gate requires branch coverage of at least 80 %. The suite uses `RecordedLLM` for fixed reply
sequences and `FakeLLM` when a test needs to inspect the messages sent to the planner. No test needs
network access or an API key.

Covered behaviour includes plan validation and every `PlanError` code, whole-string reference
substitution, truncation at 4000 characters, the step budget, replan exhaustion, `ToolFailure`
versus generic tool exceptions (and that a generic exception's message never leaks into prompts),
and the content-free `on_event` payloads.
