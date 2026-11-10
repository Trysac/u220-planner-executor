# Security

`planner_executor` ships no runtime dependencies, no network code and no credential handling. The
attacker-controlled input is the planner's reply, so the package treats it defensively:

- replies are size-limited and JSON depth is bounded, with `RecursionError` converted to
  `PlanError("invalid_plan")`;
- plans are fully validated before any tool runs, so unknown tools and bad references never reach
  execution;
- tool outputs are truncated to 4000 characters before being stored or placed in prompts;
- `ToolFailure` messages are capped at 200 characters, while any other exception is reported as
  `tool_error` without its message, so tool internals never leak to the planner;
- `on_event` payloads carry only event kind, step id, tool name and ok/reason flags, never
  arguments, outputs or the goal.

Report vulnerabilities through the repository's security policy (`SECURITY.md` at the root).
