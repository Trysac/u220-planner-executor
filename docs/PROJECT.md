# Project record

- Name: `trysac-planner-executor`, import name `planner_executor`.
- Category: Agents and MCP.
- Distribution: public open source, MIT licensed.
- Python: 3.10 or newer, standard library only, fully typed (`py.typed`).
- Status: 0.1.0.

The library implements the planner-executor agent pattern: a validated JSON plan of tool steps, step
references, bounded replanning after failures and content-free events. It deliberately excludes
parallel step execution, memory across runs and real LLM clients; callers supply any object that
satisfies `LLMPort`.

Quality gates: ruff lint and format, mypy strict, pytest with branch coverage of at least 80 %, the
documentation manifest (`docs-manifest.json`) and gitleaks.

## Factory run u220 (measured from API usage)

| Date (UTC) | Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|---|
| 2026-11-10 | generation | deepseek-v4-pro | yes | 1763 | 23685 | 0 | 3328 | 0.0481 |
| 2026-11-10 | fix-1 | deepseek-flash | no | 16327 | 11303 | 0 | 148480 | 0.0097 |
| 2026-11-10 | fix-2 | deepseek-flash | no | 22973 | 6985 | 0 | 174848 | 0.0082 |
| 2026-11-10 | audit | deepseek-flash | no | 30497 | 20613 | 0 | 374656 | 0.0181 |
| 2026-11-10 | fix-3 | deepseek-flash | no | 22590 | 9674 | 0 | 314112 | 0.0101 |
| 2026-11-10 | docs | deepseek-flash | no | 95486 | 16000 | 0 | 256 | 0.0239 |

Total: $0.1181. Prices: deepseek line, as measured by its adapter. Providers serve the current version of each model id, so the date tells which version answered; versions, launch stages and price sources per line are in the factory's model register (docs/MODELS.md).
