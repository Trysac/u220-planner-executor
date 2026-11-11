# planner-executor

[![PyPI version](https://img.shields.io/pypi/v/trysac-planner-executor)](https://pypi.org/project/trysac-planner-executor/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-planner-executor)](LICENSE)
[![CI](https://github.com/Trysac/u220-planner-executor/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u220-planner-executor/actions/workflows/ci.yml)

Planner-executor agent pattern over a text-only LLM port: a validated JSON plan of tool steps, step references, bounded replanning after failures and content-free events

## Problem

Agents that decide one tool call at a time drift, repeat themselves and spend tokens re-reading the
whole history, while plans written once break on the first failing step. `planner-executor` keeps the
pattern small: the model writes a validated JSON plan, code executes it deterministically, and only a
failure triggers a new plan, with hard bounds on replans and total steps.

## Install

```sh
pip install trysac-planner-executor
# or
uv add trysac-planner-executor
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
from planner_executor import PlanError, RecordedLLM, parse_plan, run

plan = (
    '{"steps": [{"id": "s1", "tool": "add", "args": {"a": 2, "b": 3}},'
    ' {"id": "s2", "tool": "finish", "args": {"answer": "$s1"}}]}'
)

result = run(
    "add 2 and 3",
    RecordedLLM([plan]),
    {"add": lambda args: str(args["a"] + args["b"])},
)
print(result.status, result.answer)  # done 5
print(result.replans, result.reason)  # 0 None

# Invalid plans raise PlanError with a machine-readable code.
try:
    parse_plan('{"steps": []}')
except PlanError as exc:
    print(exc.code)  # missing_finish
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
