# planner-executor

[![PyPI version](https://img.shields.io/pypi/v/trysac-planner-executor)](https://pypi.org/project/trysac-planner-executor/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-planner-executor)](LICENSE)
[![CI](https://github.com/Trysac/u220-planner-executor/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u220-planner-executor/actions/workflows/ci.yml)

Planner-executor agent pattern over a text-only LLM port: a validated JSON plan of tool steps, step references, bounded replanning after failures and content-free events

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-planner-executor
# or
uv add trysac-planner-executor
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import planner_executor

# normalize() trims surrounding whitespace and lower-cases the text.
print(planner_executor.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    planner_executor.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
