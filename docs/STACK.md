# Stack

- Runtime: Python 3.10+ standard library only; no runtime dependencies.
- Tests: pytest with pytest-cov, branch coverage floor of 80 %.
- Types: mypy in strict mode over `src`.
- Lint and format: ruff.
- Security: bandit, pip-audit, gitleaks and osv-scanner.
- Build: hatchling.

Discarded alternatives: agent frameworks were rejected because the pattern fits in one small,
fully typed module and frameworks bring dependencies plus unbounded loops; third-party JSON
validators were rejected because `json` plus explicit checks keeps the validation codes stable.
