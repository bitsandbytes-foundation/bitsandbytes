# Linting Guide

This project enforces linting and formatting via CI on every pull request. The Lint workflow runs `pre-commit run --all-files`, meaning **all files** in the repo are checked, not just the ones you changed. Your PR will be blocked if any check fails.

## Quick Reference

Before committing and pushing, run the full pre-commit suite:

```bash
pre-commit run --all-files
```

This runs all 10 hooks (ruff, ruff format, typos, clang-format, trailing-whitespace,
and others). Do **not** run only `ruff check` and `ruff format` — those are just 2 of
the 10 hooks. CI runs the full suite and will reject PRs that fail any hook.

If any hook makes changes, **stage and commit those changes** before pushing.

## What CI Checks

The Lint workflow (`.github/workflows/lint.yml`) runs all hooks defined in `.pre-commit-config.yaml`:

| Hook | What it does |
|---|---|
| **ruff** (linter) | Checks for pyflakes, pycodestyle, isort, bugbear, implicit string concat, pyupgrade, and ruff-specific rules |
| **ruff format** | Enforces consistent code formatting (line wrapping, spacing, trailing commas, etc.) |
| **check-merge-conflict** | Ensures no merge conflict markers are left in files |
| **check-yaml** | Validates YAML file syntax |
| **end-of-file-fixer** | Ensures files end with a single newline |
| **fix-byte-order-marker** | Removes UTF-8 BOM |
| **trailing-whitespace** | Removes trailing whitespace from lines |
| **mixed-line-ending** | Enforces LF line endings (except `.bat` files) |
| **typos** | Spell-checks code and documentation |
| **clang-format** | Formats C/C++/CUDA files under `csrc/` |

## Ruff Configuration

Configuration lives in `pyproject.toml` under `[tool.ruff]`. Key settings:

- **Line length**: 119 characters
- **Target Python version**: 3.10
- **Pinned version**: `~0.14.3` (see `pyproject.toml` `[project.optional-dependencies]`)

### Enabled lint rule sets

| Code | Rules |
|---|---|
| `B` | flake8-bugbear (security / correctness warnings) |
| `E` | pycodestyle errors |
| `W` | pycodestyle warnings |
| `F` | pyflakes |
| `I` | isort (import ordering) |
| `ISC` | implicit string concatenation |
| `UP` | pyupgrade (modern Python syntax) |
| `RUF` | ruff-specific rules |

### Notable ignored rules

- `E501` — line-too-long is not enforced by the linter (but `ruff format` still wraps lines as it sees fit)
- `E731` — lambda assignments are allowed
- `B905` — `zip()` without `strict=` is allowed
- Full list in `pyproject.toml` under `[tool.ruff.lint] ignore`

### Per-file relaxations

- `__init__.py` files: unused imports (`F401`) are allowed
- `tests/**` and `benchmarking/**`: several additional rules are relaxed (B007, B011, B023, E701, E731, F841, UP030)
- `bitsandbytes/**/triton/**`: import order (`I001`) is relaxed

## Common Agent Mistakes

### 1. Not running `ruff format`

The most frequent failure. `ruff check` (the linter) and `ruff format` (the formatter) are **separate tools**. You must run both. The formatter rewraps lines, adjusts trailing commas, and normalizes spacing in ways the linter does not check.

Example: a long `assert` with an f-string message that looks fine to the linter will be reformatted by `ruff format`:

```python
# Before (fails ruff format):
assert err < threshold, f"Error {err:.6f} exceeds {threshold:.6f} + {N}*{std:.6f}"

# After (ruff format wraps it):
assert err < threshold, (
    f"Error {err:.6f} exceeds {threshold:.6f} + {N}*{std:.6f}"
)
```

### 2. Only checking changed files

CI runs `pre-commit run --all-files`. If there is a pre-existing formatting issue anywhere in the repo, your PR will fail even if your changes are clean. Always run the checks on the entire repo, not just your changed files.

### 3. Forgetting C/CUDA formatting

If you modify files under `csrc/`, `clang-format` will run. Make sure to format C/C++/CUDA code as well:

```bash
# If you have clang-format installed:
clang-format -i csrc/your_file.cu

# Or just run pre-commit which handles it:
pre-commit run --all-files
```

### 4. Typos in variable names or comments

The `typos` checker scans all text. If it flags a false positive (e.g., a domain-specific abbreviation), you can add an exception to a `[default.extend-words]` section in a `_typos.toml` or `typos.toml` config file — but check with a maintainer first.

## Recommended Workflow

1. Make your code changes
2. Run `pre-commit run --all-files` to run all lint and formatting hooks
3. Review the changes the hooks made (especially ruff `--fix` auto-corrections)
4. Stage everything and commit
5. Run `pre-commit run --all-files` again to confirm everything passes
