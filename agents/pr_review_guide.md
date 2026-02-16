# Pull Request Review Guide

This document defines the complete workflow for reviewing pull requests to bitsandbytes.
It is written for agent-reviewers who will analyze PRs autonomously, but it applies equally
to human reviewers. The guide is procedural: it tells you what to do, in what order, and
what to check at each step.

This guide does **not** duplicate reference material from other agent documents. Instead,
it tells you when and how to consult them. You must read the prerequisite documents before
performing your first review.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Review Workflow Overview](#2-review-workflow-overview)
3. [Step 1: Fetch PR Metadata](#3-step-1-fetch-pr-metadata)
4. [Step 2: Classify the PR](#4-step-2-classify-the-pr)
5. [Step 3: Check CI Status](#5-step-3-check-ci-status)
6. [Step 4: Read the Linked Issue](#6-step-4-read-the-linked-issue)
7. [Step 5: Read All Changed Files](#7-step-5-read-all-changed-files)
8. [Step 6: Classification-Specific Deep Review](#8-step-6-classification-specific-deep-review)
9. [Step 7: Downstream Impact Assessment](#9-step-7-downstream-impact-assessment)
10. [Step 8: Cross-PR Conflict Check](#10-step-8-cross-pr-conflict-check)
11. [Step 9: Test Assessment](#11-step-9-test-assessment)
12. [Step 10: Performance Impact Assessment](#12-step-10-performance-impact-assessment)
13. [Step 11: torch.compile Compatibility](#13-step-11-torchcompile-compatibility)
14. [Step 12: Checkpoint and Serialization Backward Compatibility](#14-step-12-checkpoint-and-serialization-backward-compatibility)
15. [Step 13: Platform-Specific Review](#15-step-13-platform-specific-review)
16. [Step 14: Commit Hygiene](#16-step-14-commit-hygiene)
17. [Step 15: Produce and Post the Review](#17-step-15-produce-and-post-the-review)
18. [Merge Readiness Checklist](#18-merge-readiness-checklist)
19. [Common Review Pitfalls](#19-common-review-pitfalls)
20. [Reference: File-to-Concern Mapping](#20-reference-file-to-concern-mapping)
21. [Reference: API Change Impact Quick-Lookup](#21-reference-api-change-impact-quick-lookup)
22. [Reference: Review Depth by Classification](#22-reference-review-depth-by-classification)

---

## 1. Prerequisites

Before performing any PR review, you must have read and internalized the following documents.
Each one provides reference knowledge that this guide will tell you to consult at specific
steps. Do not skip any of them.

| Document | What it provides | When you need it |
|---|---|---|
| `agents/architecture_guide.md` | Full codebase architecture: layer stack, module organization, backend dispatch, CUDA kernel structure, build system | Understanding what code does, where things belong, whether changes follow existing patterns |
| `agents/code_standards.md` | Naming conventions, error handling patterns, test patterns, docstring style, type annotation expectations, backend registration patterns | Evaluating code quality, spotting pattern violations, assessing whether code matches project style |
| `agents/api_surface.md` | Complete catalog of every public API: classes, functions, parameters, return types, module-level attributes | Detecting API changes, verifying backward compatibility, checking if new code matches existing signatures |
| `agents/downstream_integrations.md` | How Transformers, PEFT, Accelerate, TGI, and vLLM use bitsandbytes: exact API calls, attribute access, isinstance checks, serialization formats, breaking-change risk tables | Assessing downstream impact of any change that touches public APIs, parameter classes, or serialization |
| `agents/kbit_gemm_context.md` | Design context for kbit quantization and GEMM kernels: bit-plane format, codebook design, E4M4 absmax, CUDA kernel architecture | Reviewing CUDA kernel changes, quantization changes, or anything touching the kbit subsystem |
| `agents/linting_guide.md` | Pre-commit hooks, ruff configuration, clang-format for C/CUDA, common agent mistakes | Verifying the PR will pass CI lint checks |
| `agents/testing_guide.md` | Test suite characteristics, parallelization, known architecture-specific failures, build prerequisites | Assessing test adequacy, understanding test failures |
| `agents/security_guide.md` | Trust model for contributors, supply chain risk assessment, security review checklist for external PRs, dependency vetting | Evaluating external contributions, assessing new dependencies, reviewing build system changes that affect the supply chain |

You do not need to re-read these documents for every review. But you must have read them at
least once, and you must consult the relevant ones during each review as directed by the
steps below.

---

## 2. Review Workflow Overview

Every PR review follows this sequence. Steps are ordered by dependency: earlier steps
inform decisions in later steps. Most steps may conclude quickly ("not applicable")
depending on the PR classification. Trivial PRs (docs, style, test-only) may skip
Steps 6-14 entirely — see Section 4.2 for the early termination criteria.

```
Step 1: Fetch PR Metadata
    |
Step 2: Classify the PR
    |
    +-- Trivial PR? (Section 4.2) ---> Step 3 -> 4 -> 5 -> skip to Step 15
    |
Step 3: Check CI Status
    |
Step 4: Read the Linked Issue (if any)
    |
Step 5: Read All Changed Files
    |
Step 6: Classification-Specific Deep Review
    |
Step 7: Downstream Impact Assessment
    |
Step 8: Cross-PR Conflict Check
    |
Step 9: Test Assessment
    |
Step 10: Performance Impact Assessment
    |
Step 11: torch.compile Compatibility
    |
Step 12: Checkpoint/Serialization Backward Compatibility
    |
Step 13: Platform-Specific Review
    |
Step 14: Commit Hygiene
    |
Step 15: Produce and Post the Review
```

After posting the review, consult the Merge Readiness Checklist (Section 18) if the
verdict is "Approve" or "Approve with minor changes."

---

## 3. Step 1: Fetch PR Metadata

Before reading any code, gather the PR's metadata. This gives you the full picture before
you invest time reading files.

### 3.1 Required Information

Fetch all of the following:

```bash
# Basic PR info
gh pr view <NUMBER> --json title,body,author,labels,state,headRefName,baseRefName,additions,deletions,changedFiles,commits,reviews,comments,mergeStateStatus

# Changed files list
gh pr diff <NUMBER> --stat

# Full diff
gh pr diff <NUMBER>

# CI check status
gh pr checks <NUMBER>

# Comments and review threads
gh pr view <NUMBER> --comments
```

### 3.2 What to Record

From the metadata, note:

- **Title and description**: What does the PR claim to do?
- **Author**: Is this a maintainer, a known contributor, or a first-time contributor?
- **Size**: Lines added/deleted, number of files changed. This calibrates review depth.
- **Branch name**: Often encodes intent (e.g., `fix/issue-1234`, `feature/kbit-quantization`).
- **Labels**: May indicate category (CI/CD, Windows, etc.).
- **Linked issues**: Look for "Fixes #NNN" or "Closes #NNN" in the body.
- **Number of commits**: Single-commit PRs are simpler; multi-commit PRs may contain
  unrelated changes.
- **Existing reviews and comments**: Has anyone already reviewed? Are there unresolved
  threads?

### 3.3 Size Calibration

Use the PR size to calibrate your review depth:

| Size | Lines changed | Expected review depth |
|---|---|---|
| Trivial | < 20 lines, 1-2 files | Quick scan, verify correctness |
| Small | 20-100 lines, 1-4 files | Careful line-by-line review |
| Medium | 100-500 lines, 3-10 files | Full review with all checklists |
| Large | 500-2000 lines, 5-20 files | Full review, may need multiple passes |
| Very large | > 2000 lines | Consider whether the PR should be split |

Very large PRs (> 2000 lines) are a yellow flag. Unless the PR is a new feature with
mostly new files (which is acceptable), suggest splitting it into smaller, independently
reviewable pieces.

---

## 4. Step 2: Classify the PR

Every PR falls into one or more of the following categories. Classification determines
which checklists apply and how deep the review needs to go.

### 4.1 Classification Decision Tree

Read the PR title, description, and changed files list. Then classify:

```
Is every changed file under docs/ or *.md?
  YES -> DOCUMENTATION
  NO  -> continue

Is every changed file under .github/ or build/CI config?
  YES -> BUILD/CI
  NO  -> continue

Does the PR add a new module, class, or major function?
  YES -> NEW FEATURE
  NO  -> continue

Does the PR remove existing APIs, classes, or modules?
  YES -> DEPRECATION/REMOVAL
  NO  -> continue

Does the PR restructure code without changing behavior?
  YES -> REFACTORING
  NO  -> continue

Does the PR only change test files?
  YES -> TEST CHANGE
  NO  -> continue

Does the PR fix a bug (linked to an issue, title says "fix")?
  YES -> BUG FIX
  NO  -> continue

Does the PR touch CMakeLists.txt, setup.py, pyproject.toml, or csrc/ build files?
  YES -> BUILD SYSTEM
  NO  -> GENERAL CHANGE (apply all relevant checklists)
```

A PR may have multiple classifications. For example, a bug fix that also adds tests and
updates documentation is BUG FIX + TEST CHANGE + DOCUMENTATION. Apply all relevant
checklists.

### 4.2 Early Termination for Trivial PRs

After classification, determine whether the PR qualifies for an abbreviated review.
If **all** of the following are true, skip Steps 6-14 and go directly to Step 15:

- Classification is solely `[docs]`, `[style]`, or `[test]` (no code changes)
- Total lines changed < 50
- No changes to `pyproject.toml`, `CMakeLists.txt`, `CLAUDE.md`, or any file in
  `agents/`, `.github/`, or `csrc/`
- The diff contains no suspicious patterns (run the pre-review automated scans from
  `agents/security_guide.md` Section 17.1 on all changed files regardless)

For these PRs, Steps 3 (CI status), 4 (linked issue), and 5 (read changed files) are
still required. But you do not need to assess downstream impact, torch.compile
compatibility, serialization, performance, platform concerns, or cross-PR conflicts.

If any of the conditions above are not met, follow the full 15-step process. When in
doubt, do the full review.

### 4.3 Classification Tags

Record the classification(s) in your review output. Use these tags:

- `[bug-fix]` — Fixes a reported or discovered bug
- `[feature]` — Adds new functionality
- `[deprecation]` — Removes or deprecates existing functionality
- `[refactor]` — Restructures code without changing behavior
- `[docs]` — Documentation changes only
- `[test]` — Test changes only
- `[build]` — Build system, CI, or infrastructure changes
- `[platform]` — Platform-specific changes (Windows, ROCm, MPS, etc.)
- `[performance]` — Performance optimization
- `[style]` — Formatting, linting, or cosmetic changes only

---

## 5. Step 3: Check CI Status

### 5.1 Interpret CI Results

```bash
gh pr checks <NUMBER>
```

The CI matrix runs:

- **Lint**: `pre-commit run --all-files` (ruff, ruff format, typos, clang-format, etc.)
- **CPU build**: Builds the native library on multiple platforms and PyTorch versions
- **CPU tests**: Runs the test suite without GPU (limited coverage)
- **GPU tests**: Runs the full test suite on CUDA hardware (not always available for
  external PRs)

### 5.2 CI Status Decision Table

| CI Status | Action |
|---|---|
| All checks pass | Proceed with review |
| Lint fails | Note in review. PR cannot merge until lint passes. Check if the failure is in the PR's code or pre-existing. |
| Build fails | Note in review. Read the build log to determine if the failure is caused by the PR or is a pre-existing/infrastructure issue. |
| Tests fail | Read the failure log. Determine: (a) is the failure caused by the PR, (b) is it a known architecture-specific failure (see `testing_guide.md` Known Issues), or (c) is it a flaky test? |
| CI not triggered | Common for external contributor PRs from forks. Note this in your review — CI must run before merge. A maintainer may need to approve the workflow run. |
| Some checks pass, some pending | Wait for completion if possible. If checks have been pending for an unreasonable period, proceed with review but note the incomplete CI. |

### 5.3 Pre-existing CI Failures

Some test failures are known and pre-existing on certain architectures. Consult
`agents/testing_guide.md` Section "Known Issues by Architecture" to identify these.
A PR should not be blocked by failures that exist on the base branch.

To verify whether a failure is pre-existing:

```bash
# Check if the same test fails on main
gh run list --branch main --limit 5 --json name,status,conclusion
```

### 5.4 Missing CI for Fork PRs

When a PR comes from a fork, GitHub Actions require maintainer approval before running.
This is normal. Note it in your review:

> CI has not run for this PR. A maintainer needs to approve the workflow run before merge.

Do not block your review on missing CI — complete the code review and note CI as a
pre-merge requirement.

---

## 6. Step 4: Read the Linked Issue

### 6.1 Find the Issue

Look for issue references in:
- The PR body ("Fixes #NNN", "Closes #NNN", "Resolves #NNN")
- The PR title (e.g., "Fix: ... (#NNN)")
- The branch name (e.g., `fix/issue-1234`)
- Commit messages

If there is no linked issue, that is acceptable for:
- Documentation PRs
- Style/lint PRs
- CI/build improvements
- Small refactors

For bug fixes and features, a missing linked issue is a yellow flag. Note it in your
review: "This PR has no linked issue. Consider creating one for tracking."

### 6.2 Read the Issue

```bash
gh issue view <NUMBER> --json title,body,comments,labels,state
```

When reading the issue, determine:

1. **What is the reported problem?** Understand the user's actual experience, not just
   the title.

2. **Is there a reproducer?** A minimal script or steps to reproduce the bug. If yes,
   verify the PR's test covers the same scenario.

3. **What is the root cause?** The issue discussion may contain diagnosis. Compare this
   with what the PR actually fixes — sometimes a PR fixes a symptom rather than the root
   cause.

4. **Are there constraints mentioned?** The issue may specify that a fix must be backward
   compatible, must work on a specific platform, must not change the API, etc.

5. **Are there other proposed solutions?** The issue discussion may contain alternative
   approaches. If the PR uses a different approach, note whether it was discussed and
   agreed upon.

### 6.3 Issue-PR Alignment Check

After reading both the issue and the PR, verify:

- [ ] The PR addresses the root cause described in the issue, not just a symptom
- [ ] The PR's scope matches the issue's scope (not too narrow, not too broad)
- [ ] Any constraints mentioned in the issue are respected by the PR
- [ ] If the issue has a reproducer, the PR's test covers the same scenario
- [ ] The PR description accurately describes what was changed and why

If the PR claims to fix an issue but doesn't actually address the root cause, this is a
blocking concern.

---

## 7. Step 5: Read All Changed Files

### 7.1 Reading Order

Read the changed files in this order:

1. **Test files first**: Tests tell you what the PR is supposed to do. Read them before
   reading the implementation. This gives you a specification to check the implementation
   against.

2. **Implementation files**: Read the actual code changes.

3. **Configuration files**: `pyproject.toml`, `CMakeLists.txt`, `.github/workflows/*.yml`,
   `.pre-commit-config.yaml`, etc.

4. **Documentation files**: `docs/`, `*.md` files, docstrings.

### 7.2 Reading the Diff

For each changed file, read the full diff. Pay attention to:

- **Context around changes**: The diff shows surrounding lines. Are the changes consistent
  with the surrounding code? Do they follow the same patterns?

- **Deleted code**: What was removed? Is the removal safe? Could anything else depend on
  the deleted code?

- **Added code**: Does it follow existing patterns? Does it handle error cases? Does it
  have appropriate comments for non-obvious logic?

- **Moved code**: Sometimes code is moved between files. Verify nothing was lost or
  subtly changed during the move.

### 7.3 Understanding the Full File

For non-trivial changes, do not rely solely on the diff. Read the full file to understand:

- Where the changed code fits in the file's structure
- Whether the change is consistent with the rest of the file
- Whether there are related functions or classes that should also be updated
- Whether the change introduces duplication with existing code

```bash
# Read the full file, not just the diff
gh pr diff <NUMBER> --name-only  # get list of changed files
# Then read each file in the repo
```

### 7.4 What to Look For (General)

These checks apply to every PR regardless of classification:

**Correctness:**
- Does the code do what the PR description says it does?
- Are there off-by-one errors, wrong variable names, or logic inversions?
- Are edge cases handled (empty inputs, None values, zero-length tensors)?
- Are error messages accurate and helpful?

**Style and patterns (consult `code_standards.md`):**
- Does the code follow the naming conventions in `code_standards.md`?
- Does it use the same error handling patterns as surrounding code?
- Are imports organized correctly (stdlib, third-party, local)?
- Is the code appropriately commented? (Not over-commented, not under-commented)

**Safety:**
- No hardcoded file paths, credentials, or secrets
- No unbounded memory allocation
- No infinite loops or recursion without bounds
- CUDA code: proper error checking, no out-of-bounds memory access
- No use of `eval()`, `exec()`, or `pickle.loads()` on untrusted input

---

## 8. Step 6: Classification-Specific Deep Review

Based on the classification from Step 2, apply the relevant subsections below. If a PR
has multiple classifications, apply all relevant subsections.

### 8.1 Bug Fixes

Bug fix PRs are the most common type. They require careful analysis of whether the fix is
correct and complete.

#### 8.1.1 Root Cause Analysis

- [ ] **Identify the root cause.** Read the issue (Step 4) and the code change. Can you
  explain, in one sentence, what was wrong and why? If you can't, the fix may be
  incomplete or addressing a symptom.

- [ ] **Verify the fix targets the root cause.** A common mistake is fixing the symptom
  (e.g., catching an exception) rather than the cause (e.g., the data that triggered the
  exception). If the fix adds a try/except, ask: why does the exception occur? Should it
  be prevented instead of caught?

- [ ] **Check for related code paths.** If the bug was in function A, are there similar
  functions B and C that have the same bug? The fix should address all instances, not just
  the one that was reported.

#### 8.1.2 Regression Risk

- [ ] **Could the fix break existing behavior?** For example, if the fix changes a
  default value, what happens to code that relied on the old default?

- [ ] **Does the fix change the function's contract?** If a function previously accepted
  a certain input and now rejects it (or vice versa), that's a behavior change, not just
  a bug fix.

- [ ] **Is the fix backward compatible?** Users may have workarounds for the bug. Does
  the fix invalidate those workarounds in a harmful way?

#### 8.1.3 Test Coverage

- [ ] **Does the PR include a test that reproduces the bug?** A bug fix without a
  regression test is incomplete. The test should fail without the fix and pass with it.

- [ ] **Does the test cover the exact scenario from the issue?** If the issue has a
  reproducer, the test should be equivalent to that reproducer.

- [ ] **Are edge cases tested?** The bug may have been triggered by a specific input. Are
  related edge cases (boundary values, different dtypes, different devices) also tested?

### 8.2 New Features

New feature PRs add functionality that didn't exist before. They require the broadest
review because they affect the API surface, may have downstream implications, and set
patterns that future code will follow.

#### 8.2.1 Design Assessment

- [ ] **Is this the right approach?** Consider whether the feature could be implemented
  more simply, or whether it duplicates existing functionality.

- [ ] **Does it follow existing patterns?** Consult `architecture_guide.md` for the
  codebase's layering (functional.py → _ops.py → backends → C/CUDA). New features should
  follow the same layer structure.

- [ ] **Is the API surface appropriate?** Consult `api_surface.md`. Does the new API
  follow the naming and parameter conventions of existing APIs? Is it at the right
  abstraction level?

- [ ] **Is the scope appropriate?** Does the PR implement exactly what's needed, or does
  it over-engineer with unnecessary configuration, abstraction layers, or speculative
  future-proofing?

#### 8.2.2 API Design

- [ ] **Parameter names and defaults.** Do they follow existing conventions? Are defaults
  sensible?

- [ ] **Return types.** Are they consistent with similar functions?

- [ ] **Error handling.** What happens with invalid inputs? Are error messages clear?

- [ ] **Documentation.** New public APIs need docstrings. Check that they explain what the
  function does, what each parameter means, and what it returns.

#### 8.2.3 Backend Registration

If the feature adds a new op or modifies an existing one:

- [ ] **`_ops.py` registration.** Is the op registered with `torch.library`? Does it have
  a fake tensor implementation for `torch.compile`?

- [ ] **Backend dispatch.** Does the CUDA backend implement the op? What about the CPU
  backend? If the op is CUDA-only, does the CPU path raise a clear error?

- [ ] **C/CUDA interface.** Does `csrc/pythonInterface.cpp` have the correct extern "C"
  wrapper? Does it match the Python binding?

Consult `architecture_guide.md` Sections on the op registration pipeline and backend
dispatch for the expected patterns.

#### 8.2.4 CUDA Kernel Review (if applicable)

If the feature includes new CUDA kernels, perform a thorough kernel review:

- [ ] **Launch configuration.** Are grid and block dimensions correct? Are they bounded
  for large inputs?

- [ ] **Memory access patterns.** Are global memory accesses coalesced? Are shared memory
  accesses free of bank conflicts?

- [ ] **Boundary handling.** What happens when the input size is not a multiple of the
  block size? Are there proper bounds checks?

- [ ] **Numeric precision.** Is the accumulation dtype appropriate? Are there potential
  overflow or underflow issues?

- [ ] **Error handling.** Does the kernel check for CUDA errors after launch? Are
  assertions and bounds checks present in debug builds?

- [ ] **Template instantiation.** Are all necessary template variants instantiated? The
  common pattern is dtype (fp16, bf16, fp32) x feature-specific parameters.

Consult `kbit_gemm_context.md` for reference on the project's CUDA kernel patterns,
including the warp-level programming style, bit-plane format, and E4M4 absmax handling.

#### 8.2.5 Test Coverage for Features

- [ ] **Happy path tests.** Do tests cover the primary use case?

- [ ] **Edge cases.** Empty inputs, single-element inputs, maximum-size inputs, boundary
  values for parameters.

- [ ] **Dtype coverage.** Tests should cover at least fp16, bf16, and fp32 where
  applicable.

- [ ] **Device coverage.** Tests should cover CUDA (and CPU if the feature supports it).

- [ ] **Error path tests.** Do tests verify that invalid inputs produce clear error
  messages?

- [ ] **Round-trip tests.** For quantization features: quantize → dequantize should
  produce results within expected error bounds.

### 8.3 Deprecation and Removal

Deprecation PRs remove or deprecate existing functionality. They are high-risk because
they directly break downstream consumers.

#### 8.3.1 Removal Safety

- [ ] **Is the removed API still used by downstream projects?** Consult
  `downstream_integrations.md` Section 6 (Consolidated API Surface) and the per-project
  sections. Cross-reference every removed class, function, parameter, and attribute
  against the downstream usage tables.

- [ ] **Was the API previously deprecated with a warning?** Best practice is to deprecate
  first (with a `DeprecationWarning`), then remove in a later release. If the PR removes
  without prior deprecation, this is a concern.

- [ ] **Is there a migration path?** Users of the removed API should have a clear
  alternative. The PR description or deprecation warning should explain what to use
  instead.

- [ ] **Does the removal affect the serialization format?** If removed code was involved
  in state dict serialization or deserialization, removing it could break existing
  checkpoints. This is a critical concern.

#### 8.3.2 Scope Verification

- [ ] **Are all references removed?** If a function is deleted, are all call sites also
  updated? Search for the function name across the entire codebase.

- [ ] **Are tests updated?** Tests for removed functionality should also be removed or
  updated. Leftover tests that reference deleted code will fail.

- [ ] **Are imports cleaned up?** Removed modules should be removed from `__init__.py`
  exports.

- [ ] **Is documentation updated?** References to removed APIs in docs, docstrings, and
  comments should be cleaned up.

### 8.4 Refactoring

Refactoring PRs restructure code without changing behavior. The key risk is that the
restructuring inadvertently changes behavior.

#### 8.4.1 Behavior Preservation

- [ ] **Does the refactored code produce identical output for identical input?** For
  numerical code, this means bit-identical results. For non-numerical code, it means
  the same observable behavior.

- [ ] **Are all callers updated?** If a function's signature changes, all call sites must
  be updated.

- [ ] **Is the public API preserved?** Refactoring should not change the public API
  unless that's explicitly part of the PR's goal. Check `api_surface.md` for what's
  public.

#### 8.4.2 Justification

- [ ] **Is the refactoring motivated?** The PR should explain why the restructuring is
  needed. "Cleaner code" is weak justification; "enables X feature" or "fixes Y
  maintenance problem" is strong justification.

- [ ] **Is the scope appropriate?** Refactoring PRs that touch many files are hard to
  review and risky. If the PR touches more than ~10 files, consider whether it should
  be split.

### 8.5 Documentation

Documentation PRs change docs, docstrings, comments, or markdown files.

#### 8.5.1 Accuracy

- [ ] **Are code examples correct?** Run them mentally (or actually run them) to verify
  they work. Check that:
  - Import paths are correct
  - Function names match the actual API (consult `api_surface.md`)
  - Parameter names and types are correct
  - The example produces the described output

- [ ] **Are API references current?** If the docs reference specific functions, classes,
  or parameters, verify they still exist and have the described behavior.

- [ ] **Are version-specific claims correct?** If the docs say "available since v0.43.0"
  or "requires PyTorch >= 2.0", verify these claims.

#### 8.5.2 Completeness

- [ ] **Does the documentation cover the right scope?** Not too narrow (missing important
  details) and not too broad (including irrelevant information).

- [ ] **Are prerequisites stated?** If the documented feature requires specific hardware,
  software versions, or configuration, are these stated?

#### 8.5.3 Style

- [ ] **Consistent with existing docs.** Check the tone, formatting, and structure of
  nearby documentation. New docs should match.

- [ ] **No stale references.** If the docs reference other files or URLs, verify they
  exist and are current.

### 8.6 Build System and CI

Build system PRs change CMakeLists.txt, pyproject.toml, setup.py, GitHub Actions
workflows, or pre-commit configuration.

#### 8.6.1 Build System Changes

- [ ] **Does the change break any existing build configuration?** CMake changes that work
  for one platform may break another. Check that CUDA, ROCm, CPU, and any platform-specific
  configurations are all still valid.

- [ ] **Are new dependencies justified?** Adding a build dependency increases the
  maintenance burden. Is it necessary?

- [ ] **Is the change backward compatible with supported toolchains?** Check the minimum
  supported CMake version, compiler versions, and CUDA toolkit versions.

- [ ] **Does pyproject.toml maintain correct metadata?** Version constraints, extras,
  entry points, etc.

#### 8.6.2 CI Changes

- [ ] **Do workflow changes maintain the existing test matrix?** Removing a test
  configuration is a significant change that should be explicitly justified.

- [ ] **Are action versions pinned to SHAs?** Using `@v4` is less secure than
  `@abc123def`. If the PR upgrades actions, verify the new SHAs are from the correct
  repositories.

- [ ] **Do new workflow steps have appropriate timeouts?** CI jobs without timeouts can
  run indefinitely and block the queue.

- [ ] **Are secrets handled correctly?** Workflow changes should not expose secrets or
  change who can trigger workflows with access to secrets.

### 8.7 Test Changes

PRs that only change test files (no implementation changes).

#### 8.7.1 Test Quality

- [ ] **Do new tests test the right thing?** A test that always passes regardless of
  the implementation is useless. Verify the test would fail if the implementation had
  the bug or missing feature.

- [ ] **Are assertions specific enough?** Testing `assert result is not None` is rarely
  useful. Tests should check specific values, shapes, dtypes, and error conditions.

- [ ] **Are thresholds justified?** For numerical tests with tolerance thresholds, are
  the thresholds derived from analysis (e.g., quantization error bounds) or just picked
  to make the test pass? Consult `code_standards.md` for the project's approach to
  precision thresholds.

- [ ] **Do tests clean up after themselves?** Tests that allocate GPU memory, create
  temporary files, or modify global state should clean up. Leftover state can cause
  interference with other tests under parallel execution.

#### 8.7.2 Test Infrastructure

- [ ] **Are new test dependencies needed?** If the tests require packages not in the
  existing test dependencies, they must be added to `pyproject.toml`.

- [ ] **Are tests parametrized appropriately?** The bitsandbytes test suite uses
  extensive parametrization. New tests should follow the same pattern unless there's
  a good reason not to.

- [ ] **Will the tests work in CI?** CI may have limited GPU memory, specific CUDA
  versions, or architecture-specific behavior. Tests should not assume a specific GPU
  model.

---

## 9. Step 7: Downstream Impact Assessment

This is one of the most critical steps. Any change to bitsandbytes' public API, parameter
classes, serialization format, or behavioral semantics can break downstream projects that
serve millions of users.

### 9.1 When to Perform This Assessment

Perform the downstream impact assessment if the PR changes ANY of:

- Files in `bitsandbytes/nn/modules.py` (Linear4bit, Linear8bitLt, Params4bit, Int8Params)
- Files in `bitsandbytes/functional.py` (quantize, dequantize, matmul functions)
- Files in `bitsandbytes/_ops.py` (op registrations)
- Files in `bitsandbytes/autograd/_functions.py` (autograd wrappers)
- Files in `bitsandbytes/optim/` (optimizer classes)
- The `__init__.py` exports at any level
- Any class constructor signature
- Any attribute on Params4bit, Int8Params, QuantState, Linear4bit, Linear8bitLt,
  MatmulLtState

If the PR only changes tests, docs, CI, or internal backend code that is not reachable
through any public API, you can skip this step. But be careful: "internal" code that is
accessed by downstream projects via `module.state`, `weight.quant_state`, or similar
attribute access paths is effectively public.

### 9.2 Assessment Procedure

For each changed function, class, method, or attribute:

1. **Look it up in `downstream_integrations.md` Section 6 (Consolidated API Surface).**
   The cross-reference tables show exactly which downstream projects use each API.

2. **For each affected downstream project, read the relevant per-project section**
   (Sections 1-5 of `downstream_integrations.md`). Understand how that project uses
   the changed API.

3. **Classify the risk level:**

   | Change type | Risk | Example |
   |---|---|---|
   | Function removed | CRITICAL | Removing `dequantize_4bit()` |
   | Constructor parameter removed | CRITICAL | Removing `quant_type` from `Linear4bit()` |
   | Constructor parameter renamed | HIGH | `compress_statistics` → `double_quant` |
   | Constructor parameter reordered | HIGH | Positional args in different order |
   | New required constructor parameter | HIGH | Adding `device` as non-optional |
   | Attribute removed or renamed | HIGH | `Params4bit.quant_state` → `Params4bit.qstate` |
   | Return type changed | HIGH | Function returning Tensor now returns tuple |
   | Behavior changed for existing inputs | MEDIUM-HIGH | `quantize_4bit` now normalizes input |
   | New optional parameter with default | LOW | Adding `blocksize=64` with default 64 |
   | New function or class | LOW | Adding `Linear3bit` alongside `Linear4bit` |
   | Bug fix that makes behavior match docs | LOW | Fixing `out` parameter to actually work |
   | Internal implementation change, same API | MINIMAL | Rewriting kernel for speed |

4. **For HIGH or CRITICAL risk, list the specific downstream breakage:**

   ```
   CRITICAL: Removing Params4bit.quant_state attribute
   - Transformers: dequantize_bnb_weight() accesses weight.quant_state -> breaks
   - PEFT: all 4-bit merge/unmerge operations access weight.quant_state -> breaks
   - Accelerate: set_module_tensor_to_device() checks getattr(weight, "quant_state") -> breaks
   - TGI: Linear4bit.forward() accesses self.weight.quant_state -> breaks
   ```

### 9.3 The `__dict__` Round-Trip Rule

PEFT and Accelerate reconstruct Params4bit and Int8Params objects by passing
`old_param.__dict__` to the constructor:

```python
new_param = Params4bit(data, **old_param.__dict__)
```

This means:

- **Any new required constructor parameter that is NOT stored in `__dict__` will break
  this pattern.** The reconstructed object will raise TypeError for the missing kwarg.

- **Any attribute stored in `__dict__` that is not a valid constructor kwarg will break
  this pattern.** The constructor will raise TypeError for an unexpected kwarg.

- **Renaming a constructor parameter breaks this pattern** if the old name is still in
  `__dict__` from serialized objects.

If the PR changes Params4bit or Int8Params constructors, explicitly verify the
`__dict__` round-trip still works:

```python
p = Params4bit(data, requires_grad=False, compress_statistics=True, quant_type="nf4")
p2 = Params4bit(data, **p.__dict__)
# Must not raise. p2 must be functionally equivalent to p.
```

### 9.4 The isinstance and Class Name Rules

Downstream projects detect bitsandbytes types in two ways:

1. **isinstance checks**: `isinstance(module, bnb.nn.Linear4bit)` (used by Transformers,
   PEFT)
2. **String class name checks**: `param.__class__.__name__ == "Params4bit"` (used by
   Accelerate, PEFT's peft_model.py)

This means:

- **Renaming a class breaks all downstream projects** that check for it by name.
- **Moving a class to a different module** may break isinstance checks if the import
  path changes (though re-exporting from the old path mitigates this).
- **Creating a subclass** is generally safe for isinstance checks but may break string
  name checks.

### 9.5 Downstream Impact Report Format

Include in your review verdict:

```
## Downstream Impact

Risk level: [CRITICAL / HIGH / MEDIUM / LOW / NONE]

Affected APIs:
- [list each changed API and its risk]

Affected projects:
- Transformers: [impact description or "not affected"]
- PEFT: [impact description or "not affected"]
- Accelerate: [impact description or "not affected"]
- TGI: [impact description or "not affected"]
- vLLM: [impact description or "not affected"]

Recommendation:
- [specific recommendation, e.g., "safe to merge", "needs migration guide",
  "needs coordinated release with Transformers", "should not merge without
  downstream testing"]
```

---

## 10. Step 8: Cross-PR Conflict Check

### 10.1 Why This Matters

Multiple open PRs may modify the same files or the same logical areas of the codebase.
Merging one PR may create conflicts or semantic incompatibilities with others. The
reviewer should identify these proactively.

### 10.2 Procedure

```bash
# Get the list of files changed by this PR
gh pr diff <NUMBER> --name-only > /tmp/pr_files.txt

# Get all open PRs
gh pr list --state open --json number,title,headRefName --limit 50

# For each other open PR, check file overlap
for other_pr in $(gh pr list --state open --json number -q '.[].number'); do
    if [ "$other_pr" != "<NUMBER>" ]; then
        overlap=$(gh pr diff $other_pr --name-only 2>/dev/null | comm -12 - /tmp/pr_files.txt)
        if [ -n "$overlap" ]; then
            echo "PR #$other_pr overlaps on: $overlap"
        fi
    fi
done
```

### 10.3 Types of Conflicts

**File-level conflicts**: Two PRs modify the same file. Git may be able to merge both
automatically, but the result may not be semantically correct.

**Semantic conflicts**: Two PRs modify different files but interact logically. Examples:
- PR A adds a new function that PR B's removal would delete
- PR A changes a default value that PR B's test depends on
- PR A adds a new optimizer variant that PR B's deprecation sweep would remove
- PR A adds a `__getattr__` that interferes with PR B's attribute changes

**Dependency conflicts**: PR A depends on code introduced by PR B (or vice versa).
Merging A without B would break the build.

### 10.4 What to Do

If you find conflicts:

1. **File-level conflicts**: Note in your review which PRs overlap and on which files.
   Recommend a merge order if one PR is clearly simpler or more urgent.

2. **Semantic conflicts**: Describe the interaction in detail. Recommend which PR should
   merge first and what changes the second PR needs after the first merges.

3. **Dependency conflicts**: Note the dependency. The blocked PR cannot merge until the
   dependency is merged.

Include in your review:

```
## Cross-PR Conflicts

- PR #XXXX (title): overlaps on [files]. [description of conflict and recommendation].
- PR #YYYY (title): semantic conflict — [description].
```

If there are no conflicts, state: "No cross-PR conflicts detected."

---

## 11. Step 9: Test Assessment

### 11.1 Test Presence

Every non-trivial code change should have tests. Evaluate the PR's test coverage:

| PR Type | Test Expectation |
|---|---|
| Bug fix | Must have a regression test that fails without the fix |
| New feature | Must have tests covering happy path, edge cases, and error paths |
| Deprecation/removal | Must update or remove tests for deleted code |
| Refactoring | Existing tests should still pass; no new tests needed unless behavior is meant to change |
| Documentation | No tests needed |
| Build/CI | Build/CI tests may run as part of CI itself |
| Test-only | N/A (the PR IS the tests) |

### 11.2 Test Quality Assessment

For each test in the PR, evaluate:

**Does it test the right thing?**
- The test should verify the behavior described in the PR, not just that the code runs
  without errors.
- A test that calls the function and checks `isinstance(result, torch.Tensor)` is too
  weak. It should check values, shapes, dtypes, and device.

**Is it deterministic?**
- Tests that depend on random data should either set a seed or use tolerances that
  account for random variation.
- The bitsandbytes project uses statistical thresholds (mean + N*std) for precision
  tests. New precision tests should follow this pattern (see `code_standards.md`).

**Is it isolated?**
- Tests should not depend on other tests having run first.
- Tests should not depend on specific GPU models or CUDA versions unless explicitly
  marked as architecture-specific.
- Tests should clean up GPU memory and temporary state.

**Does it match the project's test style?**
- Consult `code_standards.md` for test patterns.
- Tests should use `pytest.mark.parametrize` for multi-configuration coverage.
- Tests should use `pytest.mark.skipif` for hardware/software-specific tests.
- Assertion messages should include enough context to debug a failure.

### 11.3 Test Coverage Gaps

Look for scenarios that the PR's tests do NOT cover but should:

- **Different dtypes**: If the feature works with fp16, bf16, and fp32, are all tested?
- **Different devices**: If the feature works on CUDA and CPU, are both tested?
- **Boundary values**: Zero-size tensors, single-element tensors, maximum-size tensors.
- **Non-contiguous inputs**: Sliced tensors, transposed tensors, tensors with strides.
- **Error paths**: What happens with invalid inputs? Are the error messages tested?

Note coverage gaps in your review, but distinguish between:
- **Blocking gaps**: Missing tests for the primary functionality (must fix before merge)
- **Non-blocking gaps**: Missing edge case tests (nice to have, can be added later)

### 11.4 Numerical Test Thresholds

For tests that compare quantized/dequantized values against reference values:

- [ ] **Are thresholds derived from analysis, not just empirical tuning?** The threshold
  should be explainable in terms of the quantization error model (e.g., codebook gap
  plus absmax encoding error plus accumulation error).

- [ ] **Do thresholds use the (mean, std) pattern?** The project standard is
  `threshold = mean + N*std` where N >= 7. See `code_standards.md` for details.

- [ ] **Are thresholds platform-independent?** A threshold that passes on RTX 4090 but
  fails on T4 or Blackwell is not robust. The (mean, std) pattern with sufficient sigma
  headroom handles this.

---

## 12. Step 10: Performance Impact Assessment

### 12.1 When to Assess

Assess performance impact when the PR changes:

- Code that runs in the hot path (forward pass, backward pass, quantization, matmul)
- Memory allocation patterns (new allocations, changed tensor sizes)
- Data movement (new `.to()` calls, new `.contiguous()` calls, new copies)
- CUDA kernel launch parameters (grid size, block size, shared memory)
- Python-level overhead in frequently-called functions

### 12.2 Hot Path Identification

The hot paths in bitsandbytes are:

1. **Forward pass**: `Linear4bit.forward()`, `Linear8bitLt.forward()`, `MatMul4Bit`,
   `matmul_4bit()`, `matmul()` (8-bit)
2. **Backward pass**: Autograd functions in `_functions.py`
3. **Quantization**: `quantize_4bit()`, `quantize_blockwise()`, and their dequantize
   counterparts
4. **Optimizer step**: `update_step()` in all optimizer classes

Changes to these paths deserve careful performance scrutiny.

### 12.3 Common Performance Concerns

**New `.contiguous()` calls:**
- `.contiguous()` is a no-op for already-contiguous tensors (just returns `self`)
- For non-contiguous tensors, it allocates a new tensor and copies data
- Adding `.contiguous()` at the top of a function is generally safe (the common case
  pays no cost), but verify that it's not called in a tight loop

**New `.clone()` calls:**
- `.clone()` always allocates and copies, even for contiguous tensors
- In the hot path, an unnecessary `.clone()` adds measurable overhead for large tensors
- If the clone is needed for correctness (e.g., preventing mutation of user data), it's
  justified. Note the tradeoff in your review.

**New Python-level conditionals:**
- Adding `if` statements to the forward path is generally fine (branch prediction)
- But adding Python-level loops or list comprehensions in the hot path is a concern

**Changed kernel launch parameters:**
- Changing grid size or block size affects occupancy and may cause performance
  regressions on some GPU architectures
- Changing shared memory usage affects the number of concurrent blocks per SM

### 12.4 Performance Assessment Output

Include in your review if applicable:

```
## Performance Impact

Hot path affected: [yes/no]
Changes:
- [description of each performance-relevant change]
- [expected impact: negligible / minor / significant / needs benchmarking]

Recommendation: [no concern / suggest benchmarking before merge / blocking]
```

---

## 13. Step 11: torch.compile Compatibility

### 13.1 When to Check

Check torch.compile compatibility when the PR:

- Modifies or adds ops in `bitsandbytes/_ops.py`
- Changes function signatures in the backends
- Adds new Python-level control flow that may interact with graph capture
- Uses data-dependent operations (e.g., `if tensor.sum() > 0:`)
- Modifies autograd functions

### 13.2 Op Registration

Every op registered with `torch.library` must have a fake tensor implementation (also
called an abstract implementation or meta registration). This tells torch.compile what
shape and dtype the op produces without actually running it.

```python
@torch.library.register_fake("bitsandbytes::some_op")
def _(input_tensor, ...):
    # Must return a tensor with the correct shape and dtype
    # Must NOT do any computation
    return torch.empty(expected_shape, dtype=expected_dtype, device=input_tensor.device)
```

Check:
- [ ] Does the fake implementation return the correct shape?
- [ ] Does the fake implementation return the correct dtype?
- [ ] Does the fake implementation handle all parameter combinations?
- [ ] Is the fake implementation registered for all new or changed ops?

### 13.3 opcheck Verification

The project uses `torch.library.opcheck` to verify op correctness. If the PR adds or
modifies ops, verify that:

- [ ] The op has an opcheck test (typically in the same test file as the op's
  functionality tests)
- [ ] The opcheck test passes with all standard opcheck test utilities

### 13.4 Graph Breaks

torch.compile traces Python code into a graph. Certain patterns cause "graph breaks"
where the compiler falls back to eager mode, reducing performance:

- `print()` statements in the forward path
- Data-dependent control flow (`if tensor.item() > 0:`)
- Calls to functions not known to the compiler
- Dynamic shape changes

If the PR introduces any of these in the forward path, note it as a potential
torch.compile regression.

---

## 14. Step 12: Checkpoint and Serialization Backward Compatibility

### 14.1 Why This Is Critical

Millions of pre-quantized model checkpoints exist on the HuggingFace Hub. These
checkpoints encode bitsandbytes state dict keys in a specific format. Any change to
this format breaks every existing checkpoint.

### 14.2 When to Check

Check serialization compatibility when the PR changes:

- `Params4bit`, `Int8Params`, or `QuantState` classes
- The `as_dict()` or `from_dict()` methods on QuantState
- The `from_prequantized()` class method on Params4bit
- State dict keys or the state dict structure of any nn.Module subclass
- The packed data format (bit-plane layout, absmax encoding)

### 14.3 Checkpoint Key Format

The current checkpoint format uses these keys per weight tensor:

**4-bit:**
```
model.layer.weight                           # packed quantized data
model.layer.weight.absmax                    # absmax scales
model.layer.weight.quant_map                 # codebook
model.layer.weight.nested_absmax             # (if double quant)
model.layer.weight.nested_quant_map          # (if double quant)
model.layer.weight.quant_state.bitsandbytes__nf4  # or __fp4
```

**8-bit:**
```
model.layer.weight                           # int8 data
model.layer.SCB                              # scale column-wise absmax
model.layer.weight_format                    # format metadata
```

Any change that adds, removes, renames, or reinterprets these keys is a **breaking
change** that affects every downstream consumer and every existing checkpoint.

### 14.4 Serialization Compatibility Checklist

- [ ] **Are state dict keys unchanged?** Compare the keys produced by `state_dict()`
  before and after the change.

- [ ] **Can old checkpoints still be loaded?** The new code must be able to load
  checkpoints saved by the previous version.

- [ ] **Can new checkpoints be loaded by old code?** If the new code changes what's
  saved, it should either be backward compatible or the PR must bump the version and
  include migration documentation.

- [ ] **Is QuantState.from_dict() still compatible?** vLLM uses this to reconstruct
  QuantState from checkpoint keys. Verify the dict format is unchanged.

- [ ] **Is the packed data format unchanged?** The bit-plane layout, blocksize, and
  E4M4 encoding must be the same, or existing quantized weights will decode incorrectly.

### 14.5 Serialization Impact Rating

| Change | Impact |
|---|---|
| Adding a new optional key to state dict | LOW (old code ignores it) |
| Renaming a key | CRITICAL (all checkpoints break) |
| Removing a key | CRITICAL (old code expecting it crashes) |
| Changing the data format behind a key | CRITICAL (silent corruption) |
| Changing QuantState.as_dict() output | HIGH (vLLM checkpoint loading breaks) |
| Changing Params4bit.from_prequantized() signature | HIGH (Transformers deserialization breaks) |

If the PR has CRITICAL serialization impact, it **must not merge** without:
1. Explicit maintainer approval
2. A migration plan for existing checkpoints
3. Coordinated releases with affected downstream projects

---

## 15. Step 13: Platform-Specific Review

### 15.1 When to Apply

Apply this section when the PR changes:

- CMakeLists.txt or any build configuration
- Platform detection code (`bitsandbytes/cuda_specs.py`, `_utils.py`)
- Conditional compilation (`#ifdef _WIN32`, `#ifdef __APPLE__`, etc.)
- ROCm-specific code paths
- Files under `csrc/` with platform-specific includes
- Anything under `.github/workflows/` that specifies platform

### 15.2 Platform Matrix

bitsandbytes supports:

| Platform | GPU Backend | Build System | Status |
|---|---|---|---|
| Linux x86_64 | CUDA | CMake | Primary, fully tested |
| Linux x86_64 | ROCm (HIP) | CMake | Supported |
| Linux aarch64 | CUDA | CMake | Supported |
| Windows x86_64 | CUDA | CMake | Supported |
| Windows x86_64 | ROCm | CMake | Experimental |
| macOS (any) | CPU only | CMake | Supported |
| macOS (Apple Silicon) | MPS | CMake | Experimental |
| Any | CPU only | CMake | Supported |

### 15.3 Platform-Specific Review Checklist

- [ ] **Does the change break other platforms?** A Windows fix should not break Linux.
  Check for platform-specific `#ifdef` guards, `platform.system()` checks, and
  conditional imports.

- [ ] **Is the platform detection robust?** Does it use `platform.system()` (reliable)
  or `os.name` (less reliable)? Does it handle edge cases (WSL, Cygwin, etc.)?

- [ ] **Are path separators correct?** Windows uses `\`, Unix uses `/`. Use
  `os.path.join()` or `pathlib.Path` instead of hardcoded separators.

- [ ] **Are subprocess calls cross-platform?** Commands like `rocminfo` may not exist
  on all platforms. Are they wrapped in try/except with appropriate fallbacks?

- [ ] **Are C/C++ includes portable?** `#include <unistd.h>` does not exist on Windows.
  Platform-specific includes need `#ifdef` guards.

- [ ] **Does the CMake change work with all supported generators?** Ninja, Make, and
  Visual Studio generators have different requirements.

### 15.4 ROCm-Specific Concerns

- ROCm uses HIP, which is similar to CUDA but not identical
- Warp size is 64 on CDNA (AMD data center GPUs) vs 32 on CUDA
- Some CUDA intrinsics (`__ballot_sync`, `__shfl_sync`) have different HIP equivalents
- `hipinfo` is used instead of `rocminfo` on Windows
- GPU architecture detection uses `rocminfo` output parsing, which differs between
  ROCm versions

### 15.5 Windows-Specific Concerns

- No `unistd.h` header (must be shimmed or avoided)
- Different shared library naming (.dll vs .so)
- Path length limits (260 characters by default)
- Different subprocess behavior (shell=True behaves differently)
- Visual Studio compiler has different warning/error behavior than GCC/Clang

---

## 16. Step 14: Commit Hygiene

### 16.1 Commit Structure

Evaluate the PR's commit history:

- [ ] **Are commits logically organized?** Each commit should represent one logical
  change. A commit that mixes a bug fix with an unrelated formatting change is messy.

- [ ] **Are commit messages descriptive?** Messages like "fix" or "update" are
  uninformative. Good messages explain what was changed and why.

- [ ] **Are there unrelated commits?** Sometimes PRs include commits from other branches
  (e.g., a formatting fix that was cherry-picked across multiple PRs). Flag these.

- [ ] **Is the commit count reasonable?** A 3-line bug fix with 15 commits (fix, fix
  again, oops, format, lint, ...) should be squash-merged.

### 16.2 Unrelated Changes

If the PR contains changes unrelated to its stated purpose:

- **Minor unrelated changes** (fixing a typo in a nearby comment, formatting an adjacent
  line): Acceptable, but note them.

- **Significant unrelated changes** (changing code in an unrelated file, adding an
  unrelated feature, modifying unrelated tests): Flag as a concern. Recommend splitting
  into separate PRs.

- **Commits from other PRs** (identical commits appearing in multiple open PRs): This
  usually means the PR branches share a common ancestor with commits that should have
  been on main. Note it and recommend rebasing.

### 16.3 Merge Strategy Recommendation

Based on the commit structure, recommend a merge strategy:

| Situation | Recommendation |
|---|---|
| Single well-structured commit | Regular merge or rebase |
| Multiple well-structured commits telling a clear story | Regular merge or rebase |
| Multiple commits with messy history | Squash merge |
| Unrelated commits mixed in | Request cleanup before merge |

---

## 17. Step 15: Produce and Post the Review

This step covers writing the review, formatting it, and posting it to GitHub. The review
length should be proportional to the number of issues found. A clean PR gets a brief
review; a problematic PR gets detail where the problems are.

### 17.1 Verdict Categories

Choose one of:

- **Approve**: The PR is ready to merge. No blocking issues. May have minor
  non-blocking suggestions.

- **Approve with minor changes**: The PR is fundamentally sound but needs small fixes
  (typos, minor code style issues, missing test edge cases). The changes are small enough
  that the author can make them without another full review.

- **Request changes**: The PR has blocking issues that must be addressed before merge.
  The author needs to make substantive changes and request re-review.

- **Needs discussion**: The PR raises architectural, design, or scope questions that
  should be discussed before proceeding. This is not a rejection — it's a request for
  clarification or consensus.

### 17.2 Review Format

The review body has three parts: a summary line, any issues or suggestions, and a
checklist of areas that were reviewed. Only issues get detailed discussion. Areas
with no problems are a single-line checklist entry.

**Clean simple PR (trivial/small, no issues):**

```markdown
## PR Review: #123 — Fix NF4 quantization edge case

Bug fix: corrects boundary handling in `dequantize_4bit` for zero-element blocks.

**No blocking issues.**

- Security: Clear
- Downstream impact: None
- Tests: Adequate
- CI: All pass
```

**Clean complex PR (medium/large, no issues):**

A well-executed complex PR deserves acknowledgment proportional to its scope.
Summarize what the PR accomplishes and confirm the key areas were checked.

```markdown
## PR Review: #789 — Add Intel XPU backend support

Adds a new `XPUBackend` class with implementations for all quantization ops,
new device detection in `cextension.py`, and XPU-specific test parametrization
across 8 test files. The implementation follows the existing backend pattern
(MPS, NPU) consistently.

**No blocking issues.**

The backend registration uses the standard `@register_kernel` pattern and all
new ops have corresponding test coverage including dtype and device edge cases.

- Security: Clear
- Downstream impact: None (additive — new backend, no changes to existing APIs)
- Tests: Adequate (new parametrization covers XPU across all op categories)
- CI: All pass
- torch.compile: Compatible (uses `torch.library` registration)
```

**PR with blocking issues:**

```markdown
## PR Review: #456 — Refactor Params4bit constructor

Refactors `Params4bit` constructor, removing the `compress_statistics` parameter.

**Blocking issues (2):**

1. **Breaks PEFT and Transformers** — `compress_statistics` is passed directly by
   both projects. Removing it without a deprecation cycle will break
   `bnb.nn.Params4bit(data, **old.__dict__)` round-trips. See inline comment at
   `bitsandbytes/nn/modules.py:142`.

2. **Missing test update** — `test_linear4bit.py` still passes `compress_statistics`
   to the constructor; this test would fail after the change but was not updated.

Suggestion: Add a deprecation warning that accepts the old kwarg for one release cycle.

- Security: Clear
- Downstream impact: HIGH (PEFT, Transformers, Accelerate)
- Tests: Need update for new signature
- CI: Not yet run
```

**Formatting rules:**

- **Summary**: One or two sentences. What the PR does and the overall assessment.
- **Issues section**: Only present when there are blocking issues. Each issue gets a
  numbered entry with a bold title, a description, and file/line references. The top
  2-5 issues should also be posted as inline comments (see Section 17.4).
- **Suggestions**: Brief, unnumbered lines below the issues section. Only include
  suggestions that are genuinely useful. Do not pad with style nits.
- **Checklist**: Always present. One line per area. If an area has a problem, the
  checklist entry states the problem instead of "Clear." The standard checklist items
  are:
  - **Security** — Clear / [describe issue]
  - **Downstream impact** — None / LOW / MEDIUM / HIGH / CRITICAL with affected projects
  - **Tests** — Adequate / Needs improvement / Missing / [describe gap]
  - **CI** — All pass / Failures / Not yet run / Not triggered
  - Additional items only when relevant: Performance, Cross-PR conflicts,
    Serialization compatibility, torch.compile

### 17.3 Posting the Review to GitHub

Reviews are posted as formal GitHub reviews using the `gh` CLI — not as plain PR
comments. Formal reviews appear in the PR's "Reviews" tab and participate in branch
protection merge checks.

**Rule: Never approve.** The agent must never submit a review with `--approve`. Even
when the verdict is "Approve" or "Approve with minor changes," submit using `--comment`.
Formal approval is reserved for human maintainers.

**Rule: Use `--request-changes` only for security issues.** When the agent identifies a
security concern (malicious code patterns, supply chain risks, credential exposure, or
any issue from the security guide's Tier 1-5 categories), use `--request-changes` to
formally block the PR. For all other blocking issues — correctness bugs, breaking API
changes, missing tests — use `--comment` and state the blocking issues clearly in the
review body.

**Verdict-to-action mapping:**

| Verdict | GitHub action | Rationale |
|---|---|---|
| Approve | `--comment` | Positive signal, but human must formally approve |
| Approve with minor changes | `--comment` | Same — positive, not a formal gate |
| Request changes (non-security) | `--comment` | States blocking issues; human decides whether to enforce |
| Request changes (security) | `--request-changes` | Formally blocks merge until resolved |
| Needs discussion | `--comment` | Raises questions, not blocking |

**Posting command (when you have no inline comments):**

```bash
# Standard review (most cases)
gh pr review <NUMBER> --comment --body "$(cat <<'EOF'
<review body here>
EOF
)"

# Security-blocking review (only for security issues)
gh pr review <NUMBER> --request-changes --body "$(cat <<'EOF'
<review body here>
EOF
)"
```

If you have inline comments to attach, use the `gh api` method in Section 17.4 instead
— it posts the review body and inline comments in a single request, replacing the
`gh pr review` command above.

### 17.4 Inline Comments

The top 2-5 findings (blocking issues and the most important suggestions) should be
posted as inline comments on specific lines in the PR diff. These are submitted as part
of a single review together with the review body, using the GitHub API. When posting
inline comments, use the `gh api` method below instead of the `gh pr review` command
in Section 17.3 — do not post both.

Inline comments make the review actionable — the author sees the feedback exactly where
the issue is in the code, rather than having to cross-reference line numbers from the
review body.

**Posting a review with inline comments:**

The repo for bitsandbytes is `bitsandbytes-foundation/bitsandbytes`. Substitute the
PR number from Step 1.

The recommended approach is to build the JSON payload in a temporary file, then pass
it to `gh api`. This avoids shell quoting issues with inline JSON:

```bash
# Step 1: Write the JSON payload to a temp file
cat > /tmp/review_payload.json <<'REVIEW_JSON'
{
  "body": "## PR Review: #456 — Refactor Params4bit constructor\n\nRefactors `Params4bit` constructor, removing the `compress_statistics` parameter.\n\n**Blocking issues (2):**\n\n1. **Breaks PEFT and Transformers** — see inline comment.\n2. **Missing test update** — see inline comment.\n\n- Security: Clear\n- Downstream impact: HIGH (PEFT, Transformers, Accelerate)\n- Tests: Need update for new signature\n- CI: Not yet run",
  "event": "COMMENT",
  "comments": [
    {
      "path": "bitsandbytes/nn/modules.py",
      "line": 142,
      "side": "RIGHT",
      "body": "Removing `compress_statistics` breaks PEFT and Transformers — both pass this kwarg directly via `Params4bit(data, **old.__dict__)`."
    },
    {
      "path": "tests/test_linear4bit.py",
      "line": 87,
      "side": "RIGHT",
      "body": "This test still passes `compress_statistics` to the constructor. It would fail after this change."
    }
  ]
}
REVIEW_JSON

# Step 2: Post the review
gh api repos/bitsandbytes-foundation/bitsandbytes/pulls/456/reviews \
  --method POST \
  --input /tmp/review_payload.json

# Step 3: Clean up
rm /tmp/review_payload.json
```

For security-blocking reviews, change `"event": "COMMENT"` to
`"event": "REQUEST_CHANGES"` in the JSON.

**JSON field reference:**

| Field | Type | Description |
|---|---|---|
| `body` | string | The full review body text. Use `\n` for newlines. |
| `event` | string | `COMMENT` for standard reviews, `REQUEST_CHANGES` for security blocks. Never use `APPROVE`. |
| `comments` | array | Inline comments to attach. Optional — omit or pass `[]` if none. |
| `comments[].path` | string | File path relative to repo root (e.g., `bitsandbytes/nn/modules.py`). |
| `comments[].line` | integer | Line number in the file that appears in the diff. For `RIGHT`, this is the line number in the new version. The line must be visible in `gh pr diff` output (a changed line or a context line around a change). The API rejects lines not in the diff. |
| `comments[].side` | string | `RIGHT` for lines in the new version (most common). `LEFT` for deleted lines only visible in the old version. |
| `comments[].body` | string | The inline comment text. Use `\n` for newlines. |

**Inline comment guidelines:**

- Use the `line` and `side` fields (not the deprecated `position` field). `line` is
  the line number in the file. `side` is `RIGHT` for lines in the new version of the
  file (the most common case) or `LEFT` for lines only in the old version (deletions).
- Each inline comment should be self-contained — the reader should understand the issue
  without needing to read the full review body.
- Keep inline comments concise. One to three sentences. If the issue needs a longer
  explanation (e.g., downstream impact details), put the full analysis in the review
  body and keep the inline comment as a pointer: "This changes the constructor
  signature. See review body for downstream impact analysis."
- Do not use inline comments for non-issues or praise. They are for problems and
  specific suggestions only.

**When not to use inline comments:**

- If the review has no blocking issues and no significant suggestions, skip inline
  comments entirely. The review body checklist is sufficient.
- If the issue is architectural or cross-cutting (affects the whole PR, not a specific
  line), put it only in the review body.

### 17.5 Re-Reviews

When the PR author pushes changes in response to a review, submit a new review — do
not edit or delete the previous one. The previous review stays as history.

The re-review should:
- State which previous blocking issues are resolved and which remain
- Identify any new issues introduced by the changes
- Update the checklist accordingly

A re-review follows the same format and posting rules as the initial review. If all
previous blocking issues are resolved and no new ones are found, the re-review is a
brief "No blocking issues" review.

### 17.6 Severity Guidelines

When classifying issues as blocking vs non-blocking, use these guidelines:

**Always blocking:**
- Correctness bugs in the implementation
- Missing tests for new functionality or bug fixes
- Breaking changes to public API without justification
- CRITICAL or HIGH downstream impact without mitigation plan
- Serialization format changes without migration plan
- Security issues (hardcoded secrets, command injection, etc.)
- Build failures caused by the PR
- CI lint failures caused by the PR

**Usually blocking (use judgment):**
- Missing error handling for likely error cases
- Performance regressions in the hot path
- Incomplete implementations (TODO/FIXME left in code)
- Tests that don't actually test the right thing
- torch.compile incompatibilities

**Usually non-blocking:**
- Code style issues beyond what linters catch
- Missing tests for unlikely edge cases
- Documentation improvements
- Commit message quality
- Minor naming suggestions
- Additional comments or docstrings

---

## 18. Merge Readiness Checklist

After producing an "Approve" or "Approve with minor changes" verdict, verify these
merge prerequisites:

### 18.1 Pre-Merge Checks

- [ ] **CI is green.** All required checks pass. If CI hasn't run (fork PR), note that
  a maintainer must approve the workflow run first.

- [ ] **No merge conflicts.** The PR cleanly merges into the base branch. If there are
  conflicts, the author must rebase.

- [ ] **All review comments are resolved.** If there were previous review rounds, verify
  that all requested changes have been addressed.

- [ ] **Approval from maintainer.** The PR has approval from at least one maintainer
  (not just this automated review).

### 18.2 Changelog Considerations

Determine whether the PR warrants a changelog entry:

| PR Type | Changelog? |
|---|---|
| Bug fix affecting users | Yes |
| New user-facing feature | Yes |
| API deprecation or removal | Yes |
| Performance improvement | Yes, if significant |
| Internal refactoring | No |
| Documentation only | No |
| Test only | No |
| CI/build only | No, unless it affects user build process |
| Style/lint only | No |

If a changelog entry is needed and the PR doesn't include one, note it as a non-blocking
suggestion.

### 18.3 Version Considerations

Determine whether the PR requires a version bump:

- **Patch version** (0.x.Y → 0.x.Y+1): Bug fixes, minor improvements
- **Minor version** (0.X.0 → 0.X+1.0): New features, non-breaking API additions
- **Major version** (X.0.0 → X+1.0.0): Breaking API changes

Individual PRs do not typically bump the version — that's done at release time. But if
the PR introduces breaking changes, note that a version bump will be needed at the next
release.

---

## 19. Common Review Pitfalls

These are mistakes that reviewers (both human and agent) commonly make. Be aware of them.

### 19.1 Approving Based on Tests Alone

A PR with comprehensive tests can still have fundamental design problems. Tests tell you
the code works for the tested cases; they don't tell you the approach is correct, the
API is well-designed, or the change won't break downstream consumers.

Always evaluate design and downstream impact independently of test coverage.

### 19.2 Missing the Behavioral Change in a "Bug Fix"

Some PRs labeled as "bug fixes" actually change behavior in ways that affect users.
For example:

- A "fix" that changes a default parameter value
- A "fix" that adds validation that rejects previously-accepted input
- A "fix" that changes the output format (e.g., different dtype, different shape)

These are behavior changes, not just bug fixes, and need to be evaluated as such.

### 19.3 Ignoring the Diff Context

The diff shows lines around the changes. These context lines often reveal:

- The changed code is inside a rarely-used branch
- The changed code is inside a hot loop
- There's a comment explaining why the old code was written that way
- There's a TODO or FIXME that the PR should have addressed

Read the context, not just the green/red lines.

### 19.4 Over-Focusing on Style

Style issues are the easiest to spot and the least impactful. If you spend all your
review time on naming and formatting, you may miss correctness bugs, downstream breakage,
or design problems.

The linting pipeline catches most style issues automatically. Focus your review on things
the linter cannot check: correctness, design, compatibility, and completeness.

### 19.5 Assuming Tests Pass Because CI Is Green

CI runs on specific hardware with specific configurations. Tests may pass in CI but fail
on other hardware (different GPU architecture, different CUDA version, different OS).

If the PR adds hardware-specific code, consider whether the CI matrix covers the relevant
configurations.

### 19.6 Missing Interactions Between Changed Files

When a PR changes multiple files, review the interactions between the changes, not just
each file in isolation. Common interaction bugs:

- Function signature changed in one file but not all call sites updated
- New import added but the import order violates the project's isort config
- New parameter added to a constructor but not passed through from the wrapper layer

### 19.7 Reviewing Against the Wrong Base

Verify what the PR is based on:

```bash
gh pr view <NUMBER> --json baseRefName -q '.baseRefName'
```

Most PRs target `main`. A PR targeting a feature branch needs to be reviewed in that
context. A PR targeting the wrong base branch is a red flag.

---

## 20. Reference: File-to-Concern Mapping

When a PR changes a file, this table tells you which review concerns apply beyond the
general checklist.

### 20.1 Python Source Files

| File/Pattern | Primary Concern | Secondary Concerns |
|---|---|---|
| `bitsandbytes/__init__.py` | Public API exports | Downstream isinstance checks, import paths |
| `bitsandbytes/nn/__init__.py` | Module type exports | PEFT/Transformers isinstance checks |
| `bitsandbytes/nn/modules.py` | Linear4bit, Linear8bitLt, Params4bit, Int8Params | **ALL downstream projects**, serialization, `__dict__` round-trip, FSDP, torch.compile |
| `bitsandbytes/functional.py` | Quantization functions, QuantState | Downstream dequantize calls, checkpoint format, matmul semantics |
| `bitsandbytes/_ops.py` | Op registration | torch.compile fake implementations, backend dispatch |
| `bitsandbytes/autograd/_functions.py` | Autograd wrappers | Backward pass correctness, gradient computation |
| `bitsandbytes/optim/*.py` | Optimizer classes | Transformers trainer integration, state dict format |
| `bitsandbytes/optim/optimizer.py` | Base optimizer, GlobalOptimManager | Transformers' `manager.register_module_override()` |
| `bitsandbytes/backends/cuda/ops.py` | CUDA backend dispatch | Kernel launch parameters, dtype handling |
| `bitsandbytes/backends/cpu/ops.py` | CPU backend | CPU fallback behavior |
| `bitsandbytes/cuda_specs.py` | GPU detection, CUDA version | Platform-specific behavior, ROCm compatibility |
| `bitsandbytes/_utils.py` | Utility functions | Platform detection, path handling |

### 20.2 C/CUDA Source Files

| File/Pattern | Primary Concern | Secondary Concerns |
|---|---|---|
| `csrc/kernels.cu` | CUDA kernel correctness | Memory safety, precision, launch config, template instantiation |
| `csrc/kernels.cuh` | Kernel declarations | Must match `kernels.cu` |
| `csrc/ops.cu` | C++ launch wrappers | Dtype dispatch, grid/block calculation, error handling |
| `csrc/ops.cuh` | Op declarations | Must match `ops.cu` |
| `csrc/pythonInterface.cpp` | Python bindings | Must match Python op registrations in `_ops.py` |
| `csrc/common.h` | Shared constants and types | Affects all CUDA code |
| `CMakeLists.txt` | Build configuration | Platform compatibility, CUDA architectures, dependencies |

### 20.3 Test Files

| File/Pattern | Primary Concern | Secondary Concerns |
|---|---|---|
| `tests/test_functional.py` | Core quantization and matmul tests | Precision thresholds, parametrization coverage |
| `tests/test_linear4bit.py` | Linear4bit module tests | Serialization round-trip, device movement |
| `tests/test_linear8bitlt.py` | Linear8bitLt module tests | Threshold behavior, mixed precision |
| `tests/test_optim.py` | Optimizer tests | State dict round-trip, convergence, all variants |
| `tests/test_autograd.py` | Autograd tests | Gradient correctness, graph capture |
| `tests/test_nn.py` | Neural network module tests | Forward/backward, parameter handling |
| `tests/test_parametrize.py` | Parameter/module interaction tests | Precision, shapes, devices |

### 20.4 Configuration Files

| File/Pattern | Primary Concern | Secondary Concerns |
|---|---|---|
| `pyproject.toml` | Build metadata, dependencies | Version constraints, extras, ruff config |
| `.pre-commit-config.yaml` | Lint hooks | Hook versions, configurations |
| `.github/workflows/*.yml` | CI pipelines | Test matrix, action versions, secrets |
| `_typos.toml` | Spell-check exceptions | False positive allowlist |

---

## 21. Reference: API Change Impact Quick-Lookup

This is a condensed version of the downstream impact tables from `downstream_integrations.md`.
Use it for quick lookups during review. For full details, consult the source document.

### 21.1 Maximally Dangerous APIs (used by 4+ downstream projects)

Changing any of these breaks the most downstream consumers:

| API | Projects using it |
|---|---|
| `bnb.nn.Linear4bit` (class) | Transformers, PEFT, Accelerate, (TGI reimplements) |
| `bnb.nn.Linear8bitLt` (class) | Transformers, PEFT, Accelerate, (TGI reimplements) |
| `bnb.nn.Params4bit` (class) | Transformers, PEFT, Accelerate, TGI |
| `bnb.nn.Int8Params` (class) | Transformers, PEFT, Accelerate, TGI, vLLM |
| `Params4bit.quant_state` (attribute) | Transformers, PEFT, Accelerate, TGI |
| `Int8Params.SCB` (attribute) | Transformers, PEFT, Accelerate, TGI |
| `functional.dequantize_4bit()` | Transformers, PEFT, vLLM |
| `bnb.matmul()` | TGI, vLLM |
| `bnb.matmul_4bit()` | TGI, vLLM |
| `bnb.MatmulLtState` | TGI, vLLM |

### 21.2 High-Risk Attribute Access

These attributes are accessed directly by downstream projects (not through methods):

| Attribute | Accessed by |
|---|---|
| `Params4bit.__dict__` (full round-trip) | PEFT, Accelerate |
| `Params4bit.compress_statistics` | Transformers, PEFT |
| `Params4bit.quant_type` | Transformers, PEFT |
| `Params4bit.bnb_quantized` | PEFT |
| `Params4bit.quant_storage` | Transformers, PEFT |
| `Linear4bit.compute_dtype` | Transformers, PEFT |
| `Linear8bitLt.state` | Transformers, PEFT |
| `MatmulLtState.CB` | TGI, vLLM |
| `MatmulLtState.SCB` | TGI, vLLM |
| `MatmulLtState.CxB` | TGI, vLLM |
| `MatmulLtState.threshold` | PEFT, TGI, vLLM |
| `MatmulLtState.has_fp16_weights` | PEFT, TGI, vLLM |

### 21.3 String-Based Class Name Checks

These class names are checked by string comparison (not isinstance) in downstream code.
Renaming them breaks downstream even though the functionality is unchanged:

| Class name | Checked by |
|---|---|
| `"Int8Params"` | Accelerate (`set_module_tensor_to_device`) |
| `"Params4bit"` | Accelerate (`set_module_tensor_to_device`, `fsdp_utils.py`), PEFT (`peft_model.py`) |
| `"FP4Params"` | Accelerate (`set_module_tensor_to_device`) — legacy |
| `"Linear8bitLt"` | Accelerate (`set_module_tensor_to_device`) |
| `"Linear4bit"` | Accelerate (`set_module_tensor_to_device`) |

### 21.4 Serialization Keys

These checkpoint key patterns are used by downstream loaders. Changing them breaks every
pre-quantized checkpoint:

| Key pattern | Used by |
|---|---|
| `weight.absmax` | Transformers, vLLM |
| `weight.quant_map` | Transformers, vLLM |
| `weight.nested_absmax` | Transformers, vLLM |
| `weight.nested_quant_map` | Transformers, vLLM |
| `weight.quant_state.bitsandbytes__nf4` | Transformers, vLLM |
| `weight.quant_state.bitsandbytes__fp4` | Transformers, vLLM |
| `weight.SCB` (8-bit) | Transformers, Accelerate |

---

## 22. Reference: Review Depth by Classification

This table summarizes which review steps require deep analysis vs a quick check for each
PR classification.

| Step | Bug Fix | Feature | Deprecation | Refactor | Docs | Build/CI | Test |
|---|---|---|---|---|---|---|---|
| CI Status | Quick | Quick | Quick | Quick | Quick | Deep | Quick |
| Issue Linkage | Deep | Deep | Deep | Quick | Skip | Skip | Quick |
| Code Review | Deep | Deep | Deep | Deep | Quick | Deep | Deep |
| Downstream Impact | Deep | Deep | **Critical** | Medium | Skip | Skip | Skip |
| Cross-PR Conflicts | Quick | Quick | Deep | Quick | Skip | Quick | Skip |
| Test Assessment | Deep | Deep | Medium | Quick | Skip | Skip | N/A |
| Performance Impact | Medium | Deep | Skip | Quick | Skip | Skip | Skip |
| torch.compile | Quick | Deep | Quick | Quick | Skip | Skip | Skip |
| Serialization | Medium | Deep | **Critical** | Medium | Skip | Skip | Skip |
| Platform Review | Skip* | Skip* | Skip | Skip | Skip | Deep | Skip |
| Commit Hygiene | Quick | Medium | Quick | Quick | Quick | Quick | Quick |

\* Unless the bug fix or feature is platform-specific.

**Legend:**
- **Critical**: Must be done thoroughly. Blocking issues are likely.
- **Deep**: Full analysis required. Spend significant time.
- **Medium**: Check carefully but don't expect to find problems often.
- **Quick**: Scan briefly. Flag obvious issues only.
- **Skip**: Not applicable for this PR type.
- **N/A**: Not applicable by definition.
