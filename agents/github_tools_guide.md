# Using GitHub Tools for bitsandbytes Issue Analysis

The `agents/` directory contains scripts for fetching and querying GitHub issues. This guide covers how to use them for bitsandbytes specifically.

## Data Setup

Before starting any analysis, refresh the local issue data:

```bash
python3 agents/fetch_issues.py
```

This fetches all open and closed issues (~1200 total) into `agents/bitsandbytes_issues.json` (gitignored). Takes ~13 API calls, safe to run every session. The data includes full issue bodies, all comments, labels, reactions, cross-references, and timeline events.

## Getting the Landscape

Start with an overview of all open issues:

```bash
# All open issues, most recently updated first
python3 agents/query_issues.py list

# Only unlabeled issues (often untriaged)
python3 agents/query_issues.py list --unlabeled

# Most community-demanded issues
python3 agents/query_issues.py list --sort reactions

# Most discussed issues
python3 agents/query_issues.py list --sort comments

# Issues by category
python3 agents/query_issues.py list --label "Bug"
python3 agents/query_issues.py list --label "Optimizers"
python3 agents/query_issues.py list --label "CUDA Setup"
```

The `list` output includes linked PRs (shown as `PR#1234`), which indicates someone has already started work.

## Understanding an Issue

To get full context on a specific issue (body, all comments, cross-references):

```bash
python3 agents/query_issues.py show 1810
```

For multiple issues at once:

```bash
python3 agents/query_issues.py show 1810 782 547
```

Use `--brief` when you only need the headline information (truncated body, first + last comment):

```bash
python3 agents/query_issues.py show --brief 1810
```

## Finding Related and Duplicate Issues

To find issues related to a specific issue:

```bash
# With body previews and last comment (recommended)
python3 agents/query_issues.py related 1810 -v

# Only closed (resolved) issues — useful for finding prior fixes
python3 agents/query_issues.py related 1810 --state closed -v
```

For multiple issues at once:

```bash
python3 agents/query_issues.py batch-related 1810 1815 1849 -v
```

The `related` command uses keyword and error-signature matching. It is a filtering tool, not a semantic similarity engine. When it doesn't find good matches, fall back to keyword search:

```bash
python3 agents/query_issues.py search "LARS optimizer"
python3 agents/query_issues.py search "str2optimizer"
```

## Screenshot-Only Issues

Some issues post error messages as screenshots. The image URLs are in the body as `<img src="...">` tags. To extract text:

1. Download: `curl -sL -o /tmp/gh_img.png "<URL>"`
2. Read the image with the Read tool — Claude can extract text from terminal screenshots directly.
3. Use the extracted text for `search` queries.
4. Clean up: `rm /tmp/gh_img.png`

## bitsandbytes Issue Categories

These are the common patterns across the ~1200 issues:

**Platform/hardware issues** — ROCm, Ascend NPU, Intel XPU, aarch64, Windows, macOS. These require specific hardware to reproduce and test. Unless you have access to the hardware, these are not actionable.

**CUDA Setup failures** — The single largest category. "CUDA Setup failed despite GPU being available" appears in dozens of issues. Most are user environment problems (wrong CUDA version, missing libraries, container issues). Many are duplicates of each other.

**Optimizer issues** — Missing optimizers, optimizer bugs, checkpoint resumption problems. The codebase has `str2optimizer8bit_blockwise` and `str2optimizer32bit` dispatch dictionaries in `bitsandbytes/backends/cuda/ops.py` — missing entries are a common bug pattern.

**Quantization issues** — NF4/FP4/INT8/INT4 quantization bugs, wrong outputs, device movement problems, compatibility with specific models.

**Build/compile issues** — Users building from source hitting compile errors. Often specific to CUDA versions or OS.

**Integration issues** — Problems when using bitsandbytes through transformers, PEFT, or other HuggingFace libraries.

**Feature requests** — New optimizers, new quantization methods, new platform support, API improvements.

## Identifying Actionable Issues

An issue is likely actionable by an agent when it has:

- A clear error message or traceback
- Reproduction steps or code
- A pointer to specific code (file, function, line number)
- An existing open PR that needs review or completion
- A "Contributions Welcome" label
- A clear scope (missing dictionary entry, wrong error message, documentation gap)

An issue is likely NOT actionable by an agent when it:

- Requires specific hardware (ROCm, Ascend, specific GPU generation)
- Needs architectural or design decisions ("Needs Input from Tim", "To Discuss Internally")
- Is a vague performance complaint without reproduction
- Is really a question about usage, not a bug

## Label Reference

| Label | Meaning |
|---|---|
| Bug | Confirmed or suspected bug |
| Enhancement | Improvement to existing feature |
| Feature Request | New functionality |
| Question | User asking for help, not reporting a bug |
| Duplicate | Already covered by another issue |
| Proposing to Close | Maintainer thinks this can be closed |
| Waiting for Info | Blocked on info from the reporter |
| Contributions Welcome | Maintainer would accept a PR for this |
| High/Medium/Low Priority | Maintainer-assigned priority |
| CUDA Setup | CUDA detection/loading issues |
| Build | Build/compile issues |
| Optimizers | Optimizer-related |
| FSDP | FSDP integration |
| ROCm / Ascend NPU / Intel / Windows / macOS / aarch64 | Platform-specific |
