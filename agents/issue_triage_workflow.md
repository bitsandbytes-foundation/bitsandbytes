# Issue Triage Workflow: Human + Agent Collaboration

This document describes the interactive workflow for triaging GitHub issues
using a human maintainer and a Claude Code agent working together. This is
how we reduced the bitsandbytes issue tracker from 152 open issues to ~60
in a single session.

The key insight: the agent handles volume (reading every issue, spotting
patterns, drafting comments, executing closures) while the human handles
judgment (deciding what's a real bug, what tone to strike, what the project's
priorities are). Neither could do this efficiently alone.

## How It Works

### Phase 1: Landscape scan

The agent fetches all open issues and groups them by pattern. This is the
most time-consuming step if done manually, but an agent can read 150+
issues and classify them in minutes.

What the agent does:
- Fetches issue data with `fetch_issues.py`
- Queries by label (`Duplicate`, `Proposing to Close`, `Waiting for Info`, etc.)
- Reads every issue with `show --brief` in batches of 10-15
- Identifies clusters: issues that share the same root cause, error message,
  or theme

What the agent produces:
- A grouped table of issues, organized by pattern
- For each group: issue numbers, titles, and a short rationale for why
  they're closeable
- An estimate of how many issues can be closed

The human reviews the groups and says which ones to proceed with. The agent
does not close anything without human approval.

### Phase 2: Iterative triage

This is the core loop. It works in rounds:

1. **Agent presents a group** (e.g., "13 issues all report the same legacy
   CUDA setup error on bnb 0.41.x-0.42.x").

2. **Human decides** — close all, close some, investigate further, or skip.
   The human may also:
   - Ask the agent to investigate a specific issue more deeply
   - Provide domain context ("this was fixed in v0.43.0", "FSDP1 is not
     going to be supported", "the offset value was empirically optimized")
   - Override the agent's recommendation ("don't close that, it's a real bug")
   - Specify tone ("no comment needed", "explain what they were asking",
     "say we're working on it but no ETA")

3. **Agent executes** — closes issues with tailored comments, using `gh
   issue close --comment`. The agent adapts the comment to each issue's
   specific context (version, platform, error message) rather than
   copy-pasting a template.

4. **Agent reports back** — confirms what was closed, then identifies the
   next group.

This loop typically runs 5-8 rounds in a session. Each round closes 5-25
issues depending on the cluster size.

### Phase 3: Discussion and documentation

Some issues are not simply closeable — they reveal gaps in documentation,
recurring user confusion, or real bugs that need work. The triage session
naturally surfaces these:

- **Documentation gaps**: If 5 issues ask the same question about NF4, the
  code needs better docstrings. The agent drafts the documentation, the
  human reviews, and they commit together.

- **Real bugs that need work**: The agent writes a dispatch prompt file
  (see `dispatch_guide.md`) so another agent session can work on the fix
  independently.

- **Pattern documentation**: New patterns discovered during triage get added
  to `issue_patterns.md` so future triage sessions can reference them.

## The Human's Role

The human's judgment is essential for:

- **Deciding what's a real bug vs. user error.** The agent can identify
  patterns, but the human knows the codebase history and what's been fixed.

- **Setting project priorities.** "We're not going to support FSDP1" or
  "mixed quantization is something we're working toward" — these are
  project decisions the agent can't make.

- **Tone and messaging.** The human decides whether an issue gets a detailed
  explanation, a brief "this was fixed, please upgrade", or no comment at
  all. Some issues deserve a thoughtful response even when being closed.

- **Catching false positives.** The agent may recommend closing something
  that looks stale but is actually an important edge case. The human's
  domain knowledge catches these.

- **Cross-referencing.** "Before closing duplicates, are they
  cross-referenced to the canonical issue?" — the human ensures no
  information is lost.

## The Agent's Role

The agent handles the work that's tedious for humans but trivial for an LLM:

- **Reading every issue.** An agent can read and classify 150 issues in a
  few minutes. A human doing this manually would spend hours.

- **Pattern detection.** The agent identifies that 15 issues all reference
  `cuda_setup/main.py` line 166, or that 5 issues all load `.so` files
  on Windows — patterns a human might miss when reading issues one at a time.

- **Comment drafting.** Each closed issue gets a tailored comment explaining
  why it's being closed and what the user should do. The agent writes these
  with the specific context of each issue (version, platform, error message).

- **Cross-reference checking.** Before closing a duplicate, the agent
  verifies the canonical issue exists, is still open, and already
  cross-references the duplicate.

- **Batch execution.** Closing 15 issues with individual comments would
  take a human 30+ minutes of copy-paste. The agent does it in parallel.

## Practical Tips

### Starting a session

```
cd ~/git/bitsandbytes
claude
```

Then say something like: "Look at the open issues and identify groups of
issues that can be closed — duplicates, stale issues, old version problems,
questions that aren't bugs. Give me an overview before closing anything."

### Pacing

Don't try to close everything at once. Work in groups:
1. Start with the lowest-hanging fruit (already labeled Duplicate, Proposing
   to Close)
2. Move to pattern clusters (CUDA setup, Windows pre-support, etc.)
3. Then handle the one-offs (stale questions, third-party app issues)
4. End with discussion items that need human judgment

### When the agent is wrong

The agent will occasionally recommend closing something that shouldn't be
closed. This is expected and fine — that's why the human reviews before
execution. Common false positives:
- Issues that look stale but are actually waiting on a specific release
- Feature requests that look like questions but represent real community
  demand
- Issues the agent thinks are old-version problems but actually reproduce
  on current code

Just say "don't close that one" and move on.

### Turning triage into action

The best outcome of a triage session isn't just fewer open issues — it's
discovering what work actually needs to be done. Issues that survive triage
are the real backlog. During the session:

- If an issue is a real bug, consider generating a dispatch prompt
  (see `dispatch_guide.md`) so a worker agent can fix it.
- If multiple issues reveal the same documentation gap, fix the docs in
  the same session and reference the commit when closing the issues.
- If a cluster of issues reveals a systemic problem (e.g., "everyone on
  Jetson hits the same error"), that's a signal to prioritize platform
  support work.

## Related Documents

- `issue_patterns.md` — catalog of known closeable patterns with templates
- `issue_maintenance_guide.md` — autonomous agent guide for triage (no
  human in the loop)
- `dispatch_guide.md` — how to generate prompts for worker agents to fix
  real bugs
- `github_tools_guide.md` — reference for the issue query tools
