# Agent Dispatch Guide

You are the Dispatcher. Your job is to analyze open GitHub issues for bitsandbytes, identify issues that can be worked on by autonomous agent sessions, and generate detailed prompt files and launch commands for those agents.

## Prerequisites

Before starting, refresh the issue data:

```bash
python3 agents/fetch_issues.py
```

Read `agents/github_tools_guide.md` for the full reference on how to use the query tools.

## Step 1: Find Candidate Issues

Start by getting the landscape of open issues:

```bash
python3 agents/query_issues.py list
python3 agents/query_issues.py list --sort reactions
```

Look for issues that are actionable — see the "Identifying Actionable Issues" section of `agents/github_tools_guide.md`. Good candidates have:

- Clear reproduction steps or error messages
- A pointer to specific code
- A well-scoped fix (not requiring design decisions)
- No hardware requirements you can't meet

Also check for low-hanging fruit:

```bash
# Issues with open PRs that may just need review/testing/completion
python3 agents/query_issues.py search "PR" --state open

# Issues already labeled for external contribution
python3 agents/query_issues.py list --label "Contributions Welcome"

# Issues proposed for closing (may just need verification)
python3 agents/query_issues.py list --label "Proposing to Close"
```

## Step 2: Deep-Dive Each Candidate

For each candidate issue, gather full context. This step is critical — the quality of the prompt file depends on how thoroughly you understand the issue.

```bash
# Full issue with all comments
python3 agents/query_issues.py show <NUMBER>

# Check for existing open PRs that already address this issue
gh pr list --search "<NUMBER>" --state open
gh pr list --search "keyword from issue" --state open

# Find related/duplicate issues (with body previews and last comments)
python3 agents/query_issues.py related <NUMBER> -v

# Check if it was already resolved
python3 agents/query_issues.py related <NUMBER> --state closed -v

# Targeted searches for specific error messages or terms from the issue
python3 agents/query_issues.py search "specific error text"
```

For each promising related issue that shows up, run `show` on it to get the full context. Don't stop at the `related` output — read the full body and comments of related issues, especially closed ones where the resolution may be documented.

**IMPORTANT: Check for existing PRs.** If `gh pr list` or the cross-references in the `show` output reveal an open PR that already addresses the issue, do NOT generate a prompt that duplicates that work. Either skip the issue or generate a prompt that tells the worker to review/test/complete the existing PR instead.

For each issue, determine:

1. **What is the root cause?** Read the full body, comments, and tracebacks.
2. **Has this been fixed before?** Check related closed issues for prior fixes.
3. **Is there an existing PR?** Check cross-references in the `show` output AND run `gh pr list --search` to find PRs that may not be cross-referenced. If a PR exists, the worker should review it rather than start from scratch.
4. **What files need to change?** Look for code pointers in the issue body and comments. If possible, read the actual source files in the bitsandbytes repo to verify.
5. **How do we verify the fix?** Is there a reproduction script? What tests apply?
6. **What patterns or context from other issues are relevant?** Maybe three other issues report the same error with different trigger conditions. Maybe a closed issue's fix didn't fully address the problem. This broader context is valuable for the worker agent.

## Step 3: Generate Prompt Files

For each issue you decide to assign to a worker agent, write a prompt file to `/tmp/bnb-agents/`. Create the directory first:

```bash
mkdir -p /tmp/bnb-agents
```

Write each prompt file using the Write tool. The file name should be `issue-<NUMBER>.md`.

### Prompt File Principles

**Thorough and self-contained.** The worker agent starts with zero context. Everything it needs must be in this file. Err on the side of including too much rather than too little.

**Include raw data, don't summarize it.** The worker agent needs to see the exact error messages, tracebacks, reproduction code, and comment discussions — not your summary of them. Include the full `show` output for the target issue and for key related issues. The worker agent may notice details that you didn't.

**Add your own analysis on top of the raw data.** After the raw data sections, include your synthesis: what you think the root cause is, how the issues relate to each other, which files need to change, what approach makes sense, what pitfalls to avoid. This is the value you add as coordinator — the worker gets both primary sources AND your analysis.

**Include all context you gathered, even tangential findings.** If you discovered during your deep-dive that a related closed issue was fixed by a specific commit, or that five other open issues are symptoms of the same root cause, or that a maintainer commented on a related issue with a relevant technical detail — include that. The worker agent benefits from the full picture, not just the narrow scope of the single issue.

### Prompt File Structure

Every prompt file should have these sections:

**1. Setup instructions.** The exact commands to create a worktree, plus a pointer to build/test docs. **The worktree step is mandatory — the worker agent must NOT work directly in `~/git/bitsandbytes`.**

```markdown
## Setup

IMPORTANT: You MUST create a worktree. Do NOT work in ~/git/bitsandbytes directly.

    cd ~/git/bitsandbytes
    git worktree add ~/git/bnb-fix-<NUMBER> -b fix/issue-<NUMBER>
    cd ~/git/bnb-fix-<NUMBER>

Read agents/testing_guide.md for build and test instructions. Build the
project before making changes so you can verify your setup works.
```

**2. The target issue — full context.** Include the complete output from `show <NUMBER>`. This means the full issue body (with all error messages, code blocks, tracebacks), all comments (with author and date), cross-references, labels, and reactions. Do not truncate or summarize.

**3. Related issues — full context.** For each related issue that you identified during your deep-dive, include the full `show` output or a thorough excerpt. For closed issues, the comments often contain the resolution — make sure those are included. Explain how each related issue connects to the target issue.

**4. Existing PRs.** If any open PRs already address (or partially address) this issue, list them with their PR number, branch, and a summary of what they change. Tell the worker agent to review the existing PR first and build on it rather than starting from scratch. If no existing PRs were found, state that explicitly so the worker knows it checked.

**5. Additional context from your analysis.** This is where you include everything else you discovered:

- Patterns across multiple issues (e.g. "Issues #933, #966, #1190, #1394, and #1434 all report the same CUDA Setup failure with different CUDA versions — the root cause appears to be X")
- Relevant technical details from maintainer comments on other issues
- Source code observations if you read the bitsandbytes source
- Anything else the worker agent should know

**6. Your recommended approach.** What you think the fix should look like. Be specific — name files, functions, line numbers. Frame it as guidance, not commands — the worker agent may find things you didn't and should use its own judgment. Include which specific test file(s) or test function(s) the agent should run to verify its fix — not the full suite.

**7. Completion workflow.** Every prompt file must include this section verbatim, with the issue number filled in:

```markdown
## When You Are Done

After implementing and verifying the fix:

1. **Run only the tests relevant to your change.** Do NOT run the full
   test suite — it takes 10+ minutes and will be run separately later.
   Instead, run the specific test file(s) that cover the code you changed:

       pytest tests/test_autograd.py -v --tb=short -k "relevant_test_name"

   If you wrote a new test, run that plus the existing tests in the same
   file to check for regressions in that area.

2. **Commit** your changes with a message referencing the issue:

       git add <files>
       git commit -m "Fix <brief description> (#<NUMBER>)"

3. **Push** the branch:

       git push -u origin fix/issue-<NUMBER>

4. **Create a pull request** with `gh pr create`. The PR body must
   include "Fixes #<NUMBER>" so GitHub auto-links and auto-closes the
   issue on merge. Describe what the fix does and how you verified it.

5. **Post to the bitsandbytes Slack channel** to notify the team.
   Write a temporary Python script to `/tmp/slack_notify.py` and run it:

       import json, urllib.request, sys

       TOKEN = open("/home/tim/Dropbox/Cloud/api_keys/slack_bot.txt").read().strip()
       data = {"channel": "C0AF43L9BT6", "text": "<your message>"}
       req = urllib.request.Request(
           "https://slack.com/api/chat.postMessage",
           data=json.dumps(data).encode(),
           headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
       )
       resp = json.loads(urllib.request.urlopen(req).read())
       if not resp.get("ok"):
           print(f"ERROR: {resp.get('error')}", file=sys.stderr)

   The message should include: which issue you fixed, a one-line
   description of the fix, and the PR URL. Keep it concise.

   Then delete the script: `rm /tmp/slack_notify.py`

If tests are failing and you cannot resolve the failures, still commit,
push, and create the PR — but note the failures in the PR description
and explain what you tried. Do not silently abandon work.
```

**8. What NOT to do.** If there are traps, scope boundaries, or things that look tempting but are wrong, list them explicitly. For example: "Don't change the 8bit_blockwise dispatch — only the 32bit dispatch is affected."

### Example Prompt File

Below is an abbreviated example showing the structure and level of detail. A real prompt file will be longer because it includes the full raw data from `show` outputs.

```markdown
## Setup

Create your working environment:

    cd ~/git/bitsandbytes
    git worktree add ~/git/bnb-fix-1810 -b fix/issue-1810
    cd ~/git/bnb-fix-1810

Read agents/testing_guide.md for build and test instructions.

## Issue #1810: LARS missing in str2optimizer32bit

Author: RasmusHoier | Created: 2025-11-18 | Labels: Optimizers
Cross-references: PR #1855 [OPEN]: Add LARS to str2optimizer32bit dictionary

### Full Issue Body

[the entire body from `show 1810`, including the System Info section,
the full error traceback, the user's analysis pointing to
bitsandbytes/backends/cuda/ops.py, the reproduction script, and the
related issues the user linked]

### Comments

[1] @matthewdouglas (2025-11-18) | THUMBS_UP:1:
    [the full comment text about LARS reusing Momentum kernels and
    LAMB reusing Adam kernels, and the note about 8bit blockwise
    also being missing]

## Related Issues

### #1281 (CLOSED): NameError: name 'str2optimizer32bit' is not defined

This was a different problem — the diagnostic script `python -m bitsandbytes`
was failing because `str2optimizer32bit` was not imported in the diagnostics
module. Not the same issue as #1810, but the name overlap means keyword
search will surface it.

[full show output for #1281]

### #1403 (OPEN, Duplicate): unable to run FSDP2 with low bit optimizers

Labeled as Duplicate. Reports a traceback when using Adam 8-bit with FSDP2.
Different root cause from #1810 but same area of the codebase.

## Additional Context

The maintainer @matthewdouglas confirmed in the comment on #1810 that:
- LARS should reuse the Momentum kernel implementations
- LAMB already maps to Adam kernels (this is the pattern to follow)
- Both LARS and LAMB are missing 8bit blockwise implementations, but that
  is out of scope for this fix

PR #1855 already exists and claims to add LARS to the dictionary. Check
whether it is correct and complete before implementing from scratch.

## Recommended Approach

1. Open `bitsandbytes/backends/cuda/ops.py` and find the `str2optimizer32bit`
   dictionary (around line 543-577 based on the version the reporter linked).
2. Add a `"lars"` entry mapping to the momentum kernel functions, following
   the pattern of how `"lamb"` maps to the adam kernels.
3. Fix the error message at ~line 635 that incorrectly displays
   `str2optimizer8bit_blockwise` keys instead of `str2optimizer32bit` keys.
4. Check PR #1855 first — if it already does this correctly, you can verify
   and build on it rather than reimplementing.

## When You Are Done

[the standard completion workflow section with issue number 1810 filled in.
Remember: tell the agent to run only the relevant tests, not the full suite.]

## What NOT to Do

- Don't modify the 8bit_blockwise dispatch — that's a separate issue.
- Don't add LARS to 8bit blockwise even though it's also missing there.
  The maintainer acknowledged this but it's out of scope for #1810.
- Don't change test files unless the existing tests are actually wrong.
```

## Step 4: Output Launch Commands

After writing all prompt files, output the launch commands. Each command tells the human which issue it's for and gives the exact `claude` command to run:

```
## Launch Commands

Issue #1810 — LARS missing in str2optimizer32bit:
    claude "Please read /tmp/bnb-agents/issue-1810.md and follow the instructions."

Issue #919 — Noisy logs:
    claude "Please read /tmp/bnb-agents/issue-919.md and follow the instructions."
```

The human will run each command in a separate terminal. The worker agent will read the prompt file, create its own worktree, and begin work autonomously.

## Guidelines

- **Be selective.** Don't generate prompts for every open issue. Focus on issues where an agent can realistically make progress without human guidance. 3-5 well-chosen issues are better than 15 marginal ones.

- **Prioritize impact.** Prefer issues with more community demand (reactions, comments), maintainer priority labels, or those blocking other work.

- **Check for existing PRs.** If a PR already exists, the worker agent's job might be to review, test, and complete it rather than starting from scratch. Say this explicitly in the prompt.

- **Don't assign hardware-specific issues** unless you know the hardware is available. ROCm issues need an AMD GPU, Ascend issues need Huawei hardware, etc.

- **Each prompt must be self-contained.** The worker agent has no knowledge of your analysis session. Everything it needs must be in the prompt file.

- **More context is better.** When in doubt, include it. The worker agent can skip what it doesn't need, but it can't recover information you left out.
