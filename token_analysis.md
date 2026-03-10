# Claude Code Token Analysis

## Session data location

Session JSONL files are stored at:
```
~/.claude/projects/<project-path>/<session-id>.jsonl
```

Each file contains one JSON object per line with types: `user`, `assistant`, `system`, `progress`, `file-history-snapshot`.

## Methodology

### Input tokens (prefill)

Input = user prompts + tool results. These are measured from `user`-type messages in the JSONL:
- `content[].type == "text"` entries give user prompt text
- `content[].type == "tool_result"` entries give tool outputs (file reads, grep, bash)

Token count estimated at chars/4. System prompt, system injections, and the model's own prior output re-read as context are excluded — we only count new content the user/tools provide.

### Generated tokens (decode)

Generated = `output_tokens` from the `usage` field on `assistant`-type messages. This includes all model generation: text responses, tool call arguments, and thinking tokens (thinking content is encrypted so can't be separated).

### Per-turn grouping

A "turn" = one user message + all assistant API calls until the next user message. A single user turn may trigger multiple API calls (model calls a tool, gets result, calls another tool, etc.). Input for a turn = content in that user message. Output for a turn = sum of `output_tokens` across all API calls in that turn.

### Histogram bucketing

Values are bucketed to nearest power of two: `2^round(log2(n))`.

## Aggregate results: 397 sessions, 25,162 user turns

Data collected from 472 session files across all projects (75 empty/skipped). 41,537 total API calls.

| | Est. tokens |
|---|---:|
| Input (prefill) | ~31.8M |
| Generated (decode) | ~2.3M |
| **Ratio** | **13.7:1 input to output** |

### Frequency distributions

Per-turn frequency distributions (summing to 1.0) are stored in `token_distributions.json`. The file contains two distributions:

- `input_tokens_per_turn.freq` — estimated prefill tokens per user turn (user text + tool results). 24,155 non-empty turns.
- `generated_tokens_per_turn.freq` — decode tokens per user turn (from API `output_tokens`). 20,911 non-empty turns.

Keys are power-of-two bucket sizes (as strings), values are frequencies.

### Interpretation

- Input peaks at 16-32 tokens (short prompts, small tool results) with a flat tail through 2048. Reflects a mix of user typing (small) and tool results (variable).
- Output is bimodal: peaks at 2 tokens (20%, single short tool call) and 32 tokens (19%, tool call with moderate argument). Text responses and code blocks (128-2048) account for ~17% of turns.
- Heavy generation (>4096 tokens) is rare (<0.5% of turns).

## Kernel performance weighted by workload

The token distributions in `token_distributions.json` serve as a workload model for estimating which GEMM kernels matter most in practice. The key mapping: **input tokens per turn = prefill M** (new tokens processed in a single forward pass with KV cache), **generated tokens per turn = number of decode steps at M=1** (or M=batch_size in multi-user serving).

### Single-user inference (M=1 decode)

In single-user autoregressive generation, each turn involves:
- **1 prefill pass** at M = input_tokens (prompt/tool results, distributed by `input_tokens_per_turn`)
- **N decode passes** at M = 1, where N is the number of generated tokens (distributed by `generated_tokens_per_turn`)

The average generated tokens per turn is ~114. So a typical turn has 1 prefill pass + 114 decode passes. Even though large prefills are individually expensive (a single M=32768 pass costs ~23,000 us/layer), they are rare enough (~1.4% frequency) that decode at M=1 dominates total wall-clock time at **80-84%** across k=2..5.

Per-layer time breakdown (k=4, Qwen3-Coder-Next shapes):

| Component | Time/turn/layer | % of total |
|-----------|----------------:|------------|
| Decode (114 steps x 55.6 us) | 6,347 us | 83.4% |
| Prefill (distributed) | 1,260 us | 16.6% |

The scalar GEMV kernel (M=1) is faster than fp16 cuBLAS because it reads 3-4x less data (k-bit compressed weights vs fp16). Overall weighted slowdown vs fp16: **0.57x** (43% faster) at k=4.

### Multi-user serving with vLLM

Production deployments use continuous batching (vLLM), which changes the M distribution fundamentally. The vLLM V1 scheduler (`vllm/v1/core/sched/scheduler.py`) works as follows:

1. **Decode-first**: all running (decoding) requests are scheduled first, each contributing 1 token. M starts at num_decoding_users.
2. **Chunked prefill**: remaining token budget is used for at most one prefill chunk from a waiting request. Default chunk size is `max_model_len * 0.04` (e.g., 1280 for 32K context, 5120 for 128K).
3. **Token budget cap**: total tokens per step is bounded by `max_num_batched_tokens` (default 8192).
4. **One partial prefill at a time**: `max_num_partial_prefills` defaults to 1.

This creates a **bimodal M distribution**: iterations are either pure-decode (M = num_users) or decode + prefill chunk (M = num_users + chunk_size). The MMA kernel's effective range (M=8-32) falls in the gap between these modes and is rarely used.

Simulation results (k=4, chunk_size=512, token distributions from `token_distributions.json`):

| Users | Avg M | Decode-only iters | Dominant kernel | vs fp16 |
|------:|------:|------------------:|-----------------|--------:|
| 1 | 8 | 98.6% | scalar (87%) | 0.57x |
| 4 | 41 | 92.6% | scalar (59%) + dq+cuBLAS (41%) | 0.76x |
| 8 | 77 | 86.1% | MMA (45%) + dq+cuBLAS (55%) | 0.85x |
| 16 | 163 | 70.2% | dq+cuBLAS (76%) | 1.00x |
| 32 | 364 | 30.9% | dq+cuBLAS (93%) | 1.17x |
| 64 | 495 | 5.1% | dq+cuBLAS (98%) | 1.23x |

The crossover where quantized kernels become slower than fp16 is at **~16 concurrent users**. Below that, bandwidth savings from k-bit compression outweigh the dequant overhead. Above that, the dequant cost (~30 us/shape at k=4) dominates because most iterations include a large prefill chunk where cuBLAS is highly efficient.

### Optimization priorities

The analysis identifies two regimes with different optimization targets:

**1-4 users (agents, local inference, code assistants):**
The scalar GEMV at M=1..4 accounts for 59-87% of total GEMM time. This kernel is already bandwidth-bound and faster than fp16. Further optimization (better ILP in the M-loop, wider vector loads) has the highest leverage. The dq+cuBLAS path handles the occasional prefill chunk (~41% of time at 4 users) with moderate overhead (1.25x vs fp16). The MMA kernel is effectively unused.

**16+ users (serving, API endpoints):**
dq+cuBLAS dominates (75-98% of time). The ~30 us dequant overhead per shape at k=4 is the primary cost. Reducing this — through a faster dequant kernel, fusing dequant into the matmul, or accepting float32 absmax to skip format conversion — would directly reduce the 1.17-1.23x slowdown vs fp16.

**The MMA kernel has minimal impact in either regime.** Its effective range (M=8-32) corresponds to pure-decode batches at 8-32 users, which is a shrinking slice of iterations as user count grows. At 4 users, M never reaches the MMA range. At 32 users, only 31% of iterations are pure-decode at M=32, and MMA accounts for just 5.8% of total weighted time.

## Script

```python
import json, math

SESSION = "~/.claude/projects/<project>/<session-id>.jsonl"

with open(SESSION) as f:
    lines = [json.loads(l) for l in f]

timeline = [l for l in lines if l.get('type') in ('user', 'assistant')]

turns = []
for i, msg in enumerate(timeline):
    if msg['type'] != 'user':
        continue
    content = msg.get('message', {}).get('content', '')
    input_chars = 0
    if isinstance(content, list):
        for c in content:
            if c.get('type') == 'text':
                input_chars += len(c.get('text', ''))
            elif c.get('type') == 'tool_result':
                rc = c.get('content', '')
                if isinstance(rc, str):
                    input_chars += len(rc)
                elif isinstance(rc, list):
                    input_chars += sum(len(json.dumps(x)) for x in rc)
    elif isinstance(content, str):
        input_chars += len(content)

    total_output = 0
    for j in range(i + 1, len(timeline)):
        if timeline[j]['type'] == 'user':
            break
        if timeline[j]['type'] == 'assistant':
            total_output += timeline[j]['message']['usage'].get('output_tokens', 0)

    turns.append({'input_est': input_chars // 4, 'output': total_output})

def bucket(n):
    if n <= 0: return 0
    return 2 ** round(math.log2(max(n, 1)))

for label, key in [("Input", "input_est"), ("Generated", "output")]:
    vals = [t[key] for t in turns if t[key] > 0]
    buckets = {}
    for v in vals:
        b = bucket(v)
        buckets[b] = buckets.get(b, 0) + 1
    mx = max(buckets.values())
    print(f"\n=== {label} tokens per turn ({len(vals)} turns) ===")
    for b in sorted(buckets):
        bar = "#" * max(1, round(buckets[b] / mx * 40))
        print(f"{b:>8}  {buckets[b]:>5}  {bar}")
```
