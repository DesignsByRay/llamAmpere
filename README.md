# Flash-Next on two or three cards: per-device expert tables for a layer split

> **AI-assisted work.** The code on this branch and this document were written with an AI coding assistant (Claude Code). I directed the work, ran it on my own machine and reviewed the results. Every commit carries a `Co-Authored-By` line that says so. Please read the numbers and claims below with that in mind.

Branch: `multi-gpu-device-tables`, 13 commits on top of llamAmpere v0.6.4 (`a5d8684`). It is NOT rebased onto v0.7.

This page replaces the project README on this branch only. The original llamAmpere README is unchanged in [README-llamAmpere.md](README-llamAmpere.md). The code-only branch is `multi-gpu-device-tables`.

## Purpose

v0.6.4 runs Flash-Next on one card. Its release notes say that splitting it across two cards is not supported yet. On my machine (3x RTX 3090) the split path as shipped was not usable: 0.7 tokens/s writing.

This branch makes the device expert tables work per card when the model is split by layer (`-sm layer`, the default). Each card holds the experts of the layers it runs. One-card behaviour is unchanged.

## The bug found on the way (not specific to this branch)

Commit `d63a678` fixes a memory lifetime fault in the CUDA side-stream code. It is in v0.6.4 source and only shows when a model is split over several backends, so one-card use never hits it. I checked that the fix is not in v0.7.0.1.

What happens:

- A side-stream chain (the shared-expert chain or the Q/K/V group, `ggml_cuda_shexp_chain_find` and `ggml_cuda_side_group_find` in `ggml-cuda.cu`) runs on a second CUDA stream while the main stream goes on.
- The code keeps the chain's inputs allocated until the join, but only inputs that are nodes of the split graph.
- With a layer split the scheduler gives each card its own copy of a graph input, here the positions (`inp_pos`). That copy is not a graph node, so it was not kept.
- Its memory was handed to main-stream nodes while the side-stream ROPE kernels could still read it.
- When a graph is evaluated the kernels are issued in order and nothing goes wrong. When a captured CUDA graph is replayed the two streams really run in parallel. On the last card the positions were then sometimes read wrong.

Symptom: with the drafter on and a layer split, the output changed from run to run at temperature 0.

How it was found: turning off the Q/K/V side stream (`GGML_CUDA_QKV_SIDE=0`) made the output repeatable. A dump of the data ranges of every node between a group's start and its join then showed the overlaps: 6 in layer 47 on the last card, 2 each in layers 15 and 31. After the fix: 0 overlaps in 91 side runs.

The fix: inputs of a side-stream chain that are not graph nodes and not model weights are added to the chain's kept inputs (12 lines).

How to reproduce (needs two or more CUDA devices):

1. Build this branch at commit `17ed920` (the commit before the fix).
2. Start the server with Flash-Next split over the cards and the drafter on.
3. Use the repeatable settings: `--moe-skip off --moe-skip-budget 0 --moe-skip-rank-floor 0 --no-spec-mb-mtp --no-moe-cpu-split --spec-type draft-mtp --spec-draft-n-max 3`, with `LLAMA_MOE_BANK_DET=1` and `GGML_MOE_CACHE_DET=1` in the environment.
4. Send the same short prompt several times at temperature 0 and compare the token IDs. They differ between passes.
5. Add `GGML_CUDA_QKV_SIDE=0`, or build `d63a678`: the passes are identical, and identical to a one-card run.

Limits of this claim: I saw the fault on this branch's multi-card path. I did not show it on untouched v0.6.4, because the multi-card path there is too slow to test. That the same fault exists there is read from the source. There is no regression test.

## Results

Machine: Windows 11, Threadripper 9970X, 128 GB RAM, 3x RTX 3090 (24 GB), GPU clocks locked. Built with MSVC 14.50 and CUDA 13.3, `-DCMAKE_CUDA_ARCHITECTURES=86`. Model: the AI3 file with the MTP drafter, flags as in `scripts/flash-next-server.sh`. Prompt of about 86,000 tokens, 1,024 tokens written, 3 runs, server timings. Writing varies by several tokens/s from run to run.

| Setup | Writing tok/s | Prompt tok/s |
|---|---|---|
| v0.6.4 as shipped, 3 cards visible (1 run) | 0.7 | 213 |
| v0.6.4, one card, upstream flags | 72.0 | 2,299 |
| This branch, one card, upstream flags | 71.8 | 2,289 |
| This branch, 3 cards, `--moe-skip off` | 103.5 | 2,143 |
| This branch, 3 cards, `-ts 15,16,17`, `--moe-skip off` | 107.2 | 1,985 |
| This branch, 2 cards, `--moe-skip off` | 97.9 | 1,345 |
| This branch, 3 cards, AI4 file, `--moe-skip off` | 91.5 | 1,605 |

Notes on the table:

- The multi-card rows run with `--moe-skip off` because skip is not supported on a split (see below). One card with `--moe-skip off` writes 65.1 tokens/s, so the like-for-like gain is 65.1 to 103.5.
- A later repeat of the 3-card row measured 99.7 / 2,124.
- Drafter off: 3 cards 66.9 / 2,314, 2 cards 68.7 / 1,592, AI4 on 3 cards 62.1 / 1,812 (measured on earlier commits of the branch).
- With 3 cards all experts of AI3 sit on the cards. With 2 cards about 10.8 GiB of experts stay in host memory.
- Load time 15 to 18 s. Card memory peaks in a 30-minute mixed run: 22.4 / 20.4 / 22.4 GiB.

## Correctness

Checked with the repeatable settings listed above, at temperature 0, comparing token IDs and logprobs:

- One card, this branch against stock v0.6.4: identical on 5 prompts.
- 3 cards against one card: identical on 5 prompts, including a 20,600-token and an 86,000-token prompt, drafter off and on. A 2,000-token answer is identical for all 2,000 tokens.
- Context sizes 32768 and 65536: identical. A multi-turn chat: 11 of 11 turns identical.
- 2 cards against one card: identical on short prompts.
- AI4, 3 cards against one card: identical on short prompts.
- 30-minute mixed run on 3 cards: 128 requests, 0 failures, writing speed flat.

When comparing long prompts, keep `LLAMA_B1` the same on both sides. It changes the numerics.

## Known issues and limits

- **Based on v0.6.4.** A trial merge onto v0.7.0.1 conflicts in `ggml/src/ggml-cuda/moe-cache.cu` and `src/llama-context.cpp`. Not rebased, not tested on v0.7.
- **Fixed cost per request on a split.** With 2 or 3 cards every request waits about 2.6 s before the first token, however small the prompt. After each prompt the engine re-plans the expert cache and rebuilds the per-card tables. One card does not do this for a small prompt (about 0.4 s). Workaround: `LLAMA_MOE_CACHE_RELEASE_PROMPT=0` (0.1 to 0.3 s). With it, short prompts are no longer token-identical to one card. I did not find out why one card skips the re-plan.
- **Off on a split:** `--moe-skip`, expert swap, miss staging and the CPU split. Run with `--moe-skip off`.
- **Two cards, long prompts:** the prompt chunk size follows free card memory, so long prompts do not repeat exactly from run to run. Pinning `-ub 2048` makes them repeat.
- **AI4 on 3 cards:** a 20,600-token prompt differed between two passes, for the same reason.
- **`-ts 15,16,17`:** faster writing, slower prompt reading than the default split. On an earlier commit a compute-buffer allocation failed at the start of a long prompt with an even split.
- **Log lines seen on two cards:** at start `cudaMalloc failed ... retrying without pipeline parallelism` (the server then runs), and once `device expert prepare failed before a 2-row call ... it runs on host experts`. Not looked into.
- **Not run:** Linux, the project's CI, `test-backend-ops`, `llama-bench`. Only this one machine and CUDA architecture 86.
- **Tests:** `test-moe-cache` now builds on Windows. 14 of its 36 single-case options pass there; the same 22 fail on untouched v0.6.4 source. There is no new test for the layer split or for the bug fix.

Not from this branch, but seen while testing: after a reply that the client sends back changed, the server re-reads thousands of tokens (about 7,800 in my test), because it can only go back to a checkpoint on the 8,192-token grid. Stock v0.6.4 on one card does the same. `LLAMA_SJKVARN_EXACT_CKPT=1` with `LLAMA_SJKVARN_EXACT_MAX_TOKENS=200000` removes it.

## What the commits do

- `e45fbb5`, `80c3437`: experts go to the card that runs their layer; device tables per card.
- `cdd8a7d`, `17ed920`, `ef72af7`, `343e126`: long-prompt speed on a split (experts copied on the device, full prompt chunks, large expert ops on the layer's own card, one kernel per resident tensor).
- `d63a678`: the fix for the fault described above. History note: before the cause was found, the branch had a workaround commit (no graph replay on the card that runs the last layer). The fix removed it again, so I folded the two into this one commit before publishing. The source is the same as it was with both commits. The details of the search are available on request.
- `0a59125`, `13263ae`, `3400b1b`: tight card memory (smaller prompt chunks, tensors that are only partly on a card).
- `b87e733`: fixes "overlapping host registration rejected" errors when some experts stay in host memory.
- `0ec7244`: a draft context never gets pipeline parallelism, so the drafter works on a split.
- `ee009f5`: `test-moe-cache` builds on Windows.

## How to run it

Same flags as `scripts/flash-next-server.sh`, with all cards visible (or `--device CUDA1,CUDA2` for two) and `--moe-skip off` added at the end:

```
llama-server -m <AI3 first shard> -md <MTP drafter> --moe-profile data/moe/flash-next-experts.strp -ctk sjkvarn4 -ctv sjkvarn4 -ngl 999 -fa on -b 131072 -ub 8192 --fit off --moe-arena on --moe-cache on --ctx-checkpoints 4 -c 131072 -t 8 -tb 8 --parallel 1 --spec-type draft-mtp-adaptive --spec-draft-n-min-adaptive 3 --spec-draft-n-max 4 --moe-skip off
```

with the environment from `data/moe/flash-next.env` and `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

---

Drafted with Claude Code (Claude Fable 5.2). Measurements were made on my machine between 2026-11-15 and 2026-11-18.
