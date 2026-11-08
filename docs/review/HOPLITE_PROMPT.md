# Briefing for hoplite

hoplite, this is a hand-over from the review of `main` at `c92e3af`. You've done good groundwork: the P3 audit test, the fit estimator's shape, the benchmark report and the policy engine are the right building blocks. The review found that several of them give wrong answers today, and that the 30-40B goal needs a different route than the roadmap implies. Here's what I need from you.

## Read first, in this order

1. [`FINAL_REPORT.md`](FINAL_REPORT.md): sections 1-5 and 8. Section 3 is the bug list; section 8 is your task order.
2. [`../design/DESIGN_HANDBOOK.md`](../design/DESIGN_HANDBOOK.md): sections 3, 5 and 7 before you touch the TUI.
3. The component report for whatever you're working on, in [`research/`](research/). Each has file:line references and acceptance tests.

Don't start coding until you've read 1 and 2. If something there contradicts what you see in the code, trust the code, tell the owner, and note it in your PR.

## Ground rules

- **Labels.** Every user-facing claim (README, docs, CLI output, TUI) is SHIPPED, REPORTED-ONLY or PLANNED. Your last status said this was already done; it wasn't on `main`. This PR adds the labels to the docs; keep them true as you ship.
- **No numbers without evidence.** No "speedup", "x faster" or tok/s in output unless it was measured in this session, or it's labelled "ceiling estimate". Real benchmarks run only on the owner's machine (list below). You don't run them and you don't invent them.
- **One concern per PR**, each with tests and a CHANGELOG line under "Unreleased".
- **Don't touch:** the project name (stays EigenCore; no rename to "eigen"), `assets/brand/` until the owner approves the v2 mark, the ruff `<0.15` pin, and the sandbox-only `llama_cpp` test failures you already scoped out. Don't add binary files (PNG, PPTX, PDF) to PRs without the owner asking.
- **Disagree in writing.** If you think an item below is wrong, say why in the PR description with an alternative and the risk. Don't silently reorder.

## Tasks, in order

### Stage A: make what ships tell the truth

1. **Fit from real GGUF metadata.** Read `general.architecture`, then `<arch>.block_count`, `.attention.head_count`, `.attention.head_count_kv`, `.attention.key_length` / `.value_length`, `.context_length`, `.embedding_length`. Use `gguf` (gguf-py) to read metadata without loading weights; fall back to `Llama.metadata` after load. Weight bytes from the file size or tensor sizes, not params × constant. Exact KV element sizes (f16 2, q8_0 34/32, q4_0 18/32). Fix the GiB/GB mix in `hal/profiler.py`. Make the CLI pass geometry. Remove the 0.0005 GB/1K/1B heuristic or mark the result `UNKNOWN` with a deliberately pessimistic value.
   - *Verify:* fixture GGUF headers for a Llama, a Qwen2 and a Gemma; KV within 1% of the formula; context recommendation changes with RAM below 38 GB.
2. **Fix the Q4_0 reference and its test.** ggml order: low nibble of byte j → element j, high nibble → element j+16; value = (q − 8) × d. Accept raw 18-byte and 34-byte blocks so the fp16 scale is read from bytes, not a float array.
   - *Verify:* bit-exact against `gguf.quants.dequantize` for Q4_0 and Q8_0 on random blocks. Delete the test that encodes the interleaved order.
3. **Agent path safe by default.** `ToolExecutor` gets `default_policy()` unless a policy is passed explicitly; `Forge.agent()` and `ReActAgent` pass one. Tools declare which params are paths (don't infer from names); the handler receives the resolved `Path`. Replace the thread timeout with a child process you can kill (`multiprocessing` spawn context; `os.killpg` on POSIX, `TerminateJobObject` or `Process.kill` on Windows). Redact and sanitise successful output too.
   - *Verify:* EXECUTE tool denied via `Forge.agent()` with defaults; a READ tool with `src="/etc/passwd"` denied; 3 s handler with 0.2 s timeout returns in < 0.5 s and no child survives; Windows CI passes.
4. **Chat robustness.** Print model text with `markup=False` / `Text(...)`; strip C0/C1 controls and OSC/CSI sequences (keep `\n`, `\t`); catch `KeyboardInterrupt` during streaming to stop generation and keep history; never regenerate after a mid-stream exception; header tools label comes from the attached policy object.
   - *Verify:* tests with `[/]`, `\x1b]0;pwn\x07`, `\x1b[2J`, Ctrl-C mid-stream, in Rich and plain mode.
5. **Quarantine misleading optimizers.** Rewrite `SpeculativeDecoder.verify_draft` as Leviathan/Chen rejection sampling over full probability vectors, or delete it. Remove `expected_speedup`, `estimated_speedup`, `potential_speedup` from user output. Rename `analyze` output to "output-logit sparsity" and drop its "theoretical CPU speedup". Remove the "<2% perplexity" docstring.
   - *Verify:* chi-square test that accepted-token frequencies match the target distribution (the current code must fail it); `grep -ri speedup` in CLI output paths returns only measured comparisons.
6. **Architecture-aware metadata** in `engine/inference.py` (`<arch>.` prefix instead of `llama.`), sharing the helper from task 1.
7. **Harness:** count tokens via the tokenizer or usage fields, not stream chunks; report cold and warm separately.

### Stage C: the realistic 30B-class path (after Stage A)

8. **MoE support in fit and registry.** Add one MoE GGUF (Qwen3-30B-A3B class) to the registry. Fit reports `resident_gb` and `active_gb_per_token` (from `<arch>.expert_count`, `.expert_used_count` and tensor sizes). Profile and chat warn when resident > free RAM ("will read from SSD"). Expose `use_mmap` / `use_mlock`.
9. **Prompt-lookup decoding** via `llama_cpp.llama_speculative.LlamaPromptLookupDecoding(num_pred_tokens=2)`, off by default.
   - *Verify:* greedy output identical on and off. The owner decides the default from his measurements (≥ 1.2× median or it stays off).
10. **Expose KV and attention options:** `type_k`, `type_v`, `flash_attn`, `n_ubatch`; the estimator uses the chosen KV type.
11. **GPU/VRAM fit and `n_gpu_layers`.** Lower priority: the target is CPU-first. Do it after 8-10.

### Stage D: product

12. Process isolation hardening beyond task 3 (rlimits, optional Landlock on Linux, Job Object limits on Windows).
13. Approval cards per handbook 5.6, wired to the policy engine's audit log; default focus on deny.
14. Full-screen adaptive TUI and onboarding per handbook 5.3-5.5 and `docs/design/terminal-chat.md`. Put colour tokens in `tui/theme.py`.

### Stage E: research (behind flags, each with a kill criterion in the report)

15. Q4_K reference, then a C-intrinsics kernel library behind `BackendProtocol` with runtime dispatch. Promote only at ≥ 80% of measured STREAM bandwidth and bit-for-bit or tolerance-checked equality with gguf-py.
16. Offline layer pruning experiment (ShortGPT-style). 17. Activation-sparsity experiment (PowerInfer + TurboSparse model, or TEAL-style kernel). Don't build dense-model SSD streaming.

## Design identity, in practice

- Name stays **EigenCore**. Mark v2 is a proposal: don't replace `assets/brand/` until the owner approves; then generate PNG and favicons from the SVG.
- In the terminal use the token table in handbook 5.2 only, through one theme module. Teal = ok/measured/fits, amber = tight/approval/estimate, clay = denied/error/does_not_fit, indigo = brand and focus. Every colour has a text equivalent; `NO_COLOR` and `--plain` must lose no information.
- No splash animation, one spinner at most, reduced-motion respected. Never clear scrollback.

## Benchmarks only the owner can run

You prepare commands and parsers; the owner runs them on his laptop and commits the JSON with the machine named. Full list in FINAL_REPORT section 9. Short form:

1. STREAM or Intel MLC bandwidth at 1, 2, 4, 8 threads.
2. `llama-bench -p 512 -n 128 -t 2,4,8 -r 5 -o json` on a 7-8B Q4_K_M.
3. KV type and flash-attention sweep.
4. A 32B Q4_K_M over RAM with mmap: tok/s, page faults, SSD read rate.
5. A 30B-A3B MoE at Q4_K_M and Q3: tok/s at 4K context.
6. `fio` SSD read profile.
7. `llama-fit-params` and peak RSS vs `estimate_fit()`.
8. EigenCore harness vs `llama-bench`, warm and cold.
9. Prompt-lookup on/off (after task 9).
10. `llama-perplexity` per quant used.

When you finish a stage, update the README status table and `CHANGELOG.md`, and tell the owner which benchmark from this list the stage needs before it can be called done.

Thanks. The bones are good; this is about making every number it prints trustworthy.
