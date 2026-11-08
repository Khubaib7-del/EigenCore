# Study guide: the memory-bandwidth talk

For the presenter. Read it once end to end, then use sections 6 and 7 the night before. Every number here is arithmetic from published specs or model configs, labelled; none is a benchmark.

---

## 1. The argument from first principles

### 1.1 What a token costs

A transformer generates one token at a time. For each token, every layer multiplies the current hidden vector by its weight matrices (attention projections Q, K, V, O and the feed-forward matrices). For batch size 1 that is a **matrix-vector product**: each weight is read once and used for one multiply and one add, 2 FLOPs.

So per token, a dense model must:

- **read** every weight once: bytes = weight file size (roughly);
- **compute** about 2 × params FLOPs.

### 1.2 Which one runs out first

Compare the work to what the hardware can do in the same second.

- **Arithmetic intensity** of decode = FLOPs / bytes read = 2 / bytes-per-weight. At ~0.56 bytes per weight (4-bit plus scales): 2 / 0.56 ≈ **3.6 FLOP/byte**.
- **Ridge point** of the machine = peak FLOP/s ÷ peak bytes/s. An i7-10610U (4 cores, AVX2 with two FMA units per core, ~3 GHz sustained): 4 × 2 FMA × 8 lanes × 2 FLOPs × 3e9 ≈ 384 GFLOP/s. Memory peak 45.8 GB/s (Intel spec, LPDDR4-2933; 42.7 GB/s with DDR4-2666). Ridge ≈ 384 / 40 ≈ **9.6 FLOP/byte**.
- 3.6 < 9.6, so **decode is bandwidth-bound**: the cores finish their math and wait for the next bytes. (Realistic quantized-dot throughput is lower than peak FMA, so the margin is about 2×, not 3×. Still bandwidth-bound.)

This is the **roofline model**: attainable FLOP/s = min(peak FLOP/s, intensity × bandwidth).

### 1.3 Prefill is different

Prefill processes all N prompt tokens at once: a **matrix-matrix** product. Each weight read is reused N times, so intensity ≈ 3.6 × N. From N ≈ 3 upward, prefill is **compute-bound**. That is why time-to-first-token improves with better kernels and more cores, while decode tok/s mostly doesn't.

### 1.4 The ceiling formula

decode tok/s ≤ sustained bandwidth (bytes/s) ÷ bytes read per token

It is a ceiling: real runs lose to cache misses, dequantization overhead, thread sync, thermal throttling and the KV-cache reads that also grow with context.

## 2. Worked numbers

### 2.1 Weights

bytes = params × bits per weight ÷ 8. Q4_K_M files average ~4.85 bpw → ~0.61 bytes/param.

| Model | F16 (16 bpw) | Q8_0 (8.5 bpw) | Q4_K_M (~4.85 bpw) |
| --- | --- | --- | --- |
| 8B | 16 GB | 8.5 GB | ≈ 4.9 GB |
| 14B | 28 GB | 14.9 GB | ≈ 8.5 GB |
| 35B | 70 GB | 37.2 GB | ≈ 21.2 GB |

Why Q8_0 is 8.5 bpw: blocks of 32 int8 values share one fp16 scale: (32 × 8 + 16) / 32 = 8.5 bits.
Why Q4_0 is 4.5 bpw: 32 four-bit values + one fp16 scale: (32 × 4 + 16) / 32 = 4.5.

### 2.2 Decode ceilings on the i7-10610U (peak 45.8 GB/s) [estimates]

| Model and quant | Bytes per token | Ceiling |
| --- | --- | --- |
| 8B Q4_K_M | ≈ 4.9 GB | ≈ 9 tok/s |
| 14B Q4_K_M | ≈ 8.5 GB | ≈ 5 tok/s |
| 35B Q4_K_M | ≈ 21 GB | ≈ 2 tok/s (if it fit, which it doesn't on 16 GB) |
| 30B MoE, ~3.3B active, Q4_K_M | ≈ 2 GB | ≈ 20+ tok/s if fully in RAM; it needs ~18-19 GB resident, so on 16 GB it pages |

### 2.3 When the model doesn't fit

llama.cpp memory-maps the file by default. Pages not in RAM are read from the SSD on demand. A dense model reads every weight every token, so the part that doesn't fit comes from the SSD every token.

time per token ≈ (resident bytes ÷ RAM bandwidth) + (non-resident bytes ÷ SSD read rate)

Example, 35B Q4_K_M (21 GB) on 16 GB with ~11 GB resident and a 3 GB/s SSD: 11/45.8 + 10/3 ≈ 0.24 + 3.3 ≈ 3.6 s per token → **≈ 0.3 tok/s** [estimate]. Worse in practice if the page cache evicts the wrong pages.

## 3. KV-cache math

During decode, attention at the new token needs the keys and values of every previous token. Storing them (the KV cache) avoids recomputing them.

KV bytes = layers × kv_heads × head_dim × 2 (K and V) × bytes per element × tokens

- **GQA (grouped-query attention):** many query heads share fewer KV heads, which shrinks the cache. Llama 3.1 8B: 32 query heads, 8 KV heads.
- **Llama 3.1 8B** (32 layers, 8 KV heads, head_dim 128, fp16): 32 × 8 × 128 × 2 × 2 = 131,072 B = **128 KiB per token**. 4K → 0.5 GiB; 32K → 4 GiB; 128K → 16 GiB.
- **Qwen2.5-32B-class** (64 layers, 8 KV heads, 128): **256 KiB per token**; 32K → 8 GiB.
- **Quantized KV** (q8_0): 34 bytes per 32 values ≈ 1.06 bytes per element, roughly halving fp16.
- **Why head_dim must come from the file:** some models (Gemma) set head_dim independently of hidden_size / heads. GGUF stores it as `<arch>.attention.key_length`.
- KV is also **read** every token, so long contexts lower decode speed too.

## 4. What each technique does, and its limits

| Technique | What it does | Why it helps | Limits | EigenCore status |
| --- | --- | --- | --- | --- |
| Quantization | Stores weights in fewer bits | Fewer bytes per token and less RAM | Quality loss grows below ~4 bits; measure perplexity | Used via llama.cpp [SHIPPED] |
| KV-cache quantization | Stores K/V in 8 or 4 bits | Longer context in the same RAM | Small quality effect; little tok/s gain | Available in llama-cpp-python, not exposed [PLANNED] |
| Mixture of Experts | A router picks a few expert FFNs per token | Reads only active experts: ~3B of 30B | All experts still need RAM or SSD; routing varies per token | Not in the registry [PLANNED] |
| Speculative decoding | A cheap drafter proposes k tokens; the big model verifies them in one batched pass; correct rejection sampling keeps the big model's output distribution | Several tokens per expensive weight pass | Helps only if acceptance is high and verifying k tokens costs about one; on CPU, reports range from slower to ~1.7× | Module exists, incorrect, not wired [REPORTED-ONLY] |
| Prompt-lookup decoding | Drafts by copying n-grams from the prompt | Free drafter; good for code edits and summaries | Weak on open-ended chat | Available in llama-cpp-python, not used [PLANNED] |
| Layer skipping / pruning | Skip or remove layers judged redundant | Fewer bytes and FLOPs | Per-token skipping needs backend hooks and calibration; pruning needs quality checks, often fine-tuning | Planner only [REPORTED-ONLY] |
| Activation sparsity | Skip FFN neurons that output near zero | Fewer weight rows read | Dense SwiGLU models (Llama, Qwen) aren't naturally sparse; needs ReLU-fied models or special kernels and weight layout | Predictor never fed data [REPORTED-ONLY] |
| SSD streaming | Read weights from disk as needed | Lets oversized models load | SSD is 5-40× slower than RAM; viable for MoE, not dense | Only llama.cpp's default mmap |
| SIMD / assembly kernels | Vector instructions do 8-16 operations per instruction | Faster prefill | Decode is bandwidth-bound; llama.cpp already has tuned AVX2 kernels | Scalar NumPy references only; Q4_0 one has a bug [SHIPPED / PLANNED] |
| GPU offload | Put some layers in VRAM | VRAM bandwidth is much higher | Only what fits in VRAM; PCIe transfers | `n_gpu_layers=0` hard-coded [PLANNED] |

## 5. Glossary

- **GGUF:** llama.cpp's model file format: tensors plus metadata (architecture, layers, heads, tokenizer).
- **Quantization (Q4_0, Q4_K_M, Q8_0):** storing weights in blocks of low-bit integers with shared scales. K-quants use super-blocks with extra scale levels; "_M" mixes in higher precision for sensitive tensors.
- **bpw:** bits per weight, including scales.
- **Prefill:** processing the prompt; compute-bound.
- **Decode:** generating tokens one by one; bandwidth-bound.
- **TTFT:** time to first token; mostly prefill.
- **tok/s:** decode tokens per second.
- **KV cache:** stored keys and values of previous tokens.
- **GQA / MQA:** grouped/multi-query attention: fewer KV heads than query heads.
- **Head dim:** size of one attention head's vectors.
- **Arithmetic intensity:** FLOPs per byte moved.
- **Roofline:** performance = min(compute peak, intensity × bandwidth).
- **Memory-bound / compute-bound:** which roof you hit.
- **SIMD, AVX2, AVX-512, VNNI, AMX, NEON:** CPU vector and matrix instruction sets.
- **FMA:** fused multiply-add, two FLOPs in one instruction.
- **mmap:** mapping a file into memory; the OS loads pages on demand.
- **Page cache:** RAM the OS uses to hold recently read file pages.
- **MoE:** Mixture of Experts.
- **Active parameters:** parameters used per token in an MoE model.
- **Speculative decoding, draft model, acceptance rate:** see section 4.
- **Perplexity:** exp(average cross-entropy); lower is better; used to measure quality loss.
- **ReAct:** reason-and-act loop where the model alternates thoughts and tool calls.
- **GBNF:** llama.cpp's grammar format for constraining output.
- **OS sandbox:** isolation enforced by the operating system (separate process, restricted files, network, resources), as opposed to checks inside the same program.

## 6. Hard questions, honest answers

**1. "What is yours if llama.cpp does the inference?"**
The judgement around it: the hardware profile, the memory-fit estimator, the benchmark harness with model hashes and quality checks, the tool policy layer, and the terminal UI that keeps the fit verdict on screen. The transformer math is llama.cpp's and I say so on slide 7. Owning kernels is a research track behind a backend interface; we'll only route inference through our own code if it matches llama.cpp's output and measures faster.

**2. "So can it run a 35B model on 16 GB or not?"**
Dense at 4-bit: the weights alone are about 21 GB, so it doesn't fit. It will load because llama.cpp memory-maps the file, but then it reads from the SSD every token, which by our arithmetic is well under one token per second. A 30B Mixture-of-Experts model, which reads about 3B parameters per token, is the realistic route, and we haven't measured it yet.

**3. "Where are your benchmarks?"**
There are none published yet, deliberately. The harness exists; the first runs will be on this laptop and labelled with the machine. The 2 tok/s on slide 6 is a ceiling from the spec sheet, not a result.

**4. "Isn't more cores / AVX-512 the fix?"**
Not for decode. At about 3.6 FLOPs per byte, the cores already outrun memory. More cores help prefill and help reach the bandwidth limit, but they don't raise it.

**5. "Why not just use a GPU?"**
GPU VRAM bandwidth is far higher, so it helps a lot, for the layers that fit in VRAM. Laptops in this class often have none or a small one. EigenCore hard-codes CPU today; GPU fit and offload are planned.

**6. "Your earlier slides said EigenCore does layer skipping and sparsity."**
They exist as planners that report numbers, and our own audit test proves none of it reaches llama.cpp. That's why they're labelled reported-only. I'd rather show that than a speed-up we can't back.

**7. "Isn't 70-90% of a transformer's neurons inactive anyway?"**
That's true for older ReLU models. Modern models use SwiGLU, which is much less sparse, and exploiting sparsity on a CPU needs a predictor and a different weight layout, not just skipping multiplies. MoE models are the version of this idea that's trained in.

**8. "Speculative decoding changes the output, doesn't it?"**
Done correctly, with rejection sampling, it provably keeps the large model's output distribution. Our current module doesn't implement that correctly, which is one reason it isn't wired in. The first version we'd ship is prompt-lookup decoding through llama-cpp-python, tested to give identical greedy output.

**9. "Why is the KV cache so big? I thought weights were the problem."**
Weights are fixed; KV grows with every token. For an 8B model, 128K tokens of context is 16 GiB of cache, as much as the whole laptop.

**10. "How do you know the bandwidth number?"**
45.8 GB/s is Intel's published maximum for the i7-10610U with LPDDR4-2933. Sustained bandwidth is lower and has to be measured with STREAM or Intel MLC; EigenCore's own probe is single-threaded and reads low, which is a known fix.

**11. "Is the tool sandbox safe?"**
It's a policy layer, not a sandbox: tools run inside the same Python process. It also isn't on by default in the agent API yet. Process isolation and approvals are planned; I wouldn't give it shell access today.

**12. "Why Python if speed matters?"**
Python only orchestrates. The per-token work runs in llama.cpp's C/C++. Python's cost is per request, not per weight.

**13. "Why does quantization not destroy quality?"**
Weights are clustered and errors average out over large dot products; K-quants keep scales per small block and higher precision for sensitive tensors. It still costs some quality, more below 4 bits, which is why we'd measure perplexity before recommending a quant.

**14. "What would you do with another month?"**
Fix the fit estimator to read real geometry, make the tool policy default-on, run the benchmarks on this laptop, and test a 30B MoE model with prompt-lookup decoding. Kernels after that.

## 7. One-page cheat sheet

- Decode: read all weights per token → bandwidth-bound (≈ 3.6 FLOP/B vs ridge ≈ 9.6).
- Ceiling: tok/s ≤ bandwidth ÷ bytes per token. 45.8 ÷ 21 ≈ 2.
- Weights: params × bpw ÷ 8. Q4_K_M ≈ 0.61 B/param. Q8_0 = 8.5 bpw. Q4_0 = 4.5 bpw.
- KV: layers × kv_heads × head_dim × 2 × bytes × tokens. Llama 3.1 8B = 128 KiB/token.
- 35B Q4_K_M ≈ 21 GB > 16 GB → pages from SSD → < 1 tok/s (estimate).
- MoE 30B-A3B ≈ 2 GB per token → the realistic route; unmeasured.
- EigenCore owns: profile, fit, benchmark, policy, UI. llama.cpp owns: the math.
- REPORTED-ONLY: layer skip, sparsity, speculative. No numbers until measured.
