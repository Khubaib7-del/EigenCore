import type { HardwareProfile, ModelRow, Settings } from "./types";

/**
 * Registry rows copied from src/eigencore/models/registry.py (names, sizes,
 * quant, context, descriptions). Download/fit/active state is MOCK.
 * Fit is "unknown" for anything not downloaded, exactly like the backend
 * (no GGUF geometry until the file is on disk).
 */
export const MOCK_MODELS: ModelRow[] = [
  {
    name: "tinyllama-1.1b",
    params_b: 1.1,
    quant: "Q4_K_M",
    size_gb: 0.67,
    task: "general",
    context_length: 2048,
    description: "Tiny general-purpose model. Fast on any CPU.",
    downloaded: true,
    fit: "fits_with_margin",
    is_active: false,
    repo_id: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    provider: "tinyllama",
  },
  {
    name: "phi-3-mini",
    params_b: 3.8,
    quant: "Q4_K_M",
    size_gb: 2.4,
    task: "general",
    context_length: 4096,
    description: "Strong 3.8B model. Good reasoning for its size.",
    downloaded: false,
    fit: "unknown",
    is_active: false,
    repo_id: "bartowski/Phi-3-mini-4k-instruct-GGUF",
    provider: "microsoft",
  },
  {
    name: "qwen2.5-1.5b",
    params_b: 1.5,
    quant: "Q4_K_M",
    size_gb: 1.0,
    task: "general",
    context_length: 4096,
    description: "Efficient 1.5B model with strong multilingual support.",
    downloaded: true,
    fit: "fits_with_margin",
    is_active: true,
    repo_id: "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    provider: "qwen",
  },
  {
    name: "qwen2.5-3b",
    params_b: 3.0,
    quant: "Q4_K_M",
    size_gb: 1.8,
    task: "general",
    context_length: 4096,
    description: "Balanced 3B model. Good general reasoning.",
    downloaded: false,
    fit: "unknown",
    is_active: false,
    repo_id: "Qwen/Qwen2.5-3B-Instruct-GGUF",
    provider: "qwen",
  },
  {
    name: "qwen2.5-coder-1.5b",
    params_b: 1.5,
    quant: "Q4_K_M",
    size_gb: 1.0,
    task: "code",
    context_length: 4096,
    description: "Code-specialized 1.5B model.",
    downloaded: true,
    fit: "fits_with_margin",
    is_active: false,
    repo_id: "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
    provider: "qwen",
  },
  {
    name: "qwen2.5-7b",
    params_b: 7.0,
    quant: "Q4_K_M",
    size_gb: 4.7,
    task: "general",
    context_length: 4096,
    description: "Large 7B model. Best quality but needs 8GB+ free RAM.",
    downloaded: true,
    fit: "tight",
    is_active: false,
    repo_id: "Qwen/Qwen2.5-7B-Instruct-GGUF",
    provider: "qwen",
  },
  {
    name: "qwen3-30b-a3b",
    params_b: 30.5,
    active_params_b: 3.3,
    quant: "Q4_K_M",
    size_gb: 18.6,
    task: "general",
    context_length: 32768,
    description:
      "MoE 30B (3.3B active/token). Decode is bandwidth-bound on active weights; the full file must be resident or mmap-paged.",
    downloaded: false,
    fit: "unknown",
    is_active: false,
    repo_id: "Qwen/Qwen3-30B-A3B-GGUF",
    provider: "qwen",
  },
];

/** Sample machine (the owner's example laptop). RAM free is mock. */
export const MOCK_PROFILE: HardwareProfile = {
  cpu_name: "Intel Core i7-10610U",
  arch: "x86_64",
  physical_cores: 4,
  logical_cores: 8,
  instruction_sets: ["SSE4.2", "AVX", "AVX2", "FMA"],
  total_ram_gb: 15.8,
  available_ram_gb: 9.2,
  estimated_bandwidth_gbps: null,
  max_model_params_b: 7,
  optimal_quantization: "Q4_K_M",
  recommended_threads: 4,
  recommended_context_length: 4096,
  gpu_name: null,
  benchmarked: false,
};

export const DEFAULT_SETTINGS: Settings = {
  theme: "dark",
  models_dir: "~/.eigencore/models",
  threads: "auto",
  context_length: 4096,
  kv_cache_type: "f16",
  approvals: "ask",
};

export const CURRENT_VERSION = "0.4.5";
/** Mock "newer release" tag used only to demo the updater states. */
export const MOCK_NEXT_TAG = "v0.4.6";

/** The canned reply the mock streams. Qwen3-style <think> block first. */
export const MOCK_REPLY = `<think>The user wants a quick way to count words in a file. A short Python function with collections.Counter is clearest, and I should mention lower-casing and stripping punctuation.</think>Here's a small, dependency-free way to count the most common words in a text file:

\`\`\`python
import re
from collections import Counter

def top_words(path: str, n: int = 10) -> list[tuple[str, int]]:
    with open(path, encoding="utf-8") as f:
        words = re.findall(r"[a-z']+", f.read().lower())
    return Counter(words).most_common(n)

print(top_words("notes.txt"))
\`\`\`

A few notes:

- **Lower-casing** makes "The" and "the" count as one word.
- The regex keeps apostrophes, so "don't" stays whole.
- For very large files, read line by line and call \`Counter.update\` instead.`;
