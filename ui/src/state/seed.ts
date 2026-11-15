import type { Chat, TpsPoint, Turn } from "./store";

/**
 * Seeded conversation used by the preview. Per-message speed stats are left
 * empty (shown as "— tok/s") because nothing was measured; token counts are
 * just counts of the sample text.
 */
export const SEED_CHATS: Chat[] = [
  {
    id: "c1",
    title: "Explain KV cache quantization",
    group: "Today",
    model: "qwen2.5-1.5b",
    messages: [
      {
        id: "m1",
        role: "user",
        content: "What does a q8_0 KV cache change compared to fp16? Keep it short.",
      },
      {
        id: "m2",
        role: "assistant",
        model: "qwen2.5-1.5b",
        status: "done",
        statsKind: "placeholder",
        stats: { tokens: 168, seconds: 0, tok_per_sec: 0, interrupted: false, error: null, first_token_s: null },
        content:
          "<think>Compare memory per token and the quality trade-off. Mention that the cache grows with context length, so the saving matters most for long prompts.</think>" +
          "The KV cache stores keys and values for every token in the context, so it grows with context length.\n\n" +
          "- **fp16** keeps 16 bits per value. Highest fidelity, most memory.\n" +
          "- **q8_0** stores 8-bit blocks with a scale. Roughly **half the cache memory**, usually with a very small quality cost.\n\n" +
          "Switch it in Settings → Engine. Use `q8_0` when you want longer context on limited RAM:\n\n" +
          "```bash\neigencore chat --model qwen2.5-1.5b --kv-cache q8_0 --ctx 8192\n```",
      },
      {
        id: "m3",
        role: "user",
        content: "Can you read ~/Documents/notes.txt and list the top words?",
      },
      {
        id: "m4",
        role: "assistant",
        model: "qwen2.5-1.5b",
        status: "awaiting_approval",
        statsKind: "placeholder",
        content: "",
        approval: {
          id: "seed-apr",
          tool_name: "read_file",
          capability: "READ",
          args: { path: "~/Documents/notes.txt", encoding: "utf-8", api_key: "***REDACTED***" },
          reason: "Read the file you mentioned to answer the question.",
        },
        tools: [
          { tool_name: "list_dir", summary: "~/Documents · 14 entries", ok: true },
          { tool_name: "search_files", summary: "notes*.txt · 1 match", ok: true },
        ],
      },
    ],
  },
  { id: "c2", title: "Word counter in Python", group: "Today", model: "qwen2.5-coder-1.5b", messages: [] },
  { id: "c3", title: "Summarise meeting notes", group: "Yesterday", model: "qwen2.5-1.5b", messages: [] },
  { id: "c4", title: "Regex for ISO dates", group: "Yesterday", model: "qwen2.5-coder-1.5b", messages: [] },
  { id: "c5", title: "Which model fits 8 GB RAM?", group: "Previous 7 days", model: "tinyllama-1.1b", messages: [] },
  { id: "c6", title: "Translate a README to Urdu", group: "Previous 7 days", model: "qwen2.5-1.5b", messages: [] },
  { id: "c7", title: "Draft release notes", group: "Previous 7 days", model: "qwen2.5-7b", messages: [] },
];

/**
 * MOCK telemetry for the Monitor preview. Clearly labelled "Mock telemetry"
 * on screen. These are placeholders for layout, not measurements.
 */
export const SEED_TURNS: Turn[] = [
  { id: "t1", at: "14:02", model: "qwen2.5-1.5b", prompt_tokens: 212, tokens: 168, tok_per_sec: 21.4, first_token_s: 0.42, seconds: 7.9, status: "done" },
  { id: "t2", at: "13:58", model: "qwen2.5-1.5b", prompt_tokens: 96, tokens: 304, tok_per_sec: 22.1, first_token_s: 0.31, seconds: 13.8, status: "done" },
  { id: "t3", at: "13:51", model: "qwen2.5-coder-1.5b", prompt_tokens: 441, tokens: 512, tok_per_sec: 19.6, first_token_s: 0.88, seconds: 26.1, status: "done" },
  { id: "t4", at: "13:47", model: "qwen2.5-coder-1.5b", prompt_tokens: 120, tokens: 87, tok_per_sec: 20.3, first_token_s: 0.36, seconds: 4.3, status: "stopped" },
  { id: "t5", at: "13:40", model: "tinyllama-1.1b", prompt_tokens: 58, tokens: 233, tok_per_sec: 34.7, first_token_s: 0.19, seconds: 6.7, status: "done" },
];

export const SEED_TPS: number[] = [
  0, 6, 14, 19, 21, 22, 21.5, 22.4, 21.8, 22.6, 21.9, 22.2, 0, 0, 5, 12, 18, 19.8, 20.4, 19.9, 20.1, 19.2, 19.7, 0, 0, 9, 16,
  20.5, 21.6, 21.9, 22.3, 21.7, 21.2, 21.8,
];

/** MOCK live samples for Monitor's "This reply" view (preview only, labelled). */
export const SEED_LIVE_TPS: TpsPoint[] = [
  2.1, 14.2, 19.8, 21.4, 22.6, 22.1, 23.0, 22.4, 21.7, 22.9, 23.3, 22.8, 21.9, 22.5, 23.1, 22.7, 22.2, 23.4,
].map((rate, i) => ({ t: (i + 1) * 0.5, rate }));

const STREAM_REPLY =
  "Here is a small function that walks a folder and prints it as a tree, the way `tree` does on Linux:\n\n" +
  "```python\nfrom pathlib import Path\n\n\ndef print_tree(root: Path, prefix: str = \"\") -> None:\n" +
  "    entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))\n" +
  "    for i, entry in enumerate(entries):\n        last = i == len(entries) - 1\n" +
  "        print(prefix + (\"└── \" if last else \"├── \") + entry.name)\n        if entry.is_dir():\n" +
  "            print_tree(entry, prefix + (\"    \" if last else \"│   \"))";

const DONE_REPLY =
  "The desktop UI lives in `ui/src`. The parts you will touch most:\n\n" +
  "- ui/src/\n  - api/\n    - bridge.ts\n    - mock.ts\n  - components/\n    - CodeBlock.tsx\n    - PromptBar.tsx\n  - screens/\n    - ChatScreen.tsx\n    - MonitorScreen.tsx\n  - main.tsx\n\n" +
  "Events from Python arrive as tuples and are turned into typed events in one place:\n\n" +
  "```ts\nwindow.addEventListener(\"eigencore\", (ev) => {\n  for (const t of (ev as CustomEvent<unknown[][]>).detail) {\n    const e = fromTuple(t); // null for unknown kinds\n    if (e) listeners.forEach((l) => l(e));\n  }\n});\n```\n\n" +
  "Start with **bridge.ts** if you add a new backend event.";

/**
 * Preview-only chats for `?chat=thinking|streaming|done` (screenshots and
 * reviews). Every number here is MOCK and shown with a "mock stream" chip.
 */
export function seedChatFor(mode: string): Chat | null {
  const now = Date.now();
  const user = (id: string, content: string) => ({ id, role: "user" as const, content });
  if (mode === "thinking")
    return {
      id: "demo-thinking",
      title: "Refactor the event pump",
      group: "Today",
      model: "qwen2.5-1.5b",
      messages: [
        user("d1", "Refactor the event pump in web.py so batches never exceed 20 ms. Explain the change first."),
        { id: "d2", role: "assistant", model: "qwen2.5-1.5b", content: "", status: "reading", startedAt: now - 2300, budget: 512 },
      ],
    };
  if (mode === "streaming")
    return {
      id: "demo-stream",
      title: "Print a folder as a tree",
      group: "Today",
      model: "qwen2.5-coder-1.5b",
      messages: [
        user("d1", "Write a Python function that prints a folder like the tree command."),
        { id: "d2", role: "assistant", model: "qwen2.5-coder-1.5b", content: STREAM_REPLY, status: "streaming", startedAt: now - 6400, budget: 512 },
      ],
    };
  if (mode === "done")
    return {
      id: "demo-done",
      title: "Where does the UI live?",
      group: "Today",
      model: "qwen2.5-1.5b",
      messages: [
        user("d1", "Where does the desktop UI live, and where do backend events come in?"),
        {
          id: "d2",
          role: "assistant",
          model: "qwen2.5-1.5b",
          content: DONE_REPLY,
          status: "done",
          statsKind: "mock",
          budget: 512,
          stats: { tokens: 214, seconds: 9.8, tok_per_sec: 21.8, decode_tok_per_sec: 23.4, first_token_s: 0.41, interrupted: false, error: null },
        },
      ],
    };
  return null;
}
