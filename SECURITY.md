# Security policy

## Supported versions

EigenCore is pre-alpha. Only the latest commit on `main` receives fixes.

## Reporting a vulnerability

Please report privately, not in a public issue:

1. Preferred: GitHub's **Report a vulnerability** button on the repository's Security tab (private vulnerability reporting), if enabled.
2. Otherwise: email the maintainer at the address listed in `pyproject.toml`, with "EigenCore security" in the subject.

Include what you found, how to reproduce it, and the impact you expect. You'll get an acknowledgement within 7 days. Please allow time for a fix before public disclosure.

## Known limitations (not vulnerabilities to report, but read before relying on EigenCore)

- **Agent tool handlers run in the same Python process.** The policy layer (`agent/policy.py`) checks capabilities and some path arguments; it is **not an OS sandbox**. It is also opt-in: `Forge.agent()` does not attach a policy by default yet.
- **The tool timeout does not stop a running handler.**
- **Model files and model output are untrusted input.** GGUF parsing is done by llama.cpp. The chat UI does not yet strip terminal escape sequences from model output.
- **Downloads** come from Hugging Face via `huggingface_hub`; EigenCore records SHA-256 in benchmark reports but does not verify downloads against a published hash.

Fixes for these are planned; see section 8 of [the review](docs/review/FINAL_REPORT.md). Reports that show a concrete exploit of them are still welcome.
