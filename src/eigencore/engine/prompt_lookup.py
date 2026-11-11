"""
Prompt-lookup decoding (PLD) - Stage C, behind a flag, default OFF.

PLD (prompt lookup / SpecInfer-style): when the model is about to emit a span
that already appears verbatim in the prompt (common in code, quotes,
boilerplate), copy it directly from the prompt instead of running the model -
zero backend work for those tokens.

This module is a pure, backend-independent matcher: given the prompt text and
the tokens generated so far, find the longest prompt substring continuing the
generation. The engine uses it only when prompt_lookup=True.

Correctness rule: a copied span must be byte-identical to the prompt span -
no fuzzy matching, no paraphrase - so output quality is unaffected by
construction. Speedup is measured, never assumed.
"""

from __future__ import annotations


def longest_prompt_continuation(prompt: str, generated: str, max_span: int = 512) -> str:
    """Longest prompt text that continues the current generation.

    Finds the longest suffix of `generated` that occurs in `prompt` with at
    least one more prompt character after it, and returns the prompt text
    from that point (up to max_span chars). Empty string when nothing matches.
    """
    if not prompt or not generated:
        return ""
    gen = generated[-max_span:]
    for size in range(min(len(gen), max_span), 0, -1):
        suffix = gen[-size:]
        idx = prompt.find(suffix)
        while idx != -1:
            end = idx + size
            if end < len(prompt):
                return prompt[end : end + max_span]
            idx = prompt.find(suffix, idx + 1)
    return ""
