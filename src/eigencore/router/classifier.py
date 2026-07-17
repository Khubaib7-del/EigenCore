"""
Task router — classifies prompts into task domains and selects the optimal
specialist model. This is the application-level MoE: instead of routing within
one model's layers, we route between multiple small models.

Phase 1 uses keyword-based classification (zero overhead, no model needed).
Phase 2 will replace this with a tiny learned classifier (~100M params).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from eigencore.hal.profiler import HardwareProfile
from eigencore.models.registry import ModelRegistry, ModelSpec


class TaskDomain(Enum):
    GENERAL = auto()
    CODE = auto()
    MATH = auto()
    CREATIVE = auto()
    ANALYSIS = auto()
    SUMMARIZATION = auto()


@dataclass
class RoutingDecision:
    domain: TaskDomain
    confidence: float
    model: ModelSpec
    reason: str
    matched_signals: list[str] = field(default_factory=list)


# keyword signals per domain — ordered by specificity (most specific first)
_DOMAIN_SIGNALS: dict[TaskDomain, list[str]] = {
    TaskDomain.CODE: [
        r"\b(code|function|class|method|implement|debug|refactor|compile|syntax)\b",
        r"\b(python|javascript|rust|c\+\+|java|typescript|html|css|sql|api)\b",
        r"\b(bug|error|exception|traceback|stack\s*trace|segfault|runtime)\b",
        r"\b(git|commit|branch|merge|pull\s*request|repo|npm|pip|cargo)\b",
        r"\b(algorithm|data\s*structure|binary\s*tree|linked\s*list|hash\s*map)\b",
        r"```",
    ],
    TaskDomain.MATH: [
        r"\b(calculate|compute|solve|equation|formula|integral|derivative)\b",
        r"\b(matrix|vector|eigenvalue|determinant|linear\s*algebra)\b",
        r"\b(probability|statistics|mean|median|variance|standard\s*deviation)\b",
        r"\b(proof|theorem|lemma|conjecture|axiom)\b",
        r"[=+\-*/^]{2,}",
        r"\d+\s*[+\-*/^]\s*\d+",
    ],
    TaskDomain.CREATIVE: [
        r"\b(write|poem|story|essay|letter|narrative|fiction|creative)\b",
        r"\b(metaphor|analogy|imagery|tone|voice|style|prose|verse)\b",
        r"\b(blog\s*post|article|content|copywriting|tagline|slogan)\b",
    ],
    TaskDomain.ANALYSIS: [
        r"\b(analyze|compare|evaluate|assess|review|critique|examine)\b",
        r"\b(pros?\s*(?:and|&)\s*cons?|trade\s*offs?|advantages?|disadvantages?)\b",
        r"\b(research|study|findings|evidence|data|trends?|patterns?)\b",
    ],
    TaskDomain.SUMMARIZATION: [
        r"\b(summarize|summary|tldr|tl;dr|brief|overview|recap|condense)\b",
        r"\b(key\s*points?|main\s*ideas?|highlights?|takeaways?)\b",
    ],
}


class TaskRouter:
    """
    Routes prompts to the optimal specialist model based on task domain.

    Current implementation: keyword-based classification with confidence scoring.
    Future: replace with a tiny learned classifier that stays resident in memory.
    """

    def __init__(
        self,
        profile: HardwareProfile,
        registry: Optional[ModelRegistry] = None,
    ):
        self.profile = profile
        self.registry = registry or ModelRegistry()
        self._compiled_patterns: dict[TaskDomain, list[re.Pattern]] = {
            domain: [re.compile(p, re.IGNORECASE) for p in patterns]
            for domain, patterns in _DOMAIN_SIGNALS.items()
        }

    def classify(self, prompt: str) -> tuple[TaskDomain, float, list[str]]:
        """
        Classify a prompt into a task domain.
        Returns (domain, confidence, matched_signals).
        """
        scores: dict[TaskDomain, float] = {}
        matches: dict[TaskDomain, list[str]] = {}

        for domain, patterns in self._compiled_patterns.items():
            domain_matches = []
            for pattern in patterns:
                found = pattern.findall(prompt)
                if found:
                    domain_matches.extend(found)

            if domain_matches:
                # score = number of unique pattern matches / total patterns for this domain
                unique_matches = len(set(domain_matches))
                scores[domain] = unique_matches / len(patterns)
                matches[domain] = list(set(domain_matches))

        if not scores:
            return TaskDomain.GENERAL, 0.5, []

        best_domain = max(scores, key=scores.get)  # type: ignore[arg-type]
        confidence = min(scores[best_domain], 1.0)

        # boost confidence if multiple patterns matched
        if len(matches.get(best_domain, [])) >= 3:
            confidence = min(confidence + 0.2, 1.0)

        return best_domain, confidence, matches.get(best_domain, [])

    def route(self, prompt: str) -> RoutingDecision:
        """
        Classify the prompt and select the optimal model for the detected task.
        """
        domain, confidence, matched = self.classify(prompt)

        # map domain to model task type
        task_map = {
            TaskDomain.GENERAL: "general",
            TaskDomain.CODE: "code",
            TaskDomain.MATH: "general",  # no dedicated math model in registry yet
            TaskDomain.CREATIVE: "general",
            TaskDomain.ANALYSIS: "general",
            TaskDomain.SUMMARIZATION: "general",
        }

        task = task_map[domain]
        model = self.registry.recommend(self.profile, task)

        reason = f"Detected {domain.name.lower()} task"
        if matched:
            reason += f" (signals: {', '.join(matched[:3])})"
        if confidence < 0.3:
            reason += " — low confidence, using general model"

        return RoutingDecision(
            domain=domain,
            confidence=confidence,
            model=model,
            reason=reason,
            matched_signals=matched,
        )

    def should_swap(self, current_model: ModelSpec, new_decision: RoutingDecision) -> bool:
        """
        Decide whether to swap models based on routing decision.
        Avoids unnecessary swaps for marginal domain changes.
        """
        if current_model.name == new_decision.model.name:
            return False

        # only swap if confidence is high enough to justify the 2-3s load time
        if new_decision.confidence < 0.4:
            return False

        # don't swap between two general models
        if current_model.task == "general" and new_decision.model.task == "general":
            return False

        return True
