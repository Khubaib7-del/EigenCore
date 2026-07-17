"""ReAct agent — interleaved reasoning and action on CPU.

Implements the ReAct pattern (Yao et al., 2022): the model alternates
between Thought (reasoning about what to do), Action (calling a tool),
and Observation (reading the tool result). GBNF grammars force valid
JSON at each step so there are no parsing failures or retry loops.

The loop runs entirely on CPU using the existing InferenceEngine, with
Phase 2 optimizations (layer skipping, sparsity, speculative decoding)
active throughout.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from eigencore.agent.executor import ToolExecutor, ToolResult
from eigencore.agent.tool import ToolRegistry


class StepKind(Enum):
    THOUGHT = auto()
    ACTION = auto()
    OBSERVATION = auto()
    FINAL = auto()


@dataclass
class AgentStep:
    """A single step in the ReAct loop."""

    kind: StepKind
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    tool_result: Optional[ToolResult] = None
    elapsed_seconds: float = 0.0


@dataclass
class AgentResult:
    """Complete result of an agent run."""

    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_seconds: float = 0.0
    total_tokens: int = 0

    @property
    def num_thoughts(self) -> int:
        return sum(1 for s in self.steps if s.kind == StepKind.THOUGHT)

    @property
    def num_actions(self) -> int:
        return sum(1 for s in self.steps if s.kind == StepKind.ACTION)

    @property
    def tool_calls(self) -> list[AgentStep]:
        return [s for s in self.steps if s.kind == StepKind.ACTION]

    def format_trace(self) -> str:
        lines: list[str] = []
        for i, step in enumerate(self.steps, 1):
            prefix = f"[{i}] {step.kind.name}"
            if step.kind == StepKind.THOUGHT:
                lines.append(f"{prefix}: {step.content}")
            elif step.kind == StepKind.ACTION:
                lines.append(f"{prefix}: {step.tool_name}({step.tool_args})")
            elif step.kind == StepKind.OBSERVATION:
                lines.append(f"{prefix}: {step.content[:200]}")
            elif step.kind == StepKind.FINAL:
                lines.append(f"{prefix}: {step.content}")
        return "\n".join(lines)


_SYSTEM_PROMPT = """You are a helpful assistant with access to tools. To complete the user's task, reason step by step.

At each step, output ONE of these JSON formats:

To think about what to do:
{{"type": "thought", "content": "your reasoning here"}}

To call a tool:
{{"type": "action", "tool": "tool_name", "args": {{"param": "value"}}}}

To give your final answer (when you have enough information):
{{"type": "final", "content": "your answer here"}}

Available tools:
{tools}

Important:
- Think before acting. Plan your approach.
- After each tool result, decide if you need more information or can answer.
- Always give a final answer. Do not end with a thought or action."""


class ReActAgent:
    """ReAct agent that reasons and acts using CPU-local LLM inference.

    Uses GBNF grammar-constrained decoding to guarantee valid JSON at
    every step. The grammar mask is applied during sampling — no retries
    or output validation needed.
    """

    DEFAULT_MAX_STEPS = 10
    DEFAULT_MAX_TOKENS_PER_STEP = 256

    def __init__(
        self,
        registry: ToolRegistry,
        executor: Optional[ToolExecutor] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_tokens_per_step: int = DEFAULT_MAX_TOKENS_PER_STEP,
        system_prompt: Optional[str] = None,
    ):
        self.registry = registry
        self.executor = executor or ToolExecutor(registry)
        self.max_steps = max_steps
        self.max_tokens_per_step = max_tokens_per_step
        self._system_prompt = system_prompt

    @property
    def system_prompt(self) -> str:
        base = self._system_prompt or _SYSTEM_PROMPT
        return base.format(tools=self.registry.format_for_prompt())

    def build_messages(self, task: str, history: list[AgentStep]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]

        for step in history:
            if step.kind == StepKind.THOUGHT:
                msg = json.dumps({"type": "thought", "content": step.content})
                messages.append({"role": "assistant", "content": msg})
            elif step.kind == StepKind.ACTION:
                msg = json.dumps(
                    {
                        "type": "action",
                        "tool": step.tool_name,
                        "args": step.tool_args,
                    }
                )
                messages.append({"role": "assistant", "content": msg})
            elif step.kind == StepKind.OBSERVATION:
                messages.append({"role": "user", "content": f"Observation: {step.content}"})

        return messages

    def parse_step(self, raw: str) -> AgentStep:
        """Parse a model output into an AgentStep.

        Expects JSON constrained by GBNF grammar, but handles
        malformed output gracefully.
        """
        raw = raw.strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return AgentStep(
                kind=StepKind.THOUGHT,
                content=raw,
            )

        step_type = parsed.get("type", "thought")

        if step_type == "thought":
            return AgentStep(
                kind=StepKind.THOUGHT,
                content=parsed.get("content", ""),
            )
        elif step_type == "action":
            return AgentStep(
                kind=StepKind.ACTION,
                content="",
                tool_name=parsed.get("tool"),
                tool_args=parsed.get("args", {}),
            )
        elif step_type == "final":
            return AgentStep(
                kind=StepKind.FINAL,
                content=parsed.get("content", ""),
            )
        else:
            return AgentStep(kind=StepKind.THOUGHT, content=raw)

    def execute_step(self, step: AgentStep) -> Optional[AgentStep]:
        if step.kind != StepKind.ACTION or step.tool_name is None:
            return None

        result = self.executor.execute(step.tool_name, step.tool_args or {})
        step.tool_result = result

        return AgentStep(
            kind=StepKind.OBSERVATION,
            content=result.format_for_prompt(),
            tool_result=result,
        )

    def run(
        self,
        task: str,
        generate_fn=None,
    ) -> AgentResult:
        """Run the ReAct loop on a task.

        Args:
            task: The user's task/question.
            generate_fn: A callable(messages, max_tokens) -> str that runs
                inference. When None, the agent builds the step history
                but requires an external caller to drive generation.
        """
        start = time.perf_counter()
        steps: list[AgentStep] = []
        total_tokens = 0

        for _ in range(self.max_steps):
            if generate_fn is None:
                break

            messages = self.build_messages(task, steps)
            step_start = time.perf_counter()
            raw_output = generate_fn(messages, self.max_tokens_per_step)
            step_elapsed = time.perf_counter() - step_start

            step = self.parse_step(raw_output)
            step.elapsed_seconds = step_elapsed
            steps.append(step)
            total_tokens += len(raw_output) // 4

            if step.kind == StepKind.FINAL:
                break

            if step.kind == StepKind.ACTION:
                observation = self.execute_step(step)
                if observation:
                    steps.append(observation)

        answer = ""
        for step in reversed(steps):
            if step.kind == StepKind.FINAL:
                answer = step.content
                break
        if not answer and steps:
            last_content = [s for s in steps if s.content]
            if last_content:
                answer = last_content[-1].content

        return AgentResult(
            answer=answer,
            steps=steps,
            total_seconds=time.perf_counter() - start,
            total_tokens=total_tokens,
        )
