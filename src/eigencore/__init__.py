"""EigenCore — CPU-first LLM intelligence runtime."""

__version__ = "0.4.5"

from eigencore.agent.executor import ToolExecutor, ToolResult
from eigencore.agent.grammar import GBNFBuilder
from eigencore.agent.policy import AuditLog, PolicyEngine, ToolCapability, default_policy
from eigencore.agent.react import AgentResult, AgentStep, ReActAgent, StepKind
from eigencore.agent.tool import Tool, ToolParam, ToolRegistry
from eigencore.benchmark import (
    DEFAULT_CASES,
    BenchmarkReport,
    QualityCase,
    build_report,
    run_cases,
)
from eigencore.context.manager import ContextManager
from eigencore.engine.inference import GenerationConfig, GenerationResult, InferenceEngine
from eigencore.engine.layer_skip import LayerSkipScheduler, SkipPlan, SkipStrategy
from eigencore.engine.sparse_inference import SparseExecutionPlan, SparsityCache, SparsityPredictor
from eigencore.engine.speculative import SpeculativeDecoder, SpeculativeResult
from eigencore.fit import (
    FitEstimate,
    FitVerdict,
    ModelGeometry,
    estimate_fit,
    max_context_for_budget,
)
from eigencore.forge import Forge
from eigencore.hal.profiler import HardwareProfile, profile_hardware
from eigencore.kernels import ScalarBackend, rms_norm
from eigencore.models.registry import ModelRegistry, ModelSpec
from eigencore.router.classifier import RoutingDecision, TaskDomain, TaskRouter

__all__ = [
    "DEFAULT_CASES",
    "AgentResult",
    "AgentStep",
    "AuditLog",
    "BenchmarkReport",
    "ContextManager",
    "FitEstimate",
    "FitVerdict",
    "Forge",
    "GBNFBuilder",
    "GenerationConfig",
    "GenerationResult",
    "HardwareProfile",
    "InferenceEngine",
    "LayerSkipScheduler",
    "ModelGeometry",
    "ModelRegistry",
    "ModelSpec",
    "PolicyEngine",
    "QualityCase",
    "ReActAgent",
    "RoutingDecision",
    "ScalarBackend",
    "SkipPlan",
    "SkipStrategy",
    "SparseExecutionPlan",
    "SparsityCache",
    "SparsityPredictor",
    "SpeculativeDecoder",
    "SpeculativeResult",
    "StepKind",
    "TaskDomain",
    "TaskRouter",
    "Tool",
    "ToolCapability",
    "ToolExecutor",
    "ToolParam",
    "ToolRegistry",
    "ToolResult",
    "build_report",
    "default_policy",
    "estimate_fit",
    "max_context_for_budget",
    "profile_hardware",
    "rms_norm",
    "run_cases",
]
