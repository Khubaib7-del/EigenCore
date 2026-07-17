"""EigenCore — CPU-first LLM intelligence runtime."""

__version__ = "0.2.0"

from eigencore.forge import Forge
from eigencore.hal.profiler import HardwareProfile, profile_hardware
from eigencore.models.registry import ModelRegistry, ModelSpec
from eigencore.engine.inference import InferenceEngine, GenerationConfig, GenerationResult
from eigencore.engine.layer_skip import LayerSkipScheduler, SkipPlan, SkipStrategy
from eigencore.engine.speculative import SpeculativeDecoder, SpeculativeResult
from eigencore.engine.sparse_inference import SparsityPredictor, SparseExecutionPlan, SparsityCache
from eigencore.router.classifier import TaskRouter, TaskDomain, RoutingDecision
from eigencore.context.manager import ContextManager

__all__ = [
    "Forge",
    "HardwareProfile",
    "profile_hardware",
    "ModelRegistry",
    "ModelSpec",
    "InferenceEngine",
    "GenerationConfig",
    "GenerationResult",
    "LayerSkipScheduler",
    "SkipPlan",
    "SkipStrategy",
    "SpeculativeDecoder",
    "SpeculativeResult",
    "SparsityPredictor",
    "SparseExecutionPlan",
    "SparsityCache",
    "TaskRouter",
    "TaskDomain",
    "RoutingDecision",
    "ContextManager",
]
