"""EigenCore — CPU-first LLM intelligence runtime."""

__version__ = "0.2.0"

from eigencore.context.manager import ContextManager
from eigencore.engine.inference import GenerationConfig, GenerationResult, InferenceEngine
from eigencore.engine.layer_skip import LayerSkipScheduler, SkipPlan, SkipStrategy
from eigencore.engine.sparse_inference import SparseExecutionPlan, SparsityCache, SparsityPredictor
from eigencore.engine.speculative import SpeculativeDecoder, SpeculativeResult
from eigencore.forge import Forge
from eigencore.hal.profiler import HardwareProfile, profile_hardware
from eigencore.models.registry import ModelRegistry, ModelSpec
from eigencore.router.classifier import RoutingDecision, TaskDomain, TaskRouter

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
