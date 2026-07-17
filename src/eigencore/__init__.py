"""EigenCore — CPU-first LLM intelligence runtime."""

__version__ = "0.1.0"

from eigencore.forge import Forge
from eigencore.hal.profiler import HardwareProfile, profile_hardware
from eigencore.models.registry import ModelRegistry, ModelSpec
from eigencore.engine.inference import InferenceEngine, GenerationConfig, GenerationResult
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
    "TaskRouter",
    "TaskDomain",
    "RoutingDecision",
    "ContextManager",
]
