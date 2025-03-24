"""
Noesis - A modular pipeline for processing text with LLMs.

This package provides a simple interface for configuring and running
modular processing pipelines with LLMs.
"""

from .pipeline import Pipeline, PipelineStep, LLMStep
from .api import Noesis, create_pipeline, process_text
from .extensions import RainbirdStep

__version__ = "0.1.0"

__all__ = [
    "Noesis",
    "Pipeline",
    "PipelineStep", 
    "LLMStep",
    "RainbirdStep",
    "create_pipeline",
    "process_text",
] 