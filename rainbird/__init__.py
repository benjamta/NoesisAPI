"""
Rainbird - A modular pipeline for processing text with LLMs.

This package provides a simple interface for configuring and running
modular processing pipelines with LLMs.
"""

# Re-export everything from the noesis module
from noesis import Noesis, Pipeline, PipelineStep, LLMStep, RainbirdStep
from noesis import create_pipeline, process_text

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