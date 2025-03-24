"""
Core pipeline components for the Noesis framework.

This module provides the base classes for building modular processing pipelines.
"""

# Standard library imports
import os
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from pathlib import Path

# Third-party imports
from mlx_lm import generate, load

# Constants
PROMPTS_DIR = Path(__file__).parent / "prompts"


class PipelineStep(ABC):
    """Abstract base class for a step in the processing pipeline."""
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Process the input data and return the result.
        
        Args:
            input_data: The input data to process
            
        Returns:
            Processed output data
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the name of this pipeline step."""
        pass


class LLMStep(PipelineStep):
    """A pipeline step that processes text through an LLM."""
    
    def __init__(self, 
                model_path: str, 
                prompt_file: str, 
                adapter_path: str = None,
                generate_kwargs: Optional[Dict] = None):
        """
        Initialize the LLM processing step.
        
        Args:
            model_path: Path to the LLM model
            prompt_file: Path to the system prompt file
            generate_kwargs: Optional kwargs for the generate function
        """
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.prompt_file = prompt_file
        self.generate_kwargs = generate_kwargs or {
            "verbose": False,
            "temp": 0.9,
            "max_tokens": 10000
        }
        
        # Load prompt
        try:
            # Try both absolute path and relative to prompts directory
            if os.path.isfile(prompt_file):
                prompt_path = prompt_file
            else:
                prompt_path = os.path.join(PROMPTS_DIR, prompt_file)
                
            with open(prompt_path, 'r') as f:
                self.system_prompt = f.read()
        except FileNotFoundError:
            raise ValueError(f"Prompt file not found: {prompt_file}")
        
        # Load model
        self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
    
    def process(self, input_text: str) -> str:
        """
        Process the input text through the LLM and return the result.
        
        Args:
            input_text: The text to process
            
        Returns:
            Generated text from the LLM
        """
        # Initialize message history
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text}
        ]
        
        # Format prompt
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Generate response
        result = generate(
            self.model, 
            self.tokenizer, 
            prompt=formatted_prompt, 
            **self.generate_kwargs
        )
        
        return result.strip()
    
    def name(self) -> str:
        """Return the name of this pipeline step."""
        return f"LLM({os.path.basename(self.model_path)})"


class Pipeline:
    """A pipeline that processes data through a sequence of steps."""
    
    def __init__(self):
        """Initialize an empty pipeline."""
        self.steps = []
    
    def add_step(self, step: PipelineStep) -> 'Pipeline':
        """
        Add a step to the pipeline.
        
        Args:
            step: The pipeline step to add
        
        Returns:
            self, for method chaining
        """
        self.steps.append(step)
        return self
    
    def process(self, input_data: Any) -> Any:
        """
        Process data through all steps in the pipeline.
        
        Args:
            input_data: The input data for the first step
        
        Returns:
            The output from the last step
        """
        result = input_data
        for i, step in enumerate(self.steps):
            result = step.process(result)
        return result 