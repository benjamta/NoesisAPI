"""
Core pipeline components for the Noesis framework.

This module provides the base classes for building modular processing pipelines.
"""

# Standard library imports
import os
import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from pathlib import Path
import requests

# Third-party imports
from mlx_lm import generate, load

# Configure logging
logger = logging.getLogger(__name__)

# Constants
PROMPTS_DIR = Path(__file__).parent / "prompts"


class PipelineStep(ABC):
    """Abstract base class for a step in the processing pipeline."""
    
    def __init__(self, name: str = None):
        """
        Initialize a pipeline step.
        
        Args:
            name: Optional custom name for this step
        """
        self._name = name
    
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

    def name(self) -> str:
        """
        Return the name of this pipeline step.
        
        Returns:
            The custom name if set, otherwise the default name from _get_default_name()
        """
        if self._name:
            return self._name
        return self._get_default_name()
    
    def _get_default_name(self) -> str:
        """
        Get the default name for this step.
        Should be implemented by subclasses.
        
        Returns:
            The default name for this step
        """
        return self.__class__.__name__


class LLMStep(PipelineStep):
    """A pipeline step that processes text through an LLM."""
    
    def __init__(self, 
                model_path: str, 
                prompt_file: str, 
                adapter_path: str = None,
                model_type: str = "local",  # "local" or "anthropic"
                api_key: str = None,
                generate_kwargs: Optional[Dict] = None,
                name: str = None):
        """
        Initialize the LLM processing step.
        
        Args:
            model_path: Path to the LLM model or model name for remote models
            prompt_file: Path to the system prompt file
            adapter_path: Path to the adapter (for local models only)
            model_type: Type of model to use ("local" or "anthropic")
            api_key: API key for remote models (if None, will try to get from environment)
            generate_kwargs: Optional kwargs for the generate function
            name: Optional custom name for this step
        """
        super().__init__(name)
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.prompt_file = prompt_file
        self.model_type = model_type
        self.api_key = api_key
        self.generate_kwargs = generate_kwargs or {
            "verbose": False,
            "temp": 0.9,
            "max_tokens": 10000
        }
        self.model = None
        self.tokenizer = None
        
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
        
        # Initialize based on model type
        if model_type == "local":
            self._load_model()
        elif model_type == "anthropic":
            if not self.api_key:
                self.api_key = os.getenv('ANTHROPIC_API_KEY')
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in environment variables or provided")
            self.api_headers = {
                'x-api-key': self.api_key,
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json'
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _load_model(self):
        """Load the model and tokenizer if not already loaded."""
        if self.model is None:
            logger.info(f"Loading model: {self.model_path}")
            self.model, self.tokenizer = load(self.model_path, adapter_path=self.adapter_path)
            logger.info(f"Model loaded: {self.model_path}")
    
    def _unload_model(self):
        """Unload the model and tokenizer to free memory."""
        if self.model is not None:
            logger.info(f"Unloading model: {self.model_path}")
            self.model = None
            self.tokenizer = None
            logger.info(f"Model unloaded: {self.model_path}")
    
    def process(self, input_text: str) -> str:
        """
        Process the input text through the LLM and return the result.
        
        Args:
            input_text: The text to process
            
        Returns:
            Generated text from the LLM
        """
        try:
            if self.model_type == "local":
                return self._process_local(input_text)
            else:  # anthropic
                return self._process_anthropic(input_text)
        finally:
            if self.model_type == "local":
                self._unload_model()
    
    def _process_local(self, input_text: str) -> str:
        """Process text using local MLX model."""
        # Load model if needed
        self._load_model()
        
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
    
    def _process_anthropic(self, input_text: str) -> str:
        """Process text using Anthropic API."""
        # Prepare API request
        payload = {
            "model": self.model_path,
            "system": self.system_prompt,  # System prompt as top-level parameter
            "messages": [
                {"role": "user", "content": input_text}  # Only user message in messages array
            ],
            "temperature": self.generate_kwargs.get("temp", 0.9),  # Map temp to temperature
            "max_tokens": self.generate_kwargs.get("max_tokens", 10000)
        }
        
        # Make API request
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=self.api_headers,
            json=payload
        )
        
        if response.status_code != 200:
            print(f"Debug - API Response Status: {response.status_code}")
            print(f"Debug - API Response Headers: {response.headers}")
            print(f"Debug - API Response Body: {response.text}")
            response.raise_for_status()
        
        # Extract response
        result = response.json()
        return result['content'][0]['text'].strip()
    
    def _get_default_name(self) -> str:
        """Return the default name for this step."""
        if self.model_type == "local":
            return f"LLM({os.path.basename(self.model_path)})"
        else:
            return f"Anthropic({self.model_path})"


class Pipeline:
    """A pipeline that processes data through a sequence of steps."""
    
    def __init__(self, verbose: bool = False):
        """
        Initialize an empty pipeline.
        
        Args:
            verbose: Whether to print verbose logging information
        """
        self.steps = []
        self.verbose = verbose
    
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
            if self.verbose:
                print(f"Running step {i+1}: {step.name()}")
            result = step.process(result)
        return result 