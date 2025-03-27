"""
Noesis API - Clean interface for the Noesis modular pipeline.

This module provides a simple API for configuring and using the Noesis pipeline.
"""

# Standard library imports
import logging
from typing import Dict, Any, List, Optional

# Local imports
from .pipeline import Pipeline, LLMStep, PipelineStep

# Configure logging
logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    "models": {
        "noesis": {
            "type": "local",
            "path": "../models/noesis-fused-modelv0.1",
            "adapter_path": None,
            "generation": {"temperature": 0.9, "max_tokens": 10000, "verbose": False}
        },
        "preprocess": {
            "type": "local",
            "path": "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16",
            "generation": {"temperature": 0.9, "max_tokens": 10000, "verbose": False}
        },
        "validate": {
            "type": "local",
            "path": "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16",
            "generation": {"temperature": 0.9, "max_tokens": 10000, "verbose": False}
        },
        "rainbird": {
            "type": "local",
            "path": "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16",
            "generation": {"temperature": 0.9, "max_tokens": 10000, "verbose": False}
        }
    },
    "pipeline": {
        "use_preprocess": False,
        "use_validate": True,
        "use_rainbird": False
    },
    "generation": {
        "temperature": 0.9,
        "max_tokens": 10000,
        "verbose": False
    },
    "max_retries": 3,
    "graph_name_template": "request-{request_id}"
}


class Noesis:
    """
    Main API class for the Noesis pipeline.
    
    This class provides a simple interface for configuring and running
    the Noesis pipeline from Python code.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Noesis API.
        
        Args:
            config: Dictionary with configuration options.
        """
        # Initialize configuration with defaults
        self.config = DEFAULT_CONFIG.copy()
        
        # Update configuration with provided options
        if config:
            self.config.update(config)
        
        # Initialize the pipeline
        self.pipeline = None
    
    def configure(self) -> Pipeline:
        """Configure the pipeline with the current settings."""
        pipeline = Pipeline(verbose=self.config["generation"]["verbose"])
        
        # Add PREPROCESS step if enabled
        if self.config["pipeline"]["use_preprocess"]:
            preprocess_config = self.config["models"]["preprocess"]
            pipeline.add_step(
                LLMStep(
                    model_path=preprocess_config["anthropic_model"] if preprocess_config["type"] == "anthropic" else preprocess_config["path"],
                    model_type=preprocess_config["type"],
                    api_key=self.config.get("anthropic_api_key"),
                    prompt_file="preprocess.prompt",
                    name="Preprocessing input",
                    generate_kwargs=preprocess_config["generation"]
                )
            )
        
        # Add NOESIS step
        noesis_config = self.config["models"]["noesis"]
        pipeline.add_step(
            LLMStep(
                model_path=noesis_config["anthropic_model"] if noesis_config["type"] == "anthropic" else noesis_config["path"],
                model_type=noesis_config["type"],
                api_key=self.config.get("anthropic_api_key"),
                prompt_file="noesis_base.prompt",
                adapter_path=noesis_config.get("adapter_path"),
                name="Running Noesis Generation",
                generate_kwargs=noesis_config["generation"]
            )
        )
        
        # Add VALIDATE step if enabled
        if self.config["pipeline"]["use_validate"]:
            validate_config = self.config["models"]["validate"]
            pipeline.add_step(
                LLMStep(
                    model_path=validate_config["anthropic_model"] if validate_config["type"] == "anthropic" else validate_config["path"],
                    model_type=validate_config["type"],
                    api_key=self.config.get("anthropic_api_key"),
                    prompt_file="validate_stage_1.prompt",
                    name="Validate Stage 1",
                    generate_kwargs=validate_config["generation"]
                )
            )
            pipeline.add_step(
                LLMStep(
                    model_path=validate_config["anthropic_model"] if validate_config["type"] == "anthropic" else validate_config["path"],
                    model_type=validate_config["type"],
                    api_key=self.config.get("anthropic_api_key"),
                    prompt_file="validate_stage_2.prompt",
                    name="Validate Stage 2",
                    generate_kwargs=validate_config["generation"]
                )
            )
        
        # Add Rainbird step if enabled
        if self.config["pipeline"]["use_rainbird"]:
            # Import here to avoid circular imports
            from .extensions import RainbirdStep
            
            try:
                rainbird_config = self.config["models"]["rainbird"]
                pipeline.add_step(
                    RainbirdStep(
                        model_path=rainbird_config["anthropic_model"] if rainbird_config["type"] == "anthropic" else rainbird_config["path"],
                        model_type=rainbird_config["type"],
                        api_key=self.config.get("anthropic_api_key"),
                        error_prompt_file="rainbird_error.prompt",
                        max_retries=self.config["max_retries"],
                        graph_name_template=rainbird_config.get("graph_name_template", self.config["graph_name_template"]),
                        name="Process through Rainbird",
                        generate_kwargs=rainbird_config["generation"]
                    )
                )
            except Exception as e:
                logger.warning(f"Could not add Rainbird step: {e}")
        
        # Save the pipeline
        self.pipeline = pipeline
        
        return pipeline
    
    def process(self, input_text: str) -> str:
        """
        Process text through the pipeline.
        
        Args:
            input_text: The text to process.
            
        Returns:
            The processed output from the pipeline.
        """
        # Configure the pipeline if not already configured
        if self.pipeline is None:
            self.configure()
        
        # Process the input
        return self.pipeline.process(input_text)
    
    def get_pipeline_info(self) -> List[Dict[str, Any]]:
        """
        Get information about the current pipeline configuration.
        
        Returns:
            A list of dictionaries with information about each step in the pipeline.
        """
        if self.pipeline is None:
            self.configure()
        
        info = []
        for i, step in enumerate(self.pipeline.steps):
            step_info = {
                "position": i + 1,
                "name": step.name(),
                "type": step.__class__.__name__
            }
            info.append(step_info)
        
        return info
    
    def add_step(self, step: PipelineStep) -> 'Noesis':
        """
        Add a custom step to the pipeline.
        
        Args:
            step: The pipeline step to add
            
        Returns:
            self, for method chaining
        """
        if self.pipeline is None:
            self.configure()
        
        self.pipeline.add_step(step)
        return self


# Convenience functions

def process_text(input_text: str, **kwargs) -> str:
    """
    Process text through the Noesis pipeline with a simple API.
    
    Args:
        input_text: The text to process.
        **kwargs: Configuration options including:
            - noesis_model: Path to the NOESIS model
            - validate_model: Path to the VALIDATE model
            - use_validate: Whether to use the VALIDATE model (default: True)
            - use_rainbird: Whether to use the Rainbird integration (default: False)
            - temperature: Temperature for generation
            - max_tokens: Maximum tokens to generate
    
    Returns:
        The processed output from the pipeline.
    """
    # Create API instance with provided configuration
    api = Noesis(config=kwargs)
    
    # Process the input
    return api.process(input_text)


def create_pipeline(**kwargs) -> Noesis:
    """
    Create a configured Noesis pipeline instance with the specified options.
    
    Args:
        **kwargs: Configuration options including:
            - noesis_model: Path to the NOESIS model
            - validate_model: Path to the VALIDATE model
            - use_validate: Whether to use the VALIDATE model
            - use_rainbird: Whether to use the Rainbird integration
            - rainbird_model: Model to use for Rainbird
            - temperature: Temperature for generation
            - max_tokens: Maximum tokens to generate
            - max_retries: Maximum retries for pipeline steps
            - verbose: Whether to enable verbose output
    
    Returns:
        A configured Noesis instance.
    """
    # Create the API instance with passed configuration
    api = Noesis(config=kwargs)
    
    # Configure the pipeline
    api.configure()
    
    return api 