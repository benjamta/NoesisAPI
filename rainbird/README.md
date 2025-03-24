# Rainbird

A modular pipeline framework for processing text with language models.

## Overview

Rainbird provides a clean, flexible interface for building and running modular text processing pipelines with LLMs. It allows you to:

- Create pipelines with multiple processing steps
- Configure and customize LLM-based processing
- Easily extend with custom processing steps
- Process text through different models with a unified API

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rainbird.git
cd rainbird

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

```python
from rainbird import process_text

# Process text with default settings
result = process_text("People speak the language of the country they're born in")
print(result)
```

## Core Concepts

### Pipeline

A processing pipeline consists of one or more steps that transform input data. Each step takes the output of the previous step as its input.

### Pipeline Steps

- `LLMStep`: Processes text through a language model
- `RainbirdStep`: Sends data to the Rainbird API for knowledge graph generation
- Custom steps: Create your own by extending the `PipelineStep` base class

## Usage Examples

### Basic Configuration

```python
from rainbird import Noesis

# Create a pipeline with custom configuration
api = Noesis(config={
    "noesis_model": "path/to/model",
    "validate_model": "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16",
    "use_validate": True,
    "temperature": 0.7,
    "max_tokens": 5000
})

# Configure and process text
api.configure()
result = api.process("People speak the language of the country they're born in")
print(result)
```

### Creating Custom Pipelines

```python
from rainbird import create_pipeline
from rainbird import LLMStep

# Create a pipeline with just the base step
api = create_pipeline(use_validate=False)

# Add a custom step
api.add_step(
    LLMStep(
        model_path="mlx-community/Meta-Llama-3.1-8B-Instruct-bf16",
        prompt_file="custom_prompt.txt",
        generate_kwargs={"temp": 0.5}
    )
)

# Process text through the pipeline
result = api.process("Your input text here")
```

### Using Rainbird Integration

```python
from rainbird import create_pipeline

# Create a pipeline with Rainbird enabled
api = create_pipeline(
    use_validate=True,
    use_rainbird=True,
    rainbird_model="mlx-community/Meta-Llama-3.1-8B-Instruct-bf16"
)

# Process text
result = api.process("Your input text here")
```

### Creating Custom Pipeline Steps

```python
from rainbird import PipelineStep

class CustomStep(PipelineStep):
    def __init__(self, parameter):
        self.parameter = parameter
    
    def process(self, input_data):
        # Process the input data
        return processed_data
    
    def name(self):
        return "CustomStep"

# Use your custom step
from rainbird import create_pipeline
api = create_pipeline()
api.add_step(CustomStep(parameter="value"))
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `noesis_model` | Path to the NOESIS model | "../models/noesis-fused-model" |
| `validate_model` | Path to the VALIDATE model | "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16" |
| `rainbird_model` | Model to use for Rainbird | "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16" |
| `use_validate` | Whether to use the VALIDATE model | True |
| `use_rainbird` | Whether to use the Rainbird integration | False |
| `temperature` | Temperature for generation | 0.9 |
| `max_tokens` | Maximum tokens to generate | 10000 |
| `max_retries` | Maximum retries for pipeline steps | 3 |
| `verbose` | Whether to enable verbose output | False |
| `graph_name_template` | Template for Rainbird graph names | "request-{request_id}" |

## Advanced Usage

See the `example.py` script for more advanced usage examples.

## License

[Add your license information here] 