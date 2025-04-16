#!/usr/bin/env python3
"""
Command line application for processing PDF files through the Rainbird API.
"""

import os
import sys
import logging
import argparse
import yaml
import pdfplumber
import requests
from collections import defaultdict
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Add the rainbird directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rainbird'))

from rainbird import Noesis

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        sys.exit(1)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file while preserving formatting."""
    try:
        text_content = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract text with layout preservation
                text = page.extract_text(x_tolerance=3, y_tolerance=3, layout=True)
                
                # Extract tables if present
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        # Convert table to formatted text
                        table_text = []
                        for row in table:
                            # Filter out None values and join with tabs
                            row_text = '\t'.join(str(cell) if cell is not None else '' for cell in row)
                            table_text.append(row_text)
                        text += '\n' + '\n'.join(table_text) + '\n'
                
                text_content.append(text)
        
        # Join all pages with clear separation
        full_text = '\n\n'.join(text_content)
        
        # Clean up the text
        full_text = full_text.replace('\n\n\n', '\n\n')  # Remove excessive newlines
        full_text = full_text.replace('\t\t', '\t')      # Remove excessive tabs
        
        return full_text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        sys.exit(1)

def get_existing_graph(kmID, api_key):
    """Fetch an existing graph from the Rainbird API."""
    try:
        # Log the API key (masked for security)
        masked_key = api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:] if api_key else None
        logging.debug(f"Using Rainbird API key: {masked_key}")
        
        # Use HTTP Basic Auth with API key as username and empty password
        auth = (api_key, '')
        headers = {
            'Version': 'v1',
            'Content-Type': 'application/json'
        }
        
        url = f"https://api.rainbird.ai/analysis/file/{kmID}"
        logging.debug(f"Making request to: {url}")
        logging.debug(f"Using Basic Auth with username: {masked_key}")
        
        response = requests.get(url, auth=auth, headers=headers)
        
        # Log response details if there's an error
        if response.status_code != 200:
            logging.error(f"API request failed with status code: {response.status_code}")
            logging.error(f"Response headers: {response.headers}")
            logging.error(f"Response body: {response.text}")
            response.raise_for_status()
        
        return response.text
    except Exception as e:
        logging.error(f"Error fetching existing graph: {e}")
        sys.exit(1)

def process_text(text, config, existing_graph=None):
    """Process text through the Noesis pipeline."""
    pipeline = Noesis(config=config)
    
    # If we have an existing graph, use the update prompt
    if existing_graph:
        # Replace the prompt file for the noesis step
        pipeline.config["models"]["noesis"]["prompt_file"] = "rainbird/noesis/prompts/noesis_update.prompt"
                    
        return pipeline.process(f"<graph to update>\n{existing_graph}\n</graph to update>\n<expertise to add>\n{text}\n</expertise to add>")
    else:
        return pipeline.process(text)

def get_model_config(model_name, args, config):
    """Get model configuration with command line overrides."""
    model_config = config['models'][model_name].copy()
    
    # Apply command line overrides if provided
    if hasattr(args, f'{model_name}_model') and getattr(args, f'{model_name}_model') is not None:
        model_config['path'] = getattr(args, f'{model_name}_model')
    if hasattr(args, f'{model_name}_type') and getattr(args, f'{model_name}_type') is not None:
        model_config['type'] = getattr(args, f'{model_name}_type')
    if hasattr(args, f'{model_name}_anthropic_model') and getattr(args, f'{model_name}_anthropic_model') is not None:
        model_config['anthropic_model'] = getattr(args, f'{model_name}_anthropic_model')
    if hasattr(args, f'{model_name}_adapter_path') and getattr(args, f'{model_name}_adapter_path') is not None:
        model_config['adapter_path'] = getattr(args, f'{model_name}_adapter_path')
    
    # Get generation parameters from model config or global config
    generation_config = model_config.get('generation', {})
    global_generation = config.get('generation', {})
    
    # Use model-specific generation params if available, otherwise fall back to global
    model_config['generation'] = {
        'temperature': generation_config.get('temperature', global_generation.get('temperature', 0.9)),
        'max_tokens': generation_config.get('max_tokens', global_generation.get('max_tokens', 4000)),
        'verbose': global_generation.get('verbose', False)
    }
    
    return model_config

def main():
    parser = argparse.ArgumentParser(description='Process PDF files through the Rainbird API')
    parser.add_argument('pdf_file', help='Path to the PDF file to process')
    parser.add_argument('--config', default='config.yaml',
                      help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--kmID', help='Knowledge Map ID of graph to update (if updating an existing graph)')
    
    # Model configuration arguments for each step
    for step in ['noesis', 'preprocess', 'validate', 'rainbird']:
        parser.add_argument(f'--{step}-model',
                          help=f'Override {step} model path/name from config')
        parser.add_argument(f'--{step}-type', choices=['local', 'anthropic'],
                          help=f'Override {step} model type from config')
        parser.add_argument(f'--{step}-anthropic-model',
                          help=f'Override {step} Anthropic model name from config')
        parser.add_argument(f'--{step}-adapter-path',
                          help=f'Override {step} adapter path from config')
        parser.add_argument(f'--{step}-temperature', type=float,
                          help=f'Override temperature for {step} step')
        parser.add_argument(f'--{step}-max-tokens', type=int,
                          help=f'Override max tokens for {step} step')
    
    # API key arguments
    parser.add_argument('--anthropic-api-key',
                      help='Anthropic API key (if not set, will try to get from ANTHROPIC_API_KEY environment variable)')
    parser.add_argument('--rainbird-anthropic-api-key',
                      help='Anthropic API key for Rainbird error correction (if different from main API key)')
    parser.add_argument('--rainbird-api-key',
                      help='Rainbird API key (if not set, will try to get from RAINBIRD_API_KEY environment variable)')
    parser.add_argument('--graph-name',
                      help='Override the graph name template with a specific name')
    
    args = parser.parse_args()

    # Load base configuration
    config = load_config(args.config)

    # Get model configurations with overrides
    noesis_config = get_model_config('noesis', args, config)
    preprocess_config = get_model_config('preprocess', args, config)
    validate_config = get_model_config('validate', args, config)
    rainbird_config = get_model_config('rainbird', args, config)

    # Get API keys from environment or command line
    main_api_key = args.anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
    rainbird_api_key = args.rainbird_anthropic_api_key or main_api_key
    rainbird_api_key = args.rainbird_api_key or os.getenv('RAINBIRD_API_KEY')

    # Validate API keys for each step that uses Anthropic
    for step, step_config in [
        ('noesis', noesis_config),
        ('preprocess', preprocess_config),
        ('validate', validate_config),
        ('rainbird', rainbird_config)
    ]:
        if step_config['type'] == 'anthropic' and not main_api_key:
            print(f"Error: Anthropic API key is required for {step} step. Please provide it using --anthropic-api-key or set ANTHROPIC_API_KEY environment variable.")
            sys.exit(1)

    # Convert config to pipeline format
    pipeline_config = {
        "models": {
            "noesis": noesis_config,
            "preprocess": preprocess_config,
            "validate": validate_config,
            "rainbird": rainbird_config
        },
        "pipeline": config['pipeline'],
        "generation": config['generation'],
        "max_retries": config.get('max_retries', 3),
        "graph_name_template": args.graph_name if args.graph_name else config.get('graph_name_template', "request-{request_id}"),
        "anthropic_api_key": main_api_key
    }

    # Extract text from PDF
    print(f"\nProcessing PDF file: {args.pdf_file}")
    text = extract_text_from_pdf(args.pdf_file)

    # Get existing graph if updating
    existing_graph = None
    if args.kmID:
        if not rainbird_api_key:
            print("Error: Rainbird API key is required for updating graphs. Please provide it using --rainbird-api-key or set RAINBIRD_API_KEY environment variable.")
            sys.exit(1)
        print(f"\nFetching existing graph with KMID: {args.kmID}")
        existing_graph = get_existing_graph(args.kmID, rainbird_api_key)

    # Process the text
    print("\nProcessing text through pipeline...")
    result = process_text(text, pipeline_config, existing_graph)
    
    # Print results
    if args.kmID:
        print(f"\nUpdated Graph with KMID: {result['api_response']['kmID']}")
    else:
        print(f"\nCreated Graph with KMID: {result['api_response']['kmID']}")
    
if __name__ == "__main__":
    main()
