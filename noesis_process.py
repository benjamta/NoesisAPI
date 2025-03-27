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

def process_text(text, config):
    """Process text through the Noesis pipeline."""
        
    pipeline = Noesis(config=config)
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
    
    
    return model_config

def main():
    parser = argparse.ArgumentParser(description='Process PDF files through the Rainbird API')
    parser.add_argument('pdf_file', help='Path to the PDF file to process')
    parser.add_argument('--config', default='config.yaml',
                      help='Path to configuration file (default: config.yaml)')
    
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
    
    # API key arguments
    parser.add_argument('--anthropic-api-key',
                      help='Anthropic API key (if not set, will try to get from ANTHROPIC_API_KEY environment variable)')
    parser.add_argument('--rainbird-anthropic-api-key',
                      help='Anthropic API key for Rainbird error correction (if different from main API key)')
    
    # Generation settings
    parser.add_argument('--temperature', type=float,
                      help='Override temperature from config')
    parser.add_argument('--max-tokens', type=int,
                      help='Override max tokens from config')
    
    args = parser.parse_args()

    # Load base configuration
    config = load_config(args.config)

    # Get model configurations with overrides
    noesis_config = get_model_config('noesis', args, config)
    preprocess_config = get_model_config('preprocess', args, config)
    validate_config = get_model_config('validate', args, config)
    rainbird_config = get_model_config('rainbird', args, config)

    # Override generation settings
    if args.temperature is not None:
        config['generation']['temperature'] = args.temperature
    if args.max_tokens is not None:
        config['generation']['max_tokens'] = args.max_tokens

    # Get API keys from environment or command line
    main_api_key = args.anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
    rainbird_api_key = args.rainbird_anthropic_api_key or main_api_key

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
        # Noesis step
        "noesis_model": noesis_config['path'] if noesis_config['type'] == 'local' else noesis_config['anthropic_model'],
        "noesis_model_type": noesis_config['type'],
        "noesis_api_key": main_api_key,
        "adapter_path": noesis_config['adapter_path'],
        
        # Preprocess step
        "preprocess_model": preprocess_config['path'] if preprocess_config['type'] == 'local' else preprocess_config['anthropic_model'],
        "preprocess_model_type": preprocess_config['type'],
        "preprocess_api_key": main_api_key,
        
        # Validate step
        "validate_model": validate_config['path'] if validate_config['type'] == 'local' else validate_config['anthropic_model'],
        "validate_model_type": validate_config['type'],
        "validate_api_key": main_api_key,
        
        # Rainbird step
        "use_rainbird": config['pipeline']['use_rainbird'],
        "rainbird_model_type": rainbird_config['type'],
        "rainbird_anthropic_model": rainbird_config['anthropic_model'],
        "rainbird_anthropic_api_key": rainbird_api_key,
        "graph_name_template": rainbird_config['graph_name_template'],
        
        # Pipeline settings
        "use_validate": config['pipeline']['use_validate'],
        "use_preprocess": config['pipeline']['use_preprocess'],
        
        # Generation settings
        "temperature": config['generation']['temperature'],
        "max_tokens": config['generation']['max_tokens'],
        "verbose": config['generation']['verbose']
    }

    # Extract text from PDF
    print(f"\nProcessing PDF file: {args.pdf_file}")
    # text = extract_text_from_pdf(args.pdf_file)

    text = """
<?xml version="1.0" encoding="utf-8"?>
<rbl:kb xmlns:rbl="http://rbl.io/schema/RBLang">
  <concept name="person" type="string"/>
  <concept name="language" type="string"/>
  <concept name="country" type="string"/>

  <rel name="speaks" subject="person" object="language" plural="true" allowUnknown="true"/>
  <rel name="born in" subject="person" object="country" allowUnknown="true" />
  <rel name="national language" subject="country" object="language"/>  
  <rel name="lives in" subject="person" object="place" allowUnknown="true" />

  <relinst type="speaks" cf="80">
    <condition rel="born in" subject="%S" object="%COUNTRY" behaviour="optional" weight="40"/>
    <condition rel="lives in" subject="%S" object="%COUNTRY" behaviour="optional" weight="80"/>
    <condition rel="national language" subject="%COUNTRY" object="%O"/>
  </relinst>
</rbl:kb>
"""

    # Process the text
    print("\nProcessing text through pipeline...")
    result = process_text(text, pipeline_config)
    
    # Print results
    print(f"\nCreated Graph with KMID: {result['api_response']['kmID']}")
    
if __name__ == "__main__":
    main()