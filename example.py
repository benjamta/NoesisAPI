#!/usr/bin/env python3
"""
Command line application for processing PDF files through the Rainbird API.
"""

import os
import sys
import logging
import argparse
import pdfplumber
from collections import defaultdict

# Add the rainbird directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rainbird'))

from rainbird import Noesis

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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

def main():
    parser = argparse.ArgumentParser(description='Process PDF files through the Rainbird API')
    parser.add_argument('pdf_file', help='Path to the PDF file to process')
    parser.add_argument('--model', default="mlx-community/Meta-Llama-3-8B-Instruct-4bit",
                      help='Model to use for processing (default: mlx-community/Meta-Llama-3-8B-Instruct-4bit)')
    parser.add_argument('--temperature', type=float, default=0.9,
                      help='Temperature for text generation (default: 0.9)')
    parser.add_argument('--max-tokens', type=int, default=10000,
                      help='Maximum number of tokens to generate (default: 10000)')
    args = parser.parse_args()

    # Define configuration
    config = {
        "noesis_model": args.model,
        "adapter_path": None,
        "validate_model": args.model,
        "preprocess_model": args.model,
        "use_validate": True,
        "use_preprocess": True,
        "use_rainbird": False,
        "graph_name_template": "Noesis API Test 1",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "verbose": True
    }

    # Extract text from PDF
    print(f"Processing PDF file: {args.pdf_file}")
    text = extract_text_from_pdf(args.pdf_file)
    
    # Process the text
    print("\nProcessing text through pipeline...")
    result = process_text(text, config)
    
    # Print results
    print("\nResults:")
    print("-" * 50)
    print(result)

if __name__ == "__main__":
    main()