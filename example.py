#!/usr/bin/env python3
"""
Example script demonstrating how to use the Rainbird API.

This script assumes the rainbird package directory is in the Python path.
You can run it after installing the package with:
    pip install -e ./rainbird
"""

import os
import sys
import logging

# Add the rainbird directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rainbird'))

from rainbird import Noesis

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def ben_test():
    """Demonstrate using custom configuration options."""
    print("\n=== Running Example ===")
    
    # Define custom configuration
    config = {
        "noesis_model": "./models/noesis-base-0.3",
        "use_validate": False,
        "use_rainbird": False,
        "graph_name_template": "Noesis API Test 1",
        "temperature": 0.9,
        "max_tokens": 10000,
        "verbose": True
    }
    
    pipeline = Noesis(config=config)
    
    # Process text
    input_text = "People speak the language of the country they're born in"
        
    result = pipeline.process(input_text)
    
    print(f"Result: {result}")


if __name__ == "__main__":
    ben_test()