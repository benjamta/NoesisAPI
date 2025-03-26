"""
Extension modules for the Noesis pipeline.

This module provides additional pipeline steps beyond the base functionality.
"""

# Standard library imports
import os
import json
import uuid
import logging
import requests
from typing import Dict, Any, Optional

# Third-party imports
from dotenv import load_dotenv
from mlx_lm import generate, load

# Local imports
from .pipeline import PipelineStep

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Rainbird API constants
RAINBIRD_API_URL = "https://api.rainbird.ai/maps"


class RainbirdStep(PipelineStep):
    """A pipeline step that sends data to the Rainbird API and handles errors."""
    
    def __init__(self, 
                model_path: str, 
                error_prompt_file: str, 
                max_retries: int = 3, 
                graph_name_template: str = "request-{request_id}",
                model_type: str = "local",  # "local" or "anthropic"
                anthropic_model: str = "claude-3-opus-20240229",
                api_key: str = None):
        """
        Initialize the Rainbird processing step.
        
        Args:
            model_path: Path to the LLM model for error correction (for local models)
            error_prompt_file: Path to the error correction prompt file
            max_retries: Maximum number of retries for API errors
            graph_name_template: Template string for naming the Rainbird graph. 
                               Use {request_id} as a placeholder for the request ID.
            model_type: Type of model to use for error correction ("local" or "anthropic")
            anthropic_model: Name of the Anthropic model to use (if model_type is "anthropic")
            api_key: API key for Anthropic (if None, will try to get from environment)
        """
        self.model_path = model_path
        self.max_retries = max_retries
        self.graph_name_template = graph_name_template
        self.model_type = model_type
        self.anthropic_model = anthropic_model
        self.model = None
        self.tokenizer = None
        
        # Load prompt
        try:
            # Try both absolute path and relative to prompts directory
            if os.path.isfile(error_prompt_file):
                prompt_path = error_prompt_file
            else:
                # Get path to prompts directory
                from pathlib import Path
                prompts_dir = Path(__file__).parent / "prompts"
                prompt_path = os.path.join(prompts_dir, error_prompt_file)
                
            with open(prompt_path, 'r') as f:
                self.error_prompt = f.read()
        except FileNotFoundError:
            raise ValueError(f"Error prompt file not found: {error_prompt_file}")
        
        # Setup API headers for Rainbird
        rainbird_api_key = os.getenv('RAINBIRD_API_KEY')
        if not rainbird_api_key:
            logger.warning("RAINBIRD_API_KEY not set in environment variables")
            self.api_headers = {}
        else:
            self.api_headers = {
                'X-API-Key': rainbird_api_key,
                'Version': 'v1',
                'Content-Type': 'application/json'
            }
            
        # Setup Anthropic API if needed
        if model_type == "anthropic":
            self.anthropic_api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in environment variables or provided")
            self.anthropic_headers = {
                'x-api-key': self.anthropic_api_key,
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json'
            }
    
    def _load_model(self):
        """Load the model and tokenizer if not already loaded."""
        if self.model is None:
            logger.info(f"Loading model for error correction: {self.model_path}")
            self.model, self.tokenizer = load(self.model_path)
            logger.info(f"Model loaded: {self.model_path}")
    
    def _unload_model(self):
        """Unload the model and tokenizer to free memory."""
        if self.model is not None:
            logger.info(f"Unloading model: {self.model_path}")
            self.model = None
            self.tokenizer = None
            logger.info(f"Model unloaded: {self.model_path}")
    
    def clean_xml(self, text: str) -> str:
        """
        Clean XML output from LLM to ensure it only contains the XML content.
        
        Args:
            text: The raw text containing XML
            
        Returns:
            Cleaned XML content
        """
        # Find the start and end of the XML
        xml_start = text.find('<rbl:kb xmlns:rbl="http://rbl.io/schema/RBLang">')
        xml_end = text.find('</rbl:kb>') + len('</rbl:kb>')
        
        if xml_start == -1 or xml_end == -1:
            logger.warning("Could not find proper XML markers in output")
            return text
        
        # Extract just the XML content
        xml = text[xml_start:xml_end]
        
        return xml.strip()
    
    def send_to_rainbird(self, xml: str, request_id: str) -> Dict[str, Any]:
        """
        Send generated XML to Rainbird API.
        
        Args:
            xml: The XML string to send
            request_id: A unique identifier for this request
            
        Returns:
            API response as a dictionary
        """
        api_key = os.getenv('RAINBIRD_API_KEY')
        if not api_key:
            return {"error": "RAINBIRD_API_KEY not set in environment variables"}
        
        # Format the graph name using the template
        graph_name = self.graph_name_template.format(request_id=request_id)
        
        payload = {
            "rblang": xml,
            "name": graph_name,
            "description": f"Generated map for request {request_id}"
        }
        
        try:
            response = requests.post(
                RAINBIRD_API_URL,
                headers=self.api_headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending to Rainbird API for request {request_id}:")
            logger.error(f"Error: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
            return {"error": str(e)}
    
    def _generate_error_correction(self, xml: str, error_message: str) -> str:
        """
        Generate error correction using either local model or Anthropic API.
        
        Args:
            xml: The original XML that caused the error
            error_message: The error message from Rainbird API
            
        Returns:
            Corrected XML string
        """
        if self.model_type == "local":
            return self._generate_local_correction(xml, error_message)
        else:  # anthropic
            return self._generate_anthropic_correction(xml, error_message)
    
    def _generate_local_correction(self, xml: str, error_message: str) -> str:
        """Generate error correction using local model."""
        self._load_model()
        
        messages = [
            {"role": "system", "content": self.error_prompt},
            {"role": "user", "content": f"Here is my XML:\n\n{xml}"}
        ]
        
        error_message = f"""
The previous XML resulted in this error:
{error_message}

Please provide corrected XML that addresses this error. 
Just provide the corrected XML, no other text.
"""
        messages.append({"role": "user", "content": error_message})
        
        formatted_fix_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        fixed_xml = generate(
            self.model, 
            self.tokenizer,
            prompt=formatted_fix_prompt,
            verbose=False,
            max_tokens=2000
        )
        
        return self.clean_xml(fixed_xml.strip())
    
    def _generate_anthropic_correction(self, xml: str, error_message: str) -> str:
        """Generate error correction using Anthropic API."""
        messages = [
            {"role": "system", "content": self.error_prompt},
            {"role": "user", "content": f"Here is my XML:\n\n{xml}"}
        ]
        
        error_message = f"""
The previous XML resulted in this error:
{error_message}

Please provide corrected XML that addresses this error. 
Just provide the corrected XML, no other text.
"""
        messages.append({"role": "user", "content": error_message})
        
        payload = {
            "model": self.anthropic_model,
            "messages": messages,
            "temperature": 0.9,
            "max_tokens": 2000
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=self.anthropic_headers,
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        return self.clean_xml(result['content'][0]['text'].strip())
    
    def process(self, input_xml: str) -> Dict[str, Any]:
        """
        Process the input XML by sending it to the Rainbird API,
        handling errors, and attempting to fix them.
        
        Args:
            input_xml: The XML string to send to Rainbird
        
        Returns:
            Dictionary with the API response and any error correction attempts
        """
        try:
            # Generate a unique request ID
            request_id = str(uuid.uuid4())[:8]
            
            # Clean XML if needed
            xml = self.clean_xml(input_xml)
            
            # Send to Rainbird API
            api_response = self.send_to_rainbird(xml, request_id)
            
            # Track all attempts
            attempts = [{"xml": xml, "error": api_response.get('error')}]
            
            # If there's an error and retries are enabled, try to fix it
            retry_count = 0
            while 'error' in api_response and retry_count < self.max_retries:
                logger.info(f"Request {request_id} error: {api_response['error']}")
                
                # Generate fixed XML using either local model or Anthropic
                fixed_xml = self._generate_error_correction(xml, api_response['error'])
                
                # Try API again with fixed XML
                api_response = self.send_to_rainbird(
                    fixed_xml,
                    f"{request_id}_fix_{retry_count + 1}"
                )
                
                # Check if this error is the same as any previous attempt
                current_error = api_response.get('error')
                if current_error and any(attempt['error'] == current_error for attempt in attempts):
                    logger.warning(f"Request {request_id} got repeated error, stopping retries: {current_error}")
                    break
                
                # Track this attempt
                attempts.append({"xml": fixed_xml, "error": current_error})
                
                if 'error' in api_response:
                    logger.warning(f"Request {request_id} error fixing (attempt {retry_count + 1}): {api_response['error']}")
                
                xml = fixed_xml
                retry_count += 1
            
            result = {
                "generated_xml": xml,
                "api_response": api_response,
                "attempts": attempts
            }
            
            return result
        finally:
            # Always unload model after processing if using local model
            if self.model_type == "local":
                self._unload_model()
    
    def name(self) -> str:
        """Return the name of this pipeline step."""
        return "RainbirdAPI"


class AnthropicStep(PipelineStep):
    """A pipeline step that processes text through Anthropic's API."""
    
    def __init__(self, 
                model_name: str, 
                prompt_file: str,
                api_key: str = None,
                generate_kwargs: Optional[Dict] = None):
        """
        Initialize the Anthropic API processing step.
        
        Args:
            model_name: Name of the Anthropic model to use (e.g., 'claude-3-opus-20240229')
            prompt_file: Path to the system prompt file
            api_key: Anthropic API key (if None, will try to get from environment)
            generate_kwargs: Optional kwargs for the API call
        """
        self.model_name = model_name
        self.prompt_file = prompt_file
        self.generate_kwargs = generate_kwargs or {
            "temperature": 0.9,
            "max_tokens": 10000
        }
        
        # Load prompt
        try:
            # Try both absolute path and relative to prompts directory
            if os.path.isfile(prompt_file):
                prompt_path = prompt_file
            else:
                # Get path to prompts directory
                from pathlib import Path
                prompts_dir = Path(__file__).parent / "prompts"
                prompt_path = os.path.join(prompts_dir, prompt_file)
                
            with open(prompt_path, 'r') as f:
                self.system_prompt = f.read()
        except FileNotFoundError:
            raise ValueError(f"Prompt file not found: {prompt_file}")
        
        # Setup API key
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment variables or provided")
        
        # Setup API headers
        self.api_headers = {
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        }
    
    def process(self, input_text: str) -> str:
        """
        Process the input text through Anthropic's API and return the result.
        
        Args:
            input_text: The text to process
            
        Returns:
            Generated text from the API
        """
        try:
            # Initialize message history
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": input_text}
            ]
            
            # Prepare API request
            payload = {
                "model": self.model_name,
                "messages": messages,
                **self.generate_kwargs
            }
            
            # Make API request
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=self.api_headers,
                json=payload
            )
            response.raise_for_status()
            
            # Extract response
            result = response.json()
            return result['content'][0]['text'].strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Anthropic API: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
            raise
    
    def name(self) -> str:
        """Return the name of this pipeline step."""
        return f"Anthropic({self.model_name})" 