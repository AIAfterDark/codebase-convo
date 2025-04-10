"""
Ollama LLM Interface Module

This module handles communication with the local Ollama instance
running llama3.2 or other compatible models.
"""

import json
import requests
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
import hashlib
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for interacting with the Ollama API to generate
    embeddings and text responses.
    """

    def __init__(self, 
                 model_name: str = "llama3.2:latest", 
                 embedding_model_name: str = "nomic-embed-text",
                 api_base: str = "http://localhost:11434", 
                 mock_mode: bool = False):
        """
        Initialize the Ollama client.

        Args:
            model_name: Name of the Ollama model for generation/chat
            embedding_model_name: Name of the Ollama model for embeddings
            api_base: Base URL for the Ollama API
            mock_mode: If True, use mock implementations instead of calling Ollama
        """
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name # Store the embedding model name
        self.api_base = api_base.rstrip('/')
        self.embedding_endpoint = f"{self.api_base}/api/embeddings"
        self.generate_endpoint = f"{self.api_base}/api/generate"
        self.chat_endpoint = f"{self.api_base}/api/chat"
        self.mock_mode = mock_mode
        
        # No mock mode check here, assume always connecting
        self._check_connection()
        self._log_available_models() # Add call to log models

    def _check_connection(self) -> None:
        """
        Check if Ollama is running and the model is available.
        Raises an exception if not.
        """
        try:
            # Try to get model info
            response = requests.get(f"{self.api_base}/api/tags")
            response.raise_for_status()
            
            # Check if our model is available
            models = response.json().get("models", [])
            model_names = [model.get("name") for model in models]
            
            if self.model_name not in model_names:
                logger.warning(f"Model {self.model_name} not found in Ollama. Available models: {model_names}")
                logger.warning(f"You may need to pull the model with: ollama pull {self.model_name}")
            else:
                logger.info(f"Successfully connected to Ollama with model {self.model_name}")
                
        except requests.RequestException as e:
            logger.error(f"Failed to connect to Ollama at {self.api_base}: {e}")
            logger.error("Make sure Ollama is running and accessible")
            # We'll continue anyway, as Ollama might be started later

    @property
    def embedding_dimension(self) -> int:
        """Return the expected embedding dimension for the configured model."""
        # This is a simplification. Ideally, this would be dynamically determined
        # or retrieved from model metadata if the API supports it.
        if "nomic-embed-text" in self.embedding_model_name:
            return 768
        # Add other known models and their dimensions here
        # elif "mxbai-embed-large" in self.embedding_model_name:
        #     return 1024
        else:
            # Default or raise error if dimension is unknown
            logger.warning(
                f"Unknown embedding dimension for model '{self.embedding_model_name}'. "
                f"Falling back to default 768. This might cause issues."
            )
            return 768 # Fallback dimension

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for the given text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        if self.mock_mode:
            # In mock mode, generate a deterministic embedding based on the text
            return self._generate_mock_embedding(text)
            
        try:
            payload = {
                "model": self.embedding_model_name, # Use the stored embedding model name
                "prompt": text
            }
            
            logger.debug(f"Sending request to Ollama embedding API: {self.embedding_endpoint} with model {self.embedding_model_name}")
            response = requests.post(self.embedding_endpoint, json=payload)
            response.raise_for_status()
            embedding = response.json().get("embedding")
            if not embedding:
                logger.error("Ollama embedding API returned no embedding data.")
                return self._generate_mock_embedding(text) # Fallback
            logger.info(f"Successfully generated embedding for text starting with: {text[:50]}...")
            return np.array(embedding)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error generating embedding via Ollama API: {e}")
            # Include response details if available
            error_detail = ""
            if e.response is not None:
                try:
                    error_detail = e.response.json().get('error', e.response.text)
                except json.JSONDecodeError:
                    error_detail = e.response.text
            logger.error(f"Ollama API error detail: {error_detail}")
            logger.error("Falling back to basic mock embedding.") # Modified log message
            return self._generate_mock_embedding(text) # Call the dedicated mock method
        except Exception as e:
            logger.exception(f"An unexpected error occurred during embedding generation: {e}")
            logger.error("Falling back to basic mock embedding due to unexpected error.")
            return self._generate_mock_embedding(text) # Call the dedicated mock method

    def _generate_mock_embedding(self, text: str) -> np.ndarray:
        """Generates a deterministic mock embedding vector for testing/fallback."""
        # Use a hash of the text for deterministic randomness
        text_hash = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)
        # Ensure seed is within the valid range for np.random.seed (0 to 2**32 - 1)
        seed_value = text_hash % (2**32) 
        np.random.seed(seed_value)
        
        # Use the embedding_dimension property to get the correct size
        expected_dim = self.embedding_dimension 
        embedding = np.random.randn(expected_dim).astype(np.float32) # Ensure float32
        
        # Normalize to unit length to mimic real embeddings
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, current_query: Optional[str] = None) -> str:
        """
        Generate a response for the given prompt using the Ollama API.

        Args:
            prompt: User prompt (potentially including history)
            system_prompt: Optional system prompt with context
            current_query: The actual current user query (ignored in non-mock mode)

        Returns:
            Generated response from Ollama or an error message.
        """
        try:
            url = f"{self.api_base}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False # Keep streaming false for simpler handling
            }
            if system_prompt:
                payload["system"] = system_prompt

            logger.info(f"Sending request to Ollama generate API: {url}")
            response = requests.post(url, json=payload, timeout=120) # Add timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            data = response.json()
            response_text = data.get("response", "").strip()
            
            if not response_text:
                 logger.warning("Ollama API returned an empty response.")
                 return "I received an empty response from the language model. Please try rephrasing your query."
                 
            logger.info("Successfully received response from Ollama API.")
            return response_text
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection Error connecting to Ollama API at {self.api_base}: {e}")
            return f"Error: Could not connect to Ollama API at {self.api_base}. Please ensure Ollama is running and accessible."
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout connecting to Ollama API: {e}")
            return "Error: The request to the Ollama API timed out. The model might be taking too long to respond."
        except requests.exceptions.RequestException as e:
            logger.error(f"Error generating response from Ollama API: {e}")
            # Check if response exists to provide more details
            error_detail = ""
            if e.response is not None:
                try:
                    error_detail = e.response.json().get('error', e.response.text)
                except json.JSONDecodeError:
                    error_detail = e.response.text
            return f"Error: Failed to get response from Ollama API. {error_detail}"
        except Exception as e:
            logger.exception(f"An unexpected error occurred during Ollama API call: {e}") # Use logger.exception to include traceback
            return f"An unexpected error occurred: {e}"

    def chat(self, messages: List[Dict[str, str]], 
             temperature: float = 0.7,
             max_tokens: int = 2048) -> str:
        """
        Engage in a chat conversation using the Ollama model.

        Args:
            messages: List of message dictionaries (e.g., {'role': 'user', 'content': '...'}) 
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated chat response or error message
        """
        try:
            url = f"{self.api_base}/api/chat"
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }

            logger.info(f"Sending request to Ollama chat API: {url}")
            response = requests.post(url, json=payload, timeout=180) # Longer timeout for chat
            response.raise_for_status()

            data = response.json()
            response_message = data.get("message", {}).get("content", "").strip()
            
            if not response_message:
                logger.warning("Ollama chat API returned an empty message.")
                return "I received an empty response from the language model."
                
            logger.info("Successfully received response from Ollama chat API.")
            return response_message
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection Error connecting to Ollama chat API at {self.api_base}: {e}")
            return f"Error: Could not connect to Ollama API at {self.api_base}. Please ensure Ollama is running."
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout connecting to Ollama chat API: {e}")
            return "Error: The request to the Ollama API timed out."
        except requests.exceptions.RequestException as e:
            logger.error(f"Error generating chat response from Ollama API: {e}")
            error_detail = ""
            if e.response is not None:
                try:
                    error_detail = e.response.json().get('error', e.response.text)
                except json.JSONDecodeError:
                    error_detail = e.response.text
            return f"Error: Failed to get chat response from Ollama API. {error_detail}"
        except Exception as e:
            logger.exception(f"An unexpected error occurred during Ollama chat API call: {e}")
            return f"An unexpected error occurred: {e}"

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary with model information
        """
        if self.mock_mode:
            return {
                "name": f"{self.model_name} (mock)",
                "model_type": "mock",
                "parameters": "N/A",
                "quantization_level": "N/A"
            }
            
        try:
            response = requests.get(f"{self.api_base}/api/show?name={self.model_name}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}

    def _log_available_models(self):
        """Log the models available according to the Ollama API."""
        try:
            tags_endpoint = f"{self.api_base}/api/tags"
            response = requests.get(tags_endpoint, timeout=10)
            response.raise_for_status()
            models_data = response.json()
            available_models = [model.get('name') for model in models_data.get('models', []) if model.get('name')]
            if available_models:
                logger.info(f"Ollama reports available models: {available_models}")
                # Check if required models are present
                if self.model_name not in available_models:
                     logger.warning(f"Required generation model '{self.model_name}' not found in Ollama list.")
                if self.embedding_model_name not in available_models:
                     logger.warning(f"Required embedding model '{self.embedding_model_name}' not found in Ollama list.")
            else:
                logger.warning("Ollama API reported no available models.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Could not retrieve model list from Ollama API ({tags_endpoint}): {e}")
        except Exception as e:
            logger.error(f"Error processing model list from Ollama: {e}")
