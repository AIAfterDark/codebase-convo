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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for interacting with the Ollama API to generate
    embeddings and text responses.
    """

    def __init__(self, model_name: str = "llama3.2", api_base: str = "http://localhost:11434"):
        """
        Initialize the Ollama client.

        Args:
            model_name: Name of the Ollama model to use
            api_base: Base URL for the Ollama API
        """
        self.model_name = model_name
        self.api_base = api_base.rstrip('/')
        self.embedding_endpoint = f"{self.api_base}/api/embeddings"
        self.generate_endpoint = f"{self.api_base}/api/generate"
        
        # Verify connection to Ollama
        self._check_connection()

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

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for the given text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": text
            }
            
            response = requests.post(self.embedding_endpoint, json=payload)
            response.raise_for_status()
            
            embedding = response.json().get("embedding", [])
            
            if not embedding:
                logger.warning("Received empty embedding from Ollama")
                # Return a random embedding as a fallback
                return np.random.randn(384)  # Using a common embedding dimension
            
            return np.array(embedding)
            
        except requests.RequestException as e:
            logger.error(f"Error generating embedding: {e}")
            # Return a random embedding as a fallback
            return np.random.randn(384)  # Using a common embedding dimension

    def generate_response(self, prompt: str, 
                         system_prompt: Optional[str] = None,
                         temperature: float = 0.7,
                         max_tokens: int = 2048) -> str:
        """
        Generate a text response using the Ollama model.

        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt for context
            temperature: Sampling temperature (higher = more creative)
            max_tokens: Maximum number of tokens to generate

        Returns:
            Generated text response
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = requests.post(self.generate_endpoint, json=payload, stream=False)
            response.raise_for_status()
            
            return response.json().get("response", "")
            
        except requests.RequestException as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: Failed to generate response. Please check if Ollama is running with the {self.model_name} model."

    def chat(self, messages: List[Dict[str, str]], 
            temperature: float = 0.7,
            max_tokens: int = 2048) -> str:
        """
        Generate a response in a chat format.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response text
        """
        try:
            # Extract system message if present
            system_prompt = None
            filtered_messages = []
            
            for message in messages:
                if message.get("role") == "system":
                    system_prompt = message.get("content", "")
                else:
                    filtered_messages.append(message)
            
            # Format messages as a conversation
            conversation = ""
            for message in filtered_messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                if role == "user":
                    conversation += f"User: {content}\n"
                elif role == "assistant":
                    conversation += f"Assistant: {content}\n"
            
            # Add final prompt for the model to respond to
            conversation += "Assistant: "
            
            return self.generate_response(
                prompt=conversation,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"Error: Failed to generate chat response. Please check if Ollama is running with the {self.model_name} model."

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary with model information
        """
        try:
            response = requests.get(f"{self.api_base}/api/show?name={self.model_name}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}
