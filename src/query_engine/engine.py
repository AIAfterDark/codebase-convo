"""
Query Engine Module

This module transforms user questions into effective searches against
the vector database and assembles context for the LLM.
"""

import re
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from src.vector_db.vector_store import VectorStore
from src.llm_interface.ollama_client import OllamaClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Processes user queries, performs vector search, and assembles
    context for the LLM to generate responses.
    """

    def __init__(self, vector_store: VectorStore, llm_client: OllamaClient):
        """
        Initialize the query engine.

        Args:
            vector_store: Vector store for searching code chunks
            llm_client: Client for the Ollama LLM
        """
        self.vector_store = vector_store
        self.llm_client = llm_client

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and retrieve relevant code chunks.

        Args:
            query: User query string

        Returns:
            Dictionary with query results and context
        """
        # Clean and preprocess the query
        cleaned_query = self._preprocess_query(query)
        
        # Generate query embedding
        query_embedding = self._generate_embedding(cleaned_query)
        
        # Perform vector search
        search_results = self.vector_store.search(query_embedding, top_k=5)
        
        # Enhance results with additional context
        enriched_results = self._enhance_with_context(search_results)
        
        # Assemble context window for the LLM
        context = self._assemble_context(enriched_results, query)
        
        return {
            "query": query,
            "cleaned_query": cleaned_query,
            "results": enriched_results,
            "context": context
        }

    def _preprocess_query(self, query: str) -> str:
        """
        Clean and preprocess the query.

        Args:
            query: Raw user query

        Returns:
            Cleaned query
        """
        # Convert to lowercase
        query = query.lower()
        
        # Remove special characters except spaces and alphanumerics
        query = re.sub(r'[^\w\s]', ' ', query)
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query

    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for the given text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # In a real implementation, this would use a proper embedding model
        # For now, we'll use a simple placeholder implementation
        
        # Ask the LLM client to generate an embedding
        embedding = self.llm_client.generate_embedding(text)
        
        return embedding

    def _enhance_with_context(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance search results with additional context.

        Args:
            results: Search results from vector store

        Returns:
            Enhanced results with additional context
        """
        enhanced_results = []
        
        for result in results:
            # Create a copy to avoid modifying the original
            enhanced_result = result.copy()
            
            # Add file context if available
            file_path = result.get("file_path")
            if file_path:
                # In a real implementation, we might look up additional file metadata
                # or related files here
                pass
            
            # Add function/class context if available
            chunk_type = result.get("chunk_type")
            chunk_name = result.get("chunk_name")
            if chunk_type and chunk_name:
                # In a real implementation, we might look up additional information
                # about the function or class here
                pass
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results

    def _assemble_context(self, results: List[Dict[str, Any]], query: str) -> str:
        """
        Assemble context window for the LLM.

        Args:
            results: Enhanced search results
            query: Original user query

        Returns:
            Assembled context as a string
        """
        context_parts = []
        
        # Add introduction
        context_parts.append(f"The user asked: {query}")
        context_parts.append("Here are the most relevant code snippets:")
        
        # Add code snippets with context
        for i, result in enumerate(results, 1):
            file_path = result.get("file_path", "unknown_file")
            language = result.get("language", "unknown")
            similarity = result.get("similarity", 0.0)
            code = result.get("code", "")
            
            # Truncate code if too long
            if len(code) > 1500:  # Arbitrary limit to avoid context window issues
                code = code[:1500] + "...[truncated]"
            
            context_parts.append(f"\n--- Snippet {i} (Relevance: {similarity:.2f}) ---")
            context_parts.append(f"File: {file_path}")
            context_parts.append(f"Language: {language}")
            
            if "chunk_type" in result and "chunk_name" in result:
                context_parts.append(f"Type: {result['chunk_type']}")
                context_parts.append(f"Name: {result['chunk_name']}")
            
            if "start_line" in result and "end_line" in result:
                context_parts.append(f"Lines: {result['start_line']}-{result['end_line']}")
            
            context_parts.append(f"```{language}")
            context_parts.append(code)
            context_parts.append("```")
        
        # Add instructions for the LLM
        context_parts.append("\nBased on the code snippets above, please answer the user's question.")
        context_parts.append("If the information in the snippets is not sufficient, please say so.")
        
        return "\n".join(context_parts)

    def answer_query(self, query: str) -> str:
        """
        Process a query and generate an answer using the LLM.

        Args:
            query: User query

        Returns:
            LLM-generated answer
        """
        # Process the query to get context
        query_result = self.process_query(query)
        
        # Generate answer using the LLM
        answer = self.llm_client.generate_response(query_result["context"])
        
        return answer
