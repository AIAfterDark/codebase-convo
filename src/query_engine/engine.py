"""
Query Engine Module

This module processes user queries and retrieves relevant code chunks
from the vector store to generate accurate responses.
"""

import logging
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Processes user queries and retrieves relevant code chunks
    to generate accurate responses.
    """

    def __init__(self, vector_store, llm_client):
        """
        Initialize the query engine.

        Args:
            vector_store: Vector store for similarity search
            llm_client: LLM client for generating embeddings and responses
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.top_k = 5  # Number of relevant chunks to retrieve
        
    def process_query(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Process a user query and generate a response.

        Args:
            query: User query
            conversation_history: Optional conversation history for context

        Returns:
            Generated response
        """
        logger.info(f"Processing query: {query}")
        
        # Generate embedding for the query
        query_embedding = self.llm_client.generate_embedding(query)
        
        # Search for relevant code chunks
        similar_results = self.vector_store.find_similar(query_embedding, top_n=self.top_k)
        
        if not similar_results:
            logger.warning("No relevant code chunks found")
            return "I couldn't find any relevant code in the codebase for your query. Could you please rephrase or provide more details?"
        
        logger.info(f"Found {len(similar_results)} relevant code chunks")
        
        # Convert the results to the expected format for _format_chunks_for_llm
        # Each result from find_similar is a tuple of (metadata, similarity_score)
        relevant_chunks = []
        for metadata, similarity in similar_results:
            # Create a dictionary with the metadata and add the similarity score
            chunk_dict = metadata.copy()  # Copy the metadata dictionary
            chunk_dict["similarity"] = similarity
            relevant_chunks.append(chunk_dict)
        
        # Format the chunks for the LLM
        formatted_chunks = self._format_chunks_for_llm(relevant_chunks)
        
        # Generate system prompt
        system_prompt = self._generate_system_prompt(formatted_chunks)
        
        # Format user query with conversation history if available
        formatted_query = self._format_query_with_history(query, conversation_history)
        
        # Generate response using the LLM
        response = self.llm_client.generate_response(
            prompt=formatted_query,
            system_prompt=system_prompt,
            current_query=query  # Pass the original query for mock logic
        )
        
        return response
    
    def _format_chunks_for_llm(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format code chunks for the LLM.

        Args:
            chunks: List of code chunks

        Returns:
            Formatted chunks as a string
        """
        formatted_text = "RELEVANT CODE CHUNKS:\n\n"
        
        for i, chunk in enumerate(chunks, 1):
            similarity = chunk.get("similarity", 0.0)
            file_path = chunk.get("file_path", "unknown")
            language = chunk.get("language", "unknown")
            chunk_type = chunk.get("chunk_type", "code")
            chunk_name = chunk.get("chunk_name", "unnamed")
            code = chunk.get("code", "")
            
            formatted_text += f"CHUNK {i} (Similarity: {similarity:.2f}):\n"
            formatted_text += f"File: {file_path}\n"
            formatted_text += f"Type: {chunk_type} {chunk_name}\n"
            formatted_text += f"Language: {language}\n"
            formatted_text += "```\n"
            formatted_text += code + "\n"
            formatted_text += "```\n\n"
        
        return formatted_text
    
    def _generate_system_prompt(self, formatted_chunks: str) -> str:
        """
        Generate a system prompt for the LLM.

        Args:
            formatted_chunks: Formatted code chunks

        Returns:
            System prompt
        """
        return f"""You are a helpful code analysis assistant. Your task is to answer questions about a codebase.
Below are code chunks from the codebase that are most relevant to the user's query.
Use these code chunks to provide accurate, helpful responses.

{formatted_chunks}

When answering:
1. Focus on the code chunks provided above.
2. If the code chunks don't contain enough information to answer the query, say so.
3. Be concise but thorough in your explanations.
4. If you identify security issues or bugs, point them out.
5. If you're asked about implementation details, refer to the specific code in your answer.
"""
    
    def _format_query_with_history(self, query: str, conversation_history: Optional[List[Dict[str, str]]]) -> str:
        """
        Format the query with conversation history if available.

        Args:
            query: User query
            conversation_history: Conversation history

        Returns:
            Formatted query
        """
        if not conversation_history or len(conversation_history) <= 1:
            return query
        
        # Get the last few turns of conversation for context
        # Skip the current query which is already at the end of the history
        relevant_history = conversation_history[:-1]
        if len(relevant_history) > 4:  # Limit to last 2 turns (4 messages)
            relevant_history = relevant_history[-4:]
        
        formatted_query = "Previous conversation:\n"
        for message in relevant_history:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            formatted_query += f"{role.capitalize()}: {content}\n"
        
        formatted_query += f"\nCurrent query: {query}"
        
        return formatted_query
