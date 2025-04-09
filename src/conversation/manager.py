"""
Conversation Management Module

This module maintains conversation context and manages the interaction flow
between the user and the codebase analysis system.
"""

import time
from typing import Dict, List, Any, Optional
import logging

from src.query_engine.engine import QueryEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation history and context to provide
    coherent responses across multiple interactions.
    """

    def __init__(self, query_engine: QueryEngine, max_history: int = 5):
        """
        Initialize the conversation manager.

        Args:
            query_engine: Query engine for processing queries
            max_history: Maximum number of conversation turns to keep in history
        """
        self.query_engine = query_engine
        self.max_history = max_history
        self.conversation_history = []
        self.session_start_time = time.time()

    def process_query(self, query: str) -> str:
        """
        Process a user query and generate a response.

        Args:
            query: User query string

        Returns:
            Response to the user
        """
        # Check if this is a follow-up question
        is_followup = self._is_followup_question(query)
        
        # Process the query
        if is_followup and self.conversation_history:
            # If it's a follow-up, provide context from previous exchanges
            enhanced_query = self._enhance_with_history(query)
            response = self.query_engine.answer_query(enhanced_query)
        else:
            # Otherwise, process as a new query
            response = self.query_engine.answer_query(query)
        
        # Add to conversation history
        self._add_to_history(query, response)
        
        return response

    def _is_followup_question(self, query: str) -> bool:
        """
        Determine if a query is a follow-up to previous conversation.

        Args:
            query: User query string

        Returns:
            True if it's likely a follow-up, False otherwise
        """
        # Simple heuristic: check for pronouns and references
        followup_indicators = [
            "it", "this", "that", "they", "them", "those", "these",
            "the", "above", "previous", "mentioned", "you said"
        ]
        
        query_lower = query.lower()
        
        for indicator in followup_indicators:
            if f" {indicator} " in f" {query_lower} ":
                return True
        
        return False

    def _enhance_with_history(self, query: str) -> str:
        """
        Enhance a query with conversation history context.

        Args:
            query: User query string

        Returns:
            Enhanced query with context
        """
        if not self.conversation_history:
            return query
        
        # Get the most recent exchanges
        recent_history = self.conversation_history[-self.max_history:]
        
        # Format the history as context
        context = "Previous conversation:\n"
        for turn in recent_history:
            context += f"User: {turn['query']}\n"
            context += f"Assistant: {turn['response']}\n"
        
        context += f"\nNew question: {query}"
        
        return context

    def _add_to_history(self, query: str, response: str) -> None:
        """
        Add a query-response pair to the conversation history.

        Args:
            query: User query
            response: System response
        """
        self.conversation_history.append({
            "query": query,
            "response": response,
            "timestamp": time.time()
        })
        
        # Trim history if it exceeds the maximum
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history:]

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the conversation history.

        Returns:
            List of conversation turns
        """
        return self.conversation_history

    def clear_history(self) -> None:
        """
        Clear the conversation history.
        """
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_session_info(self) -> Dict[str, Any]:
        """
        Get information about the current session.

        Returns:
            Dictionary with session information
        """
        return {
            "session_start": self.session_start_time,
            "session_duration": time.time() - self.session_start_time,
            "num_exchanges": len(self.conversation_history),
        }
