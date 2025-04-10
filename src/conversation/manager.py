"""
Conversation Manager Module

This module manages the conversation context and history.
"""

import logging
from typing import Dict, List, Any, Optional
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation context and history.
    """

    def __init__(self, query_engine):
        """
        Initialize the conversation manager.

        Args:
            query_engine: Query engine for processing queries
        """
        self.query_engine = query_engine
        self.conversation_history = []
        self.max_history_length = 10  # Maximum number of conversation turns to keep

    def process_query(self, query: str) -> str:
        """
        Process a user query and update conversation history.

        Args:
            query: User query

        Returns:
            Response to the query
        """
        logger.info(f"Processing query: {query}")
        
        # Add user query to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": query
        })
        
        # Trim conversation history if it's too long
        if len(self.conversation_history) > self.max_history_length * 2:  # *2 because each turn has user and assistant messages
            self.conversation_history = self.conversation_history[-self.max_history_length * 2:]
        
        # Process the query with the query engine, passing the conversation history
        try:
            start_time = time.time()
            response = self.query_engine.process_query(query, self.conversation_history)
            end_time = time.time()
            
            logger.debug(f"Query processed in {end_time - start_time:.2f} seconds")
            
            # Add assistant response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_response = f"I'm sorry, I encountered an error while processing your query: {str(e)}"
            
            # Add error response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": error_response
            })
            
            return error_response
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.

        Returns:
            List of conversation messages
        """
        return self.conversation_history
    
    def clear_conversation_history(self) -> None:
        """
        Clear the conversation history.
        """
        self.conversation_history = []
        logger.info("Conversation history cleared")
