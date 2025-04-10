"""
Vector Database Module

This module handles the storage and retrieval of code embeddings
for semantic search capabilities using Chroma vector database.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np
import chromadb
from chromadb.utils import embedding_functions

# Assume OllamaClient is importable for type hinting
from src.llm_interface.ollama_client import OllamaClient 

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages embeddings using Chroma vector database for persistence.
    Stores embeddings and associated metadata.
    Provides similarity search capabilities.
    """

    def __init__(self, db_path: str, embedding_client: OllamaClient):
        """
        Initialize the vector store with a Chroma database.

        Args:
            db_path: Path to the Chroma database directory.
            embedding_client: An instance of OllamaClient to get embedding dimension.
        """
        self.db_path = Path(db_path)
        self.embedding_client = embedding_client
        self.embedding_dim = self.embedding_client.embedding_dimension
        
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma client with persistent storage
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Create or get the collection for code embeddings
        self.collection = self._create_collection()
        
        logger.info(f"VectorStore initialized with Chroma database: {self.db_path}")

    def _create_collection(self):
        """Create or get the collection for code embeddings."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name="code_embeddings")
            logger.debug(f"Using existing Chroma collection 'code_embeddings'")
            return collection
        except Exception:
            # Create new collection if it doesn't exist
            logger.debug(f"Creating new Chroma collection 'code_embeddings'")
            return self.client.create_collection(
                name="code_embeddings",
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )

    def add_embedding(self, chunk_id: str, embedding: np.ndarray, chunk_data: Dict[str, Any]):
        """
        Add or update an embedding and its metadata in the database.

        Args:
            chunk_id: Unique identifier for the code chunk.
            embedding: The embedding vector (NumPy array).
            chunk_data: Dictionary containing metadata about the chunk.
        """
        try:
            # Validate embedding dimensions
            if embedding.ndim != 1 or embedding.shape[0] != self.embedding_dim:
                logger.error(f"Embedding dimension mismatch for chunk {chunk_id}. Expected {self.embedding_dim}, got {embedding.shape}. Skipping.")
                return
                
            # Convert numpy array to list for Chroma
            embedding_list = embedding.tolist()
            
            # Prepare metadata - Chroma only accepts string values in metadata
            # So we need to convert all values to strings
            metadata = {}
            for key, value in chunk_data.items():
                if key != 'code':  # Store code separately as document text
                    if isinstance(value, (dict, list)):
                        metadata[key] = str(value)
                    else:
                        metadata[key] = str(value)
            
            # Get the code content
            document = chunk_data.get('code', '')
            
            # Check if the document already exists
            existing_ids = self.collection.get(ids=[chunk_id], include=[])['ids']
            if existing_ids and chunk_id in existing_ids:
                # Update existing document
                self.collection.update(
                    ids=[chunk_id],
                    embeddings=[embedding_list],
                    metadatas=[metadata],
                    documents=[document]
                )
                logger.debug(f"Updated embedding for chunk: {chunk_id}")
            else:
                # Add new document
                self.collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding_list],
                    metadatas=[metadata],
                    documents=[document]
                )
                logger.debug(f"Added new embedding for chunk: {chunk_id}")
                
        except Exception as e:
            logger.error(f"Error adding embedding for chunk {chunk_id}: {e}")

    def get_all_embeddings(self) -> List[Tuple[str, np.ndarray, Dict[str, Any]]]:
        """
        Retrieve all embeddings and their metadata from the database.
        
        Returns:
            A list of tuples, each containing (chunk_id, embedding, metadata).
        """
        try:
            # Get all items from the collection
            result = self.collection.get(include=["embeddings", "metadatas", "documents"])
            
            if not result['ids']:
                logger.debug("No embeddings found in the database.")
                return []
                
            embeddings_data = []
            for i, chunk_id in enumerate(result['ids']):
                # Convert embedding back to numpy array
                embedding = np.array(result['embeddings'][i])
                
                # Reconstruct the original metadata with code
                metadata = result['metadatas'][i].copy() if result['metadatas'][i] else {}
                metadata['code'] = result['documents'][i]
                
                embeddings_data.append((chunk_id, embedding, metadata))
                
            logger.debug(f"Retrieved {len(embeddings_data)} embeddings from database.")
            return embeddings_data
        except Exception as e:
            logger.error(f"Error retrieving embeddings: {e}")
            return []

    def find_similar(self, query_embedding: np.ndarray, top_n: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find the top_n most similar code chunks to the query embedding.

        Args:
            query_embedding: The embedding vector for the query.
            top_n: The number of similar chunks to return.

        Returns:
            A list of tuples, each containing (chunk_data, similarity_score).
            Sorted by similarity score in descending order.
        """
        if query_embedding.ndim != 1 or query_embedding.shape[0] != self.embedding_dim:
            logger.error(f"Query embedding dimension mismatch. Expected {self.embedding_dim}, got {query_embedding.shape}. Cannot perform search.")
            return []

        try:
            # Check if the collection is empty
            count_info = self.collection.count()
            if count_info == 0:
                logger.warning("No embeddings found in the database to search against.")
                return []
                
            # Convert numpy array to list for Chroma
            query_embedding_list = query_embedding.tolist()
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=min(top_n, count_info),  # Don't request more results than we have
                include=["metadatas", "documents", "distances"]
            )
            
            if not results['ids'][0]:
                logger.warning("No similar chunks found in the database.")
                return []
            
            similarities = []
            for i, chunk_id in enumerate(results['ids'][0]):
                # Get metadata and document
                metadata = results['metadatas'][0][i].copy() if results['metadatas'][0][i] else {}
                document = results['documents'][0][i]
                
                # Add code back to metadata
                metadata['code'] = document
                
                # Convert distance to similarity score (Chroma uses distance, lower is better)
                # For cosine distance, similarity = 1 - distance
                distance = results['distances'][0][i]
                similarity = 1.0 - distance
                
                similarities.append((metadata, float(similarity)))
                
            logger.debug(f"Found {len(similarities)} similar chunks.")
            return similarities
            
        except Exception as e:
            logger.error(f"Error finding similar chunks: {e}")
            return []

    def clear_all_embeddings(self):
        """Delete all entries from the embeddings collection."""
        try:
            # Delete the collection and recreate it
            try:
                self.client.delete_collection(name="code_embeddings")
            except Exception:
                # Collection might not exist
                pass
                
            self.collection = self._create_collection()
            logger.info("Cleared all embeddings from the database.")
        except Exception as e:
            logger.error(f"Error clearing embeddings: {e}")

    def close(self):
        """Close the database connection."""
        # Chroma handles connections automatically, but we'll keep this method for API compatibility
        logger.info("Closed Chroma database connection.")

    def __del__(self):
        """Ensure database connection is closed when the object is destroyed."""
        self.close()
