"""
Vector Database Module

This module handles the storage and retrieval of code embeddings
for semantic search capabilities.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# We'll use a simple implementation for now
# In a production environment, you might want to use FAISS or another vector database
class VectorStore:
    """
    Stores and manages vector embeddings for code chunks.
    Provides similarity search capabilities.
    """

    def __init__(self, embedding_dim: int = 384, index_path: str = "./.index"):
        """
        Initialize the vector store.

        Args:
            embedding_dim: Dimension of the embeddings
            index_path: Path to store the index files
        """
        self.embedding_dim = embedding_dim
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Dictionary to store chunk_id -> embedding mapping
        self.embeddings: Dict[str, np.ndarray] = {}
        
        # Dictionary to store chunk_id -> chunk_data mapping
        self.chunk_data: Dict[str, Dict[str, Any]] = {}

    def add_embedding(self, chunk_id: str, embedding: np.ndarray, chunk_data: Dict[str, Any]) -> None:
        """
        Add an embedding to the vector store.

        Args:
            chunk_id: Unique identifier for the chunk
            embedding: Vector embedding of the chunk
            chunk_data: Metadata and content of the chunk
        """
        # Validate embedding dimension
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Expected embedding dimension {self.embedding_dim}, got {embedding.shape[0]}")
        
        # Store the embedding and chunk data
        self.embeddings[chunk_id] = embedding
        self.chunk_data[chunk_id] = chunk_data

    def add_embeddings(self, embeddings: List[Tuple[str, np.ndarray, Dict[str, Any]]]) -> None:
        """
        Add multiple embeddings to the vector store.

        Args:
            embeddings: List of (chunk_id, embedding, chunk_data) tuples
        """
        for chunk_id, embedding, chunk_data in embeddings:
            self.add_embedding(chunk_id, embedding, chunk_data)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks based on embedding similarity.

        Args:
            query_embedding: Embedding of the query
            top_k: Number of results to return

        Returns:
            List of chunk data with similarity scores
        """
        if not self.embeddings:
            logger.warning("Vector store is empty, no results to return")
            return []
        
        # Calculate cosine similarity between query and all embeddings
        similarities = {}
        for chunk_id, embedding in self.embeddings.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities[chunk_id] = similarity
        
        # Sort by similarity (descending)
        sorted_chunk_ids = sorted(similarities.keys(), key=lambda k: similarities[k], reverse=True)
        
        # Get top-k results
        results = []
        for chunk_id in sorted_chunk_ids[:top_k]:
            chunk_result = self.chunk_data[chunk_id].copy()
            chunk_result["similarity"] = float(similarities[chunk_id])
            results.append(chunk_result)
        
        return results

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity (between -1 and 1)
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def save_index(self) -> None:
        """
        Save the vector store to disk.
        """
        index_file = self.index_path / "vector_index.pkl"
        data_file = self.index_path / "chunk_data.pkl"
        
        with open(index_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        
        with open(data_file, 'wb') as f:
            pickle.dump(self.chunk_data, f)
        
        logger.info(f"Saved vector store with {len(self.embeddings)} embeddings to {self.index_path}")

    def load_index(self) -> bool:
        """
        Load the vector store from disk.

        Returns:
            True if successful, False otherwise
        """
        index_file = self.index_path / "vector_index.pkl"
        data_file = self.index_path / "chunk_data.pkl"
        
        if not index_file.exists() or not data_file.exists():
            logger.warning("Index files not found, cannot load")
            return False
        
        try:
            with open(index_file, 'rb') as f:
                self.embeddings = pickle.load(f)
            
            with open(data_file, 'rb') as f:
                self.chunk_data = pickle.load(f)
            
            logger.info(f"Loaded vector store with {len(self.embeddings)} embeddings from {self.index_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False

    def index_exists(self) -> bool:
        """
        Check if the index exists on disk.

        Returns:
            True if the index exists, False otherwise
        """
        index_file = self.index_path / "vector_index.pkl"
        data_file = self.index_path / "chunk_data.pkl"
        
        return index_file.exists() and data_file.exists()

    def clear(self) -> None:
        """
        Clear the vector store.
        """
        self.embeddings = {}
        self.chunk_data = {}
        logger.info("Vector store cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with statistics
        """
        return {
            "num_embeddings": len(self.embeddings),
            "embedding_dim": self.embedding_dim,
            "index_path": str(self.index_path),
            "index_size_mb": self._get_index_size_mb(),
        }

    def _get_index_size_mb(self) -> float:
        """
        Get the size of the index files in MB.

        Returns:
            Size in MB
        """
        index_file = self.index_path / "vector_index.pkl"
        data_file = self.index_path / "chunk_data.pkl"
        
        total_size = 0
        if index_file.exists():
            total_size += index_file.stat().st_size
        if data_file.exists():
            total_size += data_file.stat().st_size
        
        return total_size / (1024 * 1024)  # Convert bytes to MB
