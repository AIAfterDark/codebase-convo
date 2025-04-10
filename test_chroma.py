#!/usr/bin/env python3
"""
Test script for Chroma vector database implementation
"""

import numpy as np
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from src.vector_db.vector_store import VectorStore
from src.llm_interface.ollama_client import OllamaClient

def main():
    """Test the Chroma vector store implementation"""
    
    # Create a test directory for Chroma
    test_db_path = Path("./test_chroma_db")
    test_db_path.mkdir(exist_ok=True)
    
    # Initialize with mock mode to avoid Ollama dependency
    ollama_client = OllamaClient(
        api_base="http://localhost:11434",
        model_name="llama3.2:latest",
        embedding_model_name="nomic-embed-text",
        mock_mode=True  # Use mock mode to avoid needing Ollama running
    )
    
    # Initialize the vector store
    vector_store = VectorStore(
        db_path=str(test_db_path),
        embedding_client=ollama_client
    )
    
    # Create some test embeddings
    test_embeddings = []
    test_chunks = []
    
    # Create 5 test chunks with random embeddings
    for i in range(5):
        chunk_id = f"test_chunk_{i}"
        # Create a deterministic random embedding for testing
        np.random.seed(i)
        embedding = np.random.rand(ollama_client.embedding_dimension).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        # Create test chunk data
        chunk_data = {
            "id": chunk_id,
            "file_path": f"test_file_{i}.py",
            "language": "python",
            "chunk_type": "function",
            "chunk_name": f"test_function_{i}",
            "code": f"def test_function_{i}():\n    return 'This is test function {i}'"
        }
        
        test_embeddings.append((chunk_id, embedding))
        test_chunks.append(chunk_data)
        
        # Add to vector store
        logger.info(f"Adding test chunk {chunk_id}")
        vector_store.add_embedding(chunk_id, embedding, chunk_data)
    
    # Test retrieval of all embeddings
    logger.info("Testing retrieval of all embeddings")
    all_embeddings = vector_store.get_all_embeddings()
    logger.info(f"Retrieved {len(all_embeddings)} embeddings")
    
    # Test similarity search
    logger.info("Testing similarity search")
    # Use the first embedding as the query
    query_embedding = test_embeddings[0][1]
    similar_chunks = vector_store.find_similar(query_embedding, top_n=3)
    
    logger.info(f"Found {len(similar_chunks)} similar chunks")
    for i, (chunk, similarity) in enumerate(similar_chunks):
        logger.info(f"Similar chunk {i+1}: {chunk.get('chunk_name')} (similarity: {similarity:.4f})")
    
    # Test clearing all embeddings
    logger.info("Testing clearing all embeddings")
    vector_store.clear_all_embeddings()
    
    # Verify that all embeddings are cleared
    all_embeddings_after_clear = vector_store.get_all_embeddings()
    logger.info(f"After clearing, there are {len(all_embeddings_after_clear)} embeddings")
    
    logger.info("All tests completed successfully!")

if __name__ == "__main__":
    main()
