#!/usr/bin/env python3
"""
Codebase Convo - Main Entry Point

A tool to analyze a codebase based on user direction, allowing users to
"ask the codebase" questions and receive accurate responses.
"""

import argparse
import os
import sys
from pathlib import Path
import logging
import time

# Add the src directory to the path so we can import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import custom modules AFTER sys.path is updated
from src.indexing.indexer import CodebaseIndexer
from src.vector_db.vector_store import VectorStore
from src.query_engine.engine import QueryEngine
from src.llm_interface.ollama_client import OllamaClient
from src.conversation.manager import ConversationManager

# Set up logging
def setup_logging(args):
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    logging.basicConfig(level=numeric_level, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Suppress overly verbose libraries if needed
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Codebase Convo - Ask questions about your codebase"
    )
    parser.add_argument(
        "--codebase-path",
        type=str,
        required=True,
        help="Path to the codebase to analyze",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:latest",
        help="Ollama chat/generation model to use (default: llama3.2:latest)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="nomic-embed-text",
        help="Ollama embedding model to use (default: nomic-embed-text)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Base URL for the Ollama API (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="codebase_vectors.db",
        help="Path to the vector database file (default: codebase_vectors.db)",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuilding the codebase index",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--mock-mode",
        action="store_true",
        help="Run in mock mode without requiring Ollama to be running",
    )
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    setup_logging(args)
    
    # Validate codebase path
    codebase_path = Path(args.codebase_path)
    if not codebase_path.exists() or not codebase_path.is_dir():
        logger.error(f"Codebase path '{args.codebase_path}' does not exist or is not a directory")
        sys.exit(1)
    
    logger.info(f"Analyzing codebase at: {codebase_path}")
    
    # Initialize components
    logger.info(f"Initializing components...")
    llm_client = OllamaClient(
        api_base=args.ollama_url,
        model_name=args.model,
        embedding_model_name=args.embedding_model,
        mock_mode=args.mock_mode  # Use mock mode if specified
    )
    vector_store = VectorStore(
        db_path=args.db_path, 
        embedding_client=llm_client
    )
    indexer = CodebaseIndexer(codebase_path=codebase_path)
    query_engine = QueryEngine(vector_store, llm_client)
    conversation_manager = ConversationManager(query_engine)
    
    db_path = Path(args.db_path)

    # --- Decide whether to index or load ---
    should_rebuild = args.rebuild_index
    logger.debug(f"Before check: should_rebuild={should_rebuild}, db_path exists={db_path.exists()}")
    if not should_rebuild and not db_path.exists():
        logger.info(f"Vector database '{db_path}' not found. Forcing index rebuild.")
        should_rebuild = True # Force rebuild if DB doesn't exist and not explicitly rebuilding
    elif should_rebuild and db_path.exists():
         logger.info(f"'--rebuild-index' flag set. Removing existing database '{db_path}'...")
         try:
             os.remove(db_path)
         except OSError as e:
             logger.error(f"Error removing existing database '{db_path}': {e}. Proceeding might lead to issues.")

    if should_rebuild:
        logger.info("Building new codebase index...")
        
        # Clear existing data if rebuilding
        if args.rebuild_index:
            logger.info("Clearing existing vector database due to --rebuild-index flag.")
            vector_store.clear_all_embeddings()
            
        # First, index the codebase to get code chunks
        code_chunks = indexer.index_codebase()
        logger.info(f"Identified {len(code_chunks)} code chunks to index.")
        
        # Then, generate embeddings for each chunk and add to vector store
        for chunk in code_chunks:
            chunk_id = chunk["id"]
            # Create a text representation of the chunk for embedding
            chunk_text = f"{chunk.get('chunk_type', '')} {chunk.get('chunk_name', '')} in {chunk.get('file_path', '')}: {chunk['code']}"
            embedding = llm_client.generate_embedding(chunk_text)
            vector_store.add_embedding(chunk_id, embedding, chunk)
        
        logger.info("Indexing complete.")
    else:
        logger.info("Using existing codebase index.")
    
    # Start the conversation loop
    print("\nWelcome to Codebase Convo!")
    print("Ask questions about your codebase or type 'exit' to quit.")
    
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ("exit", "quit", "q"):
                break
            
            # Process the query
            print("Processing your query...")
            start_time = time.time()
            response = conversation_manager.process_query(user_input)
            end_time = time.time()
            
            print(f"\n{response}")
            logger.debug(f"Query processed in {end_time - start_time:.2f} seconds")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            else:
                print(f"Error: {e}")
    
    print("Thank you for using Codebase Convo!")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    main()
