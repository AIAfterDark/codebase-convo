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

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.indexing.indexer import CodebaseIndexer
from src.vector_db.vector_store import VectorStore
from src.query_engine.engine import QueryEngine
from src.llm_interface.ollama_client import OllamaClient
from src.conversation.manager import ConversationManager


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
        default="llama3.2",
        help="Ollama model to use (default: llama3.2)",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuilding the codebase index",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    # Validate codebase path
    codebase_path = Path(args.codebase_path)
    if not codebase_path.exists() or not codebase_path.is_dir():
        print(f"Error: Codebase path '{args.codebase_path}' does not exist or is not a directory")
        sys.exit(1)
    
    print(f"Analyzing codebase at: {codebase_path}")
    
    # Initialize components
    indexer = CodebaseIndexer(codebase_path)
    vector_store = VectorStore()
    llm_client = OllamaClient(model_name=args.model)
    query_engine = QueryEngine(vector_store, llm_client)
    conversation_manager = ConversationManager(query_engine)
    
    # Index the codebase if needed
    if args.rebuild_index or not vector_store.index_exists():
        print("Indexing codebase... This may take a while.")
        indexer.index_codebase()
        vector_store.save_index()
        print("Indexing complete.")
    else:
        print("Using existing codebase index.")
    
    # Start the conversation loop
    print("\nWelcome to Codebase Convo!")
    print("Ask questions about your codebase or type 'exit' to quit.")
    
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ("exit", "quit", "q"):
                break
                
            response = conversation_manager.process_query(user_input)
            print(f"\n{response}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
    
    print("Thank you for using Codebase Convo!")


if __name__ == "__main__":
    main()
