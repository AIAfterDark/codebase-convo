#!/usr/bin/env python3
"""
Codebase Convo Test Script

A simplified script to test the Codebase Convo application with a cleaner interface.
"""

import os
import sys
from pathlib import Path
import logging
import argparse
import time

# Configure logging to file instead of console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='codebase_convo.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.indexing.indexer import CodebaseIndexer
from src.vector_db.vector_store import VectorStore
from src.llm_interface.ollama_client import OllamaClient
from src.query_engine.engine import QueryEngine
from src.conversation.manager import ConversationManager


def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Codebase Convo Test Script"
    )
    parser.add_argument(
        "--codebase-path",
        type=str,
        default="I:\\Programming-Nvme\\ai-LLM-CodeInsight\\Damn-Vulnerable-Source-Code",
        help="Path to the codebase to analyze"
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuilding the codebase index"
    )
    return parser.parse_args()


def print_banner():
    """Print the application banner."""
    clear_screen()
    print("=" * 80)
    print("                           CODEBASE CONVO                           ")
    print("                  Ask questions about your codebase                 ")
    print("=" * 80)
    print()


def print_test_questions(questions):
    """Print the list of test questions."""
    print("\nSuggested test questions:")
    for i, question in enumerate(questions, 1):
        print(f"{i}. {question}")
    print("\nType a number to select a question, or type your own question.")
    print("Type 'exit', 'quit', or 'q' to exit.")
    print("-" * 80)


def main():
    """Main entry point for the test script."""
    args = parse_arguments()
    
    print_banner()
    
    # Validate codebase path
    codebase_path = Path(args.codebase_path)
    if not codebase_path.exists() or not codebase_path.is_dir():
        print(f"Error: Codebase path '{args.codebase_path}' does not exist or is not a directory")
        sys.exit(1)
    
    print(f"Analyzing codebase at: {codebase_path}")
    
    # Initialize components
    indexer = CodebaseIndexer(codebase_path)
    vector_store = VectorStore()
    llm_client = OllamaClient(model_name="llama3.2")
    query_engine = QueryEngine(vector_store, llm_client)
    conversation_manager = ConversationManager(query_engine)
    
    # Try to load existing index
    index_loaded = False
    if not args.rebuild_index:
        index_loaded = vector_store.load_index()
    
    # Index the codebase if needed
    if args.rebuild_index or not index_loaded:
        print("Indexing codebase... This may take a while.")
        
        # First, index the codebase to get code chunks
        code_chunks = indexer.index_codebase()
        print(f"Created {len(code_chunks)} code chunks")
        
        # Then, generate embeddings for each chunk and add to vector store
        print("Generating embeddings...")
        for i, chunk in enumerate(code_chunks):
            sys.stdout.write(f"\rProcessing chunk {i+1}/{len(code_chunks)}")
            sys.stdout.flush()
            
            chunk_id = chunk["id"]
            # Create a text representation of the chunk for embedding
            chunk_text = f"{chunk.get('chunk_type', '')} {chunk.get('chunk_name', '')} in {chunk.get('file_path', '')}: {chunk['code']}"
            embedding = llm_client.generate_embedding(chunk_text)
            vector_store.add_embedding(chunk_id, embedding, chunk)
        
        print("\nSaving index...")
        # Save the index
        vector_store.save_index()
        print("Indexing complete.")
    else:
        print("Using existing codebase index.")
    
    # Predefined test questions to demonstrate functionality
    test_questions = [
        "What files are in this codebase?",
        "What is the main functionality of app.py?",
        "Are there any security vulnerabilities in the code?",
        "How does the admin login work?",
        "What are the routes defined in the application?"
    ]
    
    print_test_questions(test_questions)
    
    # Start the conversation loop
    while True:
        try:
            user_input = input("\n> ")
            
            if user_input.lower() in ("exit", "quit", "q"):
                break
            
            # Check if user entered a number for a test question
            try:
                question_num = int(user_input)
                if 1 <= question_num <= len(test_questions):
                    user_input = test_questions[question_num - 1]
                    print(f"Selected question: {user_input}")
            except ValueError:
                pass
            
            # Process the query
            print("\nProcessing your query...")
            start_time = time.time()
            
            try:
                response = conversation_manager.process_query(user_input)
                end_time = time.time()
                
                print(f"\nResponse (processed in {end_time - start_time:.2f} seconds):")
                print("-" * 80)
                print(response)
                print("-" * 80)
            except Exception as e:
                print(f"\nError processing query: {e}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThank you for using Codebase Convo!")


if __name__ == "__main__":
    main()
