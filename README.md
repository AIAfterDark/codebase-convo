# Codebase Convo

A tool to analyze a codebase based on user direction. This tool allows users to "ask the codebase" questions and receive accurate responses that leverage the codebase itself as the source of information.

## Features

- Code indexing and processing
- Vector database for semantic search using Chroma
- Efficient similarity search with HNSW algorithm
- Query engine for effective searches
- Integration with Ollama LLM (llama3.2)
- Conversation management
- Mock mode for testing without Ollama

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/codebase-convo.git
cd codebase-convo

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Online installation - Pull the required Ollama model for embeddings
ollama pull nomic-embed-text
```

### Offline Installation

For offline environments, follow these steps:

1. Download the GGUF model file from Hugging Face:
   - Go to: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF
   - Download the `nomic-embed-text-v1.5.Q4_K_M.gguf` file (84.1 MB)

2. Move the downloaded file to your offline machine

3. Create a modelfile named `nomic-embed-text.modelfile` with the following content:
   ```
   FROM nomic-embed-text-v1.5.Q4_K_M.gguf
   PARAMETER temperature 0.0
   PARAMETER embedding true
   PARAMETER mirostat 0
   PARAMETER num_ctx 2048
   ```

4. Import the model to Ollama:
   ```bash
   # Make sure both the GGUF file and modelfile are in the same directory
   ollama create nomic-embed-text -f nomic-embed-text.modelfile
   ```

5. Verify the model is available:
   ```bash
   ollama list
   ```

## Usage

```bash
# Basic usage
python main.py --codebase-path /path/to/your/codebase

# Rebuild the index
python main.py --codebase-path /path/to/your/codebase --rebuild-index

# Use mock mode (no Ollama required)
python main.py --codebase-path /path/to/your/codebase --mock-mode
```

## Testing

The project includes test scripts to verify functionality:

```bash
# Test the Chroma vector database implementation
python test_chroma.py

# Test the application with mock data
python test_app.py
```

## Project Structure

```
codebase-convo/
├── src/
│   ├── indexing/        # Code indexing & processing
│   ├── vector_db/       # Chroma vector database implementation
│   ├── query_engine/    # Query processing and search
│   ├── llm_interface/   # Ollama LLM integration
│   └── conversation/    # Conversation management
├── tests/               # Unit and integration tests
├── main.py              # Application entry point
├── requirements.txt     # Project dependencies
├── test_chroma.py       # Test script for Chroma implementation
├── test_app.py          # Test script for application
└── README.md            # Project documentation
```

## Vector Database

The application uses Chroma, a specialized vector database, for storing and retrieving code embeddings:

- **Efficient Similarity Search**: Uses HNSW (Hierarchical Navigable Small World) algorithm for fast nearest-neighbor search
- **Persistent Storage**: Embeddings are stored on disk for persistence between runs
- **Metadata Management**: Stores code chunks with associated metadata for rich retrieval
- **Cosine Similarity**: Uses cosine similarity for comparing embeddings

## License

MIT
