# Codebase Convo

A tool to analyze a codebase based on user direction. This tool allows users to "ask the codebase" questions and receive accurate responses that leverage the codebase itself as the source of information.

## Features

- Code indexing and processing
- Vector database for semantic search
- Query engine for effective searches
- Integration with Ollama LLM (llama3.2)
- Conversation management

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
```

## Usage

```bash
python main.py --codebase-path /path/to/your/codebase
```

## Project Structure

```
codebase-convo/
├── src/
│   ├── indexing/        # Code indexing & processing
│   ├── vector_db/       # Vector database implementation
│   ├── query_engine/    # Query processing and search
│   ├── llm_interface/   # Ollama LLM integration
│   └── conversation/    # Conversation management
├── tests/               # Unit and integration tests
├── main.py              # Application entry point
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

## License

MIT
