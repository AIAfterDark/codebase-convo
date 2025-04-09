"""
Code Indexing & Processing Module

This module handles scanning, parsing, and processing codebase files
to create a searchable representation.
"""

import os
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodebaseIndexer:
    """
    Handles the indexing of a codebase, including file traversal,
    parsing, and chunking of code for vector representation.
    """

    # Language extensions mapping
    LANGUAGE_EXTENSIONS = {
        "python": [".py"],
        "javascript": [".js", ".jsx", ".ts", ".tsx"],
        "rust": [".rs"],
        "c": [".c", ".h"],
        "cpp": [".cpp", ".hpp", ".cc", ".hh"],
        "assembly": [".asm", ".s"],
        "html": [".html", ".htm"],
        "css": [".css"],
        "json": [".json"],
        "markdown": [".md", ".markdown"],
    }

    def __init__(self, codebase_path: Path):
        """
        Initialize the CodebaseIndexer.

        Args:
            codebase_path: Path to the codebase to index
        """
        self.codebase_path = Path(codebase_path)
        self.indexed_files: Dict[str, Dict[str, Any]] = {}
        self.code_chunks: List[Dict[str, Any]] = []
        self.ignored_dirs = {".git", "node_modules", "venv", "__pycache__", "dist", "build"}
        self.ignored_files = {"package-lock.json", "yarn.lock"}

    def index_codebase(self) -> None:
        """
        Index the entire codebase by traversing files and processing them.
        """
        logger.info(f"Starting indexing of codebase at: {self.codebase_path}")
        
        for file_path in self._traverse_codebase():
            try:
                self._process_file(file_path)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        logger.info(f"Indexing complete. Processed {len(self.indexed_files)} files and created {len(self.code_chunks)} chunks")

    def _traverse_codebase(self) -> List[Path]:
        """
        Traverse the codebase and yield files that should be indexed.

        Returns:
            List of file paths to be indexed
        """
        files_to_index = []
        
        for root, dirs, files in os.walk(self.codebase_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignored_dirs]
            
            for file in files:
                if file in self.ignored_files:
                    continue
                
                file_path = Path(root) / file
                
                # Check if the file has a supported extension
                if self._get_language_from_extension(file_path.suffix):
                    files_to_index.append(file_path)
        
        return files_to_index

    def _get_language_from_extension(self, extension: str) -> Optional[str]:
        """
        Determine the programming language based on file extension.

        Args:
            extension: File extension including the dot (e.g., '.py')

        Returns:
            Language name or None if not recognized
        """
        for language, extensions in self.LANGUAGE_EXTENSIONS.items():
            if extension.lower() in extensions:
                return language
        return None

    def _process_file(self, file_path: Path) -> None:
        """
        Process a single file, extracting metadata and creating code chunks.

        Args:
            file_path: Path to the file to process
        """
        relative_path = file_path.relative_to(self.codebase_path)
        language = self._get_language_from_extension(file_path.suffix)
        
        if not language:
            return
        
        logger.debug(f"Processing file: {relative_path} ({language})")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            logger.warning(f"Could not decode file {relative_path} as UTF-8, skipping")
            return
        
        # Extract metadata based on language
        metadata = self._extract_metadata(content, language, file_path)
        
        # Store file metadata
        self.indexed_files[str(relative_path)] = {
            "path": str(relative_path),
            "language": language,
            "imports": metadata.get("imports", []),
            "exports": metadata.get("exports", []),
            "functions": metadata.get("functions", []),
            "classes": metadata.get("classes", []),
            "dependencies": metadata.get("dependencies", []),
            "last_modified": file_path.stat().st_mtime,
        }
        
        # Create code chunks
        self._create_chunks(content, str(relative_path), language, metadata)

    def _extract_metadata(self, content: str, language: str, file_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from file content based on language.
        
        Args:
            content: File content
            language: Programming language
            file_path: Path to the file
            
        Returns:
            Dictionary containing metadata
        """
        # This is a simplified implementation
        # In a real implementation, we would use language-specific parsers
        
        metadata = {
            "imports": [],
            "exports": [],
            "functions": [],
            "classes": [],
            "dependencies": [],
        }
        
        lines = content.split('\n')
        
        if language == "python":
            # Simple Python import detection
            for line in lines:
                line = line.strip()
                if line.startswith("import ") or line.startswith("from "):
                    metadata["imports"].append(line)
                elif line.startswith("def "):
                    # Extract function name
                    func_name = line[4:].split('(')[0].strip()
                    metadata["functions"].append(func_name)
                elif line.startswith("class "):
                    # Extract class name
                    class_name = line[6:].split('(')[0].split(':')[0].strip()
                    metadata["classes"].append(class_name)
        
        elif language in ["javascript", "typescript"]:
            # Simple JavaScript/TypeScript import/export detection
            for line in lines:
                line = line.strip()
                if line.startswith("import "):
                    metadata["imports"].append(line)
                elif line.startswith("export "):
                    metadata["exports"].append(line)
                elif "function " in line:
                    # Very simple function detection
                    parts = line.split("function ")[1].split('(')[0].strip()
                    if parts:
                        metadata["functions"].append(parts)
                elif "class " in line:
                    # Simple class detection
                    parts = line.split("class ")[1].split(' ')[0].split('{')[0].strip()
                    if parts:
                        metadata["classes"].append(parts)
        
        # More language-specific parsing would be implemented here
        
        return metadata

    def _create_chunks(self, content: str, file_path: str, language: str, metadata: Dict[str, Any]) -> None:
        """
        Create code chunks from file content.
        
        Args:
            content: File content
            file_path: Path to the file
            language: Programming language
            metadata: File metadata
        """
        lines = content.split('\n')
        
        # For simplicity, we'll create chunks based on logical units like functions and classes
        # In a real implementation, this would be more sophisticated
        
        # First, add a chunk for the entire file (useful for small files)
        if len(lines) <= 100:  # Only for reasonably sized files
            self.code_chunks.append({
                "id": f"chunk_{len(self.code_chunks)}",
                "file_path": file_path,
                "start_line": 0,
                "end_line": len(lines) - 1,
                "code": content,
                "context": f"Complete file {file_path}",
                "language": language,
                "metadata": metadata
            })
        
        # Then try to identify logical chunks like functions and classes
        # This is a simplified implementation
        
        current_chunk_start = 0
        current_chunk_type = None
        current_chunk_name = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check for the start of a new logical unit
            if language == "python":
                if line.startswith("def ") or line.startswith("class "):
                    # If we were tracking a previous chunk, save it
                    if current_chunk_type:
                        self._save_chunk(
                            lines, 
                            current_chunk_start, 
                            i - 1, 
                            file_path, 
                            current_chunk_type, 
                            current_chunk_name,
                            language,
                            metadata
                        )
                    
                    # Start tracking a new chunk
                    current_chunk_start = i
                    current_chunk_type = "function" if line.startswith("def ") else "class"
                    current_chunk_name = line.split(" ")[1].split("(")[0]
            
            elif language in ["javascript", "typescript"]:
                if "function " in line or "class " in line:
                    # If we were tracking a previous chunk, save it
                    if current_chunk_type:
                        self._save_chunk(
                            lines, 
                            current_chunk_start, 
                            i - 1, 
                            file_path, 
                            current_chunk_type, 
                            current_chunk_name,
                            language,
                            metadata
                        )
                    
                    # Start tracking a new chunk
                    current_chunk_start = i
                    current_chunk_type = "function" if "function " in line else "class"
                    
                    if "function " in line:
                        parts = line.split("function ")[1].split('(')[0].strip()
                        current_chunk_name = parts if parts else "anonymous"
                    else:
                        parts = line.split("class ")[1].split(' ')[0].split('{')[0].strip()
                        current_chunk_name = parts
        
        # Save the last chunk if we were tracking one
        if current_chunk_type:
            self._save_chunk(
                lines, 
                current_chunk_start, 
                len(lines) - 1, 
                file_path, 
                current_chunk_type, 
                current_chunk_name,
                language,
                metadata
            )

    def _save_chunk(self, lines: List[str], start: int, end: int, file_path: str, 
                   chunk_type: str, chunk_name: str, language: str, metadata: Dict[str, Any]) -> None:
        """
        Save a code chunk.
        
        Args:
            lines: List of file lines
            start: Start line index
            end: End line index
            file_path: Path to the file
            chunk_type: Type of chunk (function, class, etc.)
            chunk_name: Name of the chunk
            language: Programming language
            metadata: File metadata
        """
        # Make sure we have a reasonable chunk
        if end - start < 1:
            return
        
        chunk_code = '\n'.join(lines[start:end+1])
        
        self.code_chunks.append({
            "id": f"chunk_{len(self.code_chunks)}",
            "file_path": file_path,
            "start_line": start,
            "end_line": end,
            "code": chunk_code,
            "context": f"{chunk_type} {chunk_name} in {file_path}",
            "language": language,
            "chunk_type": chunk_type,
            "chunk_name": chunk_name,
            "metadata": metadata
        })

    def get_indexed_files(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all indexed files.
        
        Returns:
            Dictionary of indexed files
        """
        return self.indexed_files
    
    def get_code_chunks(self) -> List[Dict[str, Any]]:
        """
        Get all code chunks.
        
        Returns:
            List of code chunks
        """
        return self.code_chunks
