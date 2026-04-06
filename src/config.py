"""
config.py — Central configuration loader.

Reads environment variables from a .env file using python-dotenv so that API
keys and paths are never hard-coded in source code.  Every other module imports
from here so there is a single place to update settings.
"""

import os
from dotenv import load_dotenv

# Load variables from .env into the process environment.  load_dotenv() is a
# no-op if .env doesn't exist, so production deployments can rely on real env
# vars without any code change.
load_dotenv()

# Anthropic API key — required by generation.py.
# Add ANTHROPIC_API_KEY=sk-ant-... to your .env file.
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")

# Directory where source documents (PDFs, DOCX) live.  Defaults to ./data so
# the repo layout works out of the box.
DATA_DIR: str = os.getenv("DATA_DIR", "./data")

# Where ChromaDB will persist its vector store on disk.  An in-memory store is
# used when this path is set to ":memory:" (useful for tests).
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# Sentence-Transformers model used for embedding chunks.  The default is small
# and fast; swap for a larger model to improve retrieval quality.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Claude model used for answer generation.
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")

# Maximum number of retrieved chunks passed to the generation step.
TOP_K: int = int(os.getenv("TOP_K", "5"))
