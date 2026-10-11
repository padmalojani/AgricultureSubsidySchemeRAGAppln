import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# LangSmith Tracing Configuration
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "tn-agriculture-subsidy-rag")

# Automatically configure environment variables for LangChain tracing
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
FAISS_INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")
METADATA_DB_PATH = os.path.join(DATA_DIR, "metadata.db")

# Chunking defaults
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_RETRIEVAL_K = 5

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)