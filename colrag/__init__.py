__version__ = "0.1.7"  # Update this to your current version

from .indexer import index_documents, read_file, process_file
from .retriever import load_model_from_index, retrieve_and_rerank_documents, retrieve_and_rerank_multiple_documents
from .config import config
from .logger import get_logger

__all__ = [
    'index_documents',
    'read_file',
    'process_file',
    'load_model_from_index',
    'retrieve_and_rerank_documents',
    'retrieve_and_rerank_multiple_documents',
    'config',
    'get_logger'
]