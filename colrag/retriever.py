from ragatouille import RAGPretrainedModel
from typing import List, Dict, Any
from colrag.logger import get_logger
import os

logger = get_logger(__name__)

def load_model_from_index(index_path: str) -> RAGPretrainedModel:
    logger.info(f"Loading model from index: {index_path}")
    return RAGPretrainedModel.from_index(index_path)

def retrieve_and_rerank_documents(index_path: str, query: str, k: int = 10, rerank_k: int = 100) -> List[Dict[str, Any]]:
    logger.info(f"Retrieving and reranking documents for query: {query}")
    
    # Load the model from the index
    model = load_model_from_index(index_path)
    
    # First, retrieve a larger set of documents
    initial_results = model.search(query, k=rerank_k)
    
    # Extract document content and create a mapping to original results
    documents = []
    document_to_result = {}
    for result in initial_results:
        content = result['content']
        documents.append(content)
        document_to_result[content] = result
    
    # Then, use ColBERT's rerank method to rerank the results
    reranked_documents = model.rerank(query, documents)
    
    # Reorder the initial results based on the reranking
    reranked_results = []
    for doc in reranked_documents:
        if isinstance(doc, dict) and 'content' in doc:
            content = doc['content']
        else:
            content = doc
        if content in document_to_result:
            reranked_results.append(document_to_result[content])
    
    # Take the top k results after reranking
    final_results = reranked_results[:k]
    
    return final_results

def retrieve_and_rerank_multiple_documents(index_path: str, queries: List[str], k: int = 10, rerank_k: int = 100) -> List[List[Dict[str, Any]]]:
    logger.info(f"Retrieving and reranking documents for {len(queries)} queries")
    
    all_results = []
    for query in queries:
        results = retrieve_and_rerank_documents(index_path, query, k, rerank_k)
        all_results.append(results)
    
    return all_results