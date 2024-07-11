import pytest
from colrag.retriever import load_model_from_index, retrieve_and_rerank_documents
from ragatouille import RAGPretrainedModel

def test_load_model_from_index():
    # This is a basic test and might need to be adjusted based on your actual implementation
    with pytest.raises(Exception):  # Expect an exception if the index doesn't exist
        load_model_from_index("non_existent_index")

@pytest.mark.parametrize("query, k, rerank_k", [
    ("test query", 5, 10),
    ("another query", 10, 20),
])
def test_retrieve_and_rerank_documents(query, k, rerank_k):
    # Mock the RAGPretrainedModel for testing
    class MockRAGModel:
        def search(self, query, k):
            return [{"content": f"Result {i}", "score": 1.0 - i*0.1} for i in range(k)]
        
        def rerank(self, query, documents):
            return documents[::-1]  # Reverse the order to simulate reranking
    
    mock_model = MockRAGModel()
    
    results = retrieve_and_rerank_documents(mock_model, query, k, rerank_k)
    
    assert len(results) == k
    assert results[0]["content"] == f"Result {rerank_k-1}"  # The last result should now be first due to reranking
    assert results[-1]["content"] == f"Result {rerank_k-k}"  # The k-th from last result should now be last