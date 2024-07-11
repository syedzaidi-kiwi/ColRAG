from colrag.retriever import load_model_from_index, retrieve_and_rerank_documents
from colrag.config import config
import json

def test_reranking():
    # Load the model
    index_name = "emerald_index"  # Make sure this matches your index name
    model = load_model_from_index(index_name)

    # Test query
    query = "Trends in fish farming in the UK"

    # Perform initial search without reranking
    initial_results = model.search(query, k=100)

    # Perform search with reranking
    reranked_results = retrieve_and_rerank_documents(model, query, k=10, rerank_k=100)

    # Compare results
    print("Initial top 10 results:")
    for i, result in enumerate(initial_results[:10], 1):
        print(f"{i}. Score: {result['score']:.4f}, Content: {result['content'][:100]}...")

    print("\nReranked top 10 results:")
    for i, result in enumerate(reranked_results, 1):
        print(f"{i}. Score: {result['score']:.4f}, Content: {result['content'][:100]}...")

    # Check if the order has changed
    initial_order = [result['content'][:100] for result in initial_results[:10]]
    reranked_order = [result['content'][:100] for result in reranked_results]

    if initial_order != reranked_order:
        print("\nReranking has changed the order of results.")
    else:
        print("\nReranking did not change the order of results.")

    # Save results to files for detailed comparison
    with open('initial_results.json', 'w') as f:
        json.dump(initial_results[:10], f, indent=2)

    with open('reranked_results.json', 'w') as f:
        json.dump(reranked_results, f, indent=2)

    print("\nDetailed results saved to 'initial_results.json' and 'reranked_results.json'")

if __name__ == "__main__":
    test_reranking()