# ColRAG

ColRAG is a RAG (Retrieval-Augmented Generation) pipeline using ColBERT via RAGatouille. It provides easy-to-use functions for indexing documents from various file formats and retrieving relevant information using the ColBERT model.

## Installation

To install ColRAG, you'll need to use Poetry. If you don't have Poetry installed, you can install it by following the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).

Once you have Poetry installed, follow these steps:

1. Clone the ColRAG repository:
   ```bash
   git clone https://github.com/your-username/colrag.git
   cd colrag
   ```

2. Install the dependencies using Poetry:
   ```bash
   poetry install
   ```

This will install all the necessary dependencies, including the latest version of RAGatouille from GitHub.

## Usage

### Indexing Documents

```python
from colrag import index_documents

input_directory = "path/to/your/documents"
index_name = "my_colrag_index"
index_path = index_documents(input_directory, index_name)
print(f"Index created at: {index_path}")
```

### Retrieving Documents

```python
from colrag import load_model_from_index, retrieve_documents, retrieve_multiple_documents

# Load the model from the index
index_path = "path/to/your/index"
model = load_model_from_index(index_path)

# Single query
query = "What is the main topic of this document?"
results = retrieve_documents(model, query)
for result in results:
    print(f"Rank: {result['rank']}, Score: {result['score']}, Content: {result['content'][:100]}...")

# Multiple queries
queries = [
    "What is the main topic of this document?",
    "Who are the key people mentioned?",
    "What are the main conclusions?"
]
multi_results = retrieve_multiple_documents(model, queries)
for i, query_results in enumerate(multi_results):
    print(f"Query {i + 1}:")
    for result in query_results[:3]:  # Print top 3 results for each query
        print(f"Rank: {result['rank']}, Score: {result['score']}, Content: {result['content'][:100]}...")
```

## License

This project is licensed under the MIT License.