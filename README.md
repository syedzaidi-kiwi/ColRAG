# ColRAG

<div align="center">

[![PyPI version](https://badge.fury.io/py/colrag.svg)](https://badge.fury.io/py/colrag)
[![Python Versions](https://img.shields.io/pypi/pyversions/colrag.svg)](https://pypi.org/project/colrag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/colrag)](https://pepy.tech/project/colrag)
[![GitHub stars](https://img.shields.io/github/stars/yourusername/colrag.svg?style=social&label=Star&maxAge=2592000)](https://github.com/syedzaidi-kiwi/ColRAG.git/)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/colrag.svg?style=social&label=Fork&maxAge=2592000)](https://github.com/syedzaidi-kiwi/ColRAG.git/)
[![GitHub issues](https://img.shields.io/github/issues/yourusername/colrag.svg)](https://github.com/syedzaidi-kiwi/ColRAG/issues/)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/yourusername/colrag.svg)](https://github.com/syedzaidi-kiwi/ColRAG/pulls/)
[![GitHub contributors](https://img.shields.io/github/contributors/yourusername/colrag.svg)](https://GitHub.com/syedzaidi-kiwi/ColRAG/graphs/contributors/)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/yourusername/colrag/Python%20package)](https://github.com/syedzaidi-kiwi/ColRAG/actions)
[![codecov](https://codecov.io/gh/yourusername/colrag/branch/main/graph/badge.svg)](https://codecov.io/gh/syedzaidi-kiwi/ColRAG)
[![Documentation Status](https://readthedocs.org/projects/colrag/badge/?version=latest)](https://colrag.readthedocs.io/en/latest/?badge=latest)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/syedzaidi-kiwi/ColRAG/graphs/commit-activity)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

</div>

ColRAG is a powerful RAG (Retrieval-Augmented Generation) pipeline using ColBERT via RAGatouille. It provides an efficient and effective way to implement retrieval-augmented generation in your projects.

## 🌟 Features

- 📚 Efficient document indexing
- 🚀 Fast and accurate retrieval with reranking as an optional parameter
- 🔗 Seamless integration with ColBERT and RAGatouille
- 📄 Support for multiple file formats (PDF, CSV, XLSX, DOCX, HTML, JSON, JSONL, TXT)
- ⚙️ Customizable retrieval parameters

## 🛠️ Installation

You can install ColRAG using pip:

```bash
pip install colrag
```

You can also install ColRAG using poetry (recommended):

### Using Poetry

If you're using Poetry to manage your project dependencies, you can add ColRAG to your project with:

```bash
poetry add colrag
```

Or if you want to add it to your `pyproject.toml` manually, you can add the following line under `[tool.poetry.dependencies]`:

```toml
colrag = "^0.1.0"  # Replace with the latest version
```

Then run:

```bash
poetry install
```
## 🚀 Quick Start

Here's a simple example to get you started:

```python
from colrag import index_documents, retrieve_and_rerank_documents

# Index your documents
index_path = index_documents("/path/to/your/documents", "my_index")

# Retrieve documents
query = "What is the capital of France?"
results = retrieve_and_rerank_documents(index_path, query)

for result in results:
    print(f"Score: {result['score']}, Content: {result['content'][:100]}...")
```

## 📖 Documentation

For more detailed information about ColRAG's features and usage, please refer to our [documentation](https://colrag.readthedocs.io/).

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## 📄 License

ColRAG is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

## 📚 Citation

If you use ColRAG in your research, please cite it as follows:

```bibtex
@software{colrag,
  author = {Syed Asad},
  title = {ColRAG: A RAG pipeline using ColBERT via RAGatouille},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/syedzaidi-kiwi/ColRAG.git}}
}
```

## 📬 Contact

For any questions or feedback, please open an issue on our [GitHub repository](https://github.com/syedzaidi-kiwi/ColRAG/issues).

## 🙏 Acknowledgements

- [ColBERT](https://github.com/stanford-futuredata/ColBERT) for the underlying retrieval model
- [RAGatouille](https://github.com/bclavie/RAGatouille) for the RAG implementation

---

<div align="center">
    <sub>Built with ❤️ by <a href="https://github.com/syedzaidi-kiwi">your username</a></sub>
</div>