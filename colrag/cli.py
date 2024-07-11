import typer
from rich.console import Console
from rich.table import Table
from colrag.indexer import index_documents
from colrag.retriever import load_model_from_index, retrieve_and_rerank_documents, retrieve_and_rerank_multiple_documents
from colrag.config import config
import os
from typing import Optional

app = typer.Typer()
console = Console()

@app.command()
def index(input_directory: str, index_name: str):
    """Index documents from the specified directory."""
    index_path = index_documents(input_directory, index_name)
    console.print(f"Index created at: [bold green]{index_path}[/bold green]")

@app.command()
def search(index_name: str, query: str, k: int = 10, rerank_k: int = config.BATCH_SIZE, output_file: Optional[str] = None):
    """Search for documents using a single query."""
    model = load_model_from_index(index_name)
    results = retrieve_and_rerank_documents(model, query, k, rerank_k)
    
    table = Table(title=f"Search Results for: {query}")
    table.add_column("Rank", style="cyan", no_wrap=True)
    table.add_column("Score", style="magenta")
    table.add_column("Content", style="green")

    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"Search Results for: {query}\n\n")
            for result in results:
                f.write(f"Rank: {result['rank']}\n")
                f.write(f"Score: {result['score']:.4f}\n")
                f.write(f"Content: {result['content']}\n\n")
                table.add_row(
                    str(result['rank']),
                    f"{result['score']:.4f}",
                    result['content'][:100] + "..."
                )
        console.print(f"Full results written to: [bold green]{output_file}[/bold green]")
    else:
        for result in results:
            table.add_row(
                str(result['rank']),
                f"{result['score']:.4f}",
                result['content'][:100] + "..."
            )

    console.print(table)

@app.command()
def batch_search(index_name: str, query_file: str, k: int = 10, rerank_k: int = config.BATCH_SIZE, output_file: Optional[str] = None):
    """Search for documents using multiple queries from a file."""
    model = load_model_from_index(index_name)
    
    with open(query_file, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]
    
    results = retrieve_and_rerank_multiple_documents(model, queries, k, rerank_k)
    
    if output_file:
        with open(output_file, 'w') as f:
            for i, (query, query_results) in enumerate(zip(queries, results)):
                f.write(f"Query {i+1}: {query}\n\n")
                for result in query_results:
                    f.write(f"Rank: {result['rank']}\n")
                    f.write(f"Score: {result['score']:.4f}\n")
                    f.write(f"Content: {result['content']}\n\n")
                f.write("\n" + "-"*50 + "\n\n")
        console.print(f"Full results written to: [bold green]{output_file}[/bold green]")

    for i, (query, query_results) in enumerate(zip(queries, results)):
        console.print(f"\n[bold]Query {i+1}: {query}[/bold]")
        table = Table(title=f"Search Results")
        table.add_column("Rank", style="cyan", no_wrap=True)
        table.add_column("Score", style="magenta")
        table.add_column("Content", style="green")

        for result in query_results:
            table.add_row(
                str(result['rank']),
                f"{result['score']:.4f}",
                result['content'][:100] + "..."
            )

        console.print(table)

if __name__ == "__main__":
    app()