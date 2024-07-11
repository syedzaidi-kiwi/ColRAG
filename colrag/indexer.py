import os
from typing import List, Dict, Any
import pandas as pd
from docx import Document
from bs4 import BeautifulSoup
import json
import PyPDF2
from ragatouille import RAGPretrainedModel
from colrag.config import config
from colrag.logger import get_logger
from concurrent.futures import ProcessPoolExecutor
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console
import warnings

warnings.filterwarnings("ignore")

logger = get_logger(__name__)
console = Console()

def read_pdf(file_path: str) -> str:
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def read_csv(file_path: str) -> List[str]:
    df = pd.read_csv(file_path)
    return df.to_dict('records')

def read_excel(file_path: str) -> List[str]:
    df = pd.read_excel(file_path)
    return df.to_dict('records')

def read_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def read_html(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        return soup.get_text()

def read_json(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as file:
        return json.load(file)

def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def read_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_file(file_path: str) -> Any:
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    if extension == '.pdf':
        return read_pdf(file_path)
    elif extension == '.csv':
        return read_csv(file_path)
    elif extension in ['.xlsx', '.xls']:
        return read_excel(file_path)
    elif extension == '.docx':
        return read_docx(file_path)
    elif extension in ['.html', '.htm', '.xhtml']:
        return read_html(file_path)
    elif extension == '.json':
        return read_json(file_path)
    elif extension == '.jsonl':
        return read_jsonl(file_path)
    elif extension == '.txt':
        return read_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {extension}")

def process_file(file_path: str) -> Dict[str, Any]:
    try:
        content = read_file(file_path)
        if isinstance(content, list):
            return {
                "documents": content,
                "document_ids": [file_path] * len(content),
                "document_metadatas": [{"source": file_path}] * len(content)
            }
        else:
            return {
                "documents": [content],
                "document_ids": [file_path],
                "document_metadatas": [{"source": file_path}]
            }
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return {"documents": [], "document_ids": [], "document_metadatas": []}

def index_documents(input_directory: str, index_name: str, model_name: str = config.MODEL_NAME) -> str:
    console.print(f"[bold green]Starting indexing process for directory:[/bold green] {input_directory}")
    
    all_files = []
    for root, _, files in os.walk(input_directory):
        for file in files:
            all_files.append(os.path.join(root, file))

    documents = []
    document_ids = []
    document_metadatas = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Processing files...", total=len(all_files))
        
        with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            for result in executor.map(process_file, all_files):
                documents.extend(result["documents"])
                document_ids.extend(result["document_ids"])
                document_metadatas.extend(result["document_metadatas"])
                progress.update(task, advance=1)

    console.print(f"[bold green]Processed[/bold green] {len(documents)} documents")

    RAG = RAGPretrainedModel.from_pretrained(model_name)
    
    index_path = os.path.join(".ragatouille", "colbert", "indexes", index_name)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    console.print("[bold cyan]Indexing documents...[/bold cyan]")
    RAG.index(
        collection=documents,
        document_ids=document_ids,
        document_metadatas=document_metadatas,
        index_name=index_name,
        overwrite_index=True,
        max_document_length=256,
        split_documents=True,
        bsize=config.BATCH_SIZE,
        use_faiss=False
    )

    console.print(f"[bold green]Indexing completed. Index created at:[/bold green] {index_path}")
    return index_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Index documents for ColRAG")
    parser.add_argument("input_directory", help="Path to the directory containing documents to index")
    parser.add_argument("index_name", help="Name for the created index")
    args = parser.parse_args()

    index_path = index_documents(args.input_directory, args.index_name)
    console.print(f"[bold green]Index created at:[/bold green] {index_path}")

# Explicitly export the functions
__all__ = ['index_documents', 'read_file', 'process_file']