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
from itertools import islice
from tqdm import tqdm 

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
            documents = []
            document_ids = []
            document_metadatas = []
            for i, doc in enumerate(content):
                if not isinstance(doc, str):
                    doc = str(doc)  # Convert to string if it's not already
                documents.append(doc)
                document_ids.append(f"{file_path}_{i}")
                document_metadatas.append({"source": file_path})
            return {
                "documents": documents,
                "document_ids": document_ids,
                "document_metadatas": document_metadatas
            }
        else:
            if not isinstance(content, str):
                content = str(content)  # Convert to string if it's not already
            return {
                "documents": [content],
                "document_ids": [file_path],
                "document_metadatas": [{"source": file_path}]
            }
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return {"documents": [], "document_ids": [], "document_metadatas": []}

def index_documents(input_directory: str, index_name: str, model_name: str = config.MODEL_NAME) -> str:
    logger.info(f"Starting indexing process for directory: {input_directory}")
    
    all_files = []
    for root, _, files in os.walk(input_directory):
        for file in files:
            all_files.append(os.path.join(root, file))

    documents = []
    document_ids = []
    document_metadatas = []

    with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(process_file, all_files), total=len(all_files), desc="Processing files"))

    for result in results:
        documents.extend(result["documents"])
        document_ids.extend(result["document_ids"])
        document_metadatas.extend(result["document_metadatas"])

    logger.info(f"Processed {len(documents)} documents")

    RAG = RAGPretrainedModel.from_pretrained(model_name)
    
    index_path = os.path.join(".ragatouille", "colbert", "indexes", index_name)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    # Add error handling for indexing
    try:
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
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        raise

    logger.info(f"Indexing completed. Index created at: {index_path}")
    return index_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Index documents for ColRAG")
    parser.add_argument("input_directory", help="Path to the directory containing documents to index")
    parser.add_argument("index_name", help="Name for the created index")
    args = parser.parse_args()

    index_path = index_documents(args.input_directory, args.index_name)
    print(f"Index created at: {index_path}")