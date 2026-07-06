from langchain_core.documents import Document
import re
import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_all_text_files(text_directory, extensions=("txt",)):
    """Load all text files recursively and wrap them as LangChain Documents."""
    all_documents = []
    text_dir = Path(text_directory)
    text_files = [p for p in text_dir.rglob("**/*") if p.suffix.lower().lstrip('.') in extensions]

    print(f"Found {len(text_files)} text files to process")

    for text_file in text_files:
        print(f"\nProcessing: {text_file.name}")
        try:
            raw_text = text_file.read_text(encoding="utf-8")
            metadata = {
                "source_file": text_file.name,
                "file_type": "text",
                "source_path": str(text_file)
            }
            all_documents.append(Document(page_content=raw_text, metadata=metadata))
            print(f"  ✓ Loaded {len(raw_text.splitlines())} lines")
        except Exception as e:
            print(f"  ✗ Error reading {text_file.name}: {e}")

    print(f"\nTotal text documents loaded: {len(all_documents)}")
    return all_documents

def split_text_into_chunks(documents, chunk_size=1200, chunk_overlap=120):
    """Split documents into chunks of specified size with overlap."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            all_chunks.append(Document(page_content=chunk, metadata=doc.metadata))
    print(f"\nTotal text chunks created: {len(all_chunks)}")
    return all_chunks



