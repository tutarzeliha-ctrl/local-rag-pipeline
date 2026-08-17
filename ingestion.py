import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DataIngestionEngine:
    """
    Data ingestion module responsible for loading, processing, 
    and splitting raw documents into semantic chunks for the RAG pipeline.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Priority separators to maintain semantic context
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_document(self, file_path: str) -> List[Document]:
        """
        Loads raw files based on file extension and converts them to LangChain Document objects.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at path: {file_path}")
            
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError("Unsupported file format! Only .pdf and .txt are allowed.")
            
        return loader.load()

    def process_and_chunk(self, file_path: str) -> List[Document]:
        """
        Loads the document and splits it into semantic chunks.
        """
        raw_documents = self.load_document(file_path)
        chunked_documents = self.text_splitter.split_documents(raw_documents)
        
        print(f"[LOG] Loaded raw pages/documents count: {len(raw_documents)}")
        print(f"[LOG] Total generated semantic chunks: {len(chunked_documents)}")
        
        return chunked_documents


if __name__ == "__main__":
    # Test execution / Verification
    engine = DataIngestionEngine(chunk_size=100, chunk_overlap=20)
    
    # Create sample file for testing
    sample_file = "sample.txt"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(
            "Data Engineering is the foundation of modern AI applications.\n"
            "Retrieval-Augmented Generation (RAG) combines search systems with LLMs.\n"
            "This project processes text documents and prepares them for semantic search."
        )
    
    # Process document and generate chunks
    chunks = engine.process_and_chunk(sample_file)
    print("\n--- TEST SUCCESSFUL ---")
    print(f"First Chunk Content: {chunks[0].page_content}")