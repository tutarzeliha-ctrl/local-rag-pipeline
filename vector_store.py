import os
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from embedding import EmbeddingEngine


class VectorStoreEngine:
    """
    Vector Store module responsible for storing document embeddings in ChromaDB
    and performing similarity searches for the RAG pipeline.
    """
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_engine = EmbeddingEngine()
        self.embeddings = self.embedding_engine.get_embeddings()
        self.vector_db = None

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """
        Creates and persists a ChromaDB vector store from a list of chunked documents.
        """
        print(f"[LOG] Creating vector database in directory: {self.persist_directory}...")
        self.vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("[LOG] Vector database created and persisted successfully!")
        return self.vector_db

    def load_vector_store(self) -> Chroma:
        """
        Loads an existing ChromaDB vector database from local disk.
        """
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(f"No existing vector database found at {self.persist_directory}")
            
        print(f"[LOG] Loading existing vector database from: {self.persist_directory}...")
        self.vector_db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        return self.vector_db

    def similarity_search(self, query: str, k: int = 2) -> List[Document]:
        """
        Performs semantic similarity search for a given query and returns top-k matching documents.
        """
        if self.vector_db is None:
            self.load_vector_store()
            
        print(f"[LOG] Executing similarity search for query: '{query}'")
        results = self.vector_db.similarity_search(query, k=k)
        return results


if __name__ == "__main__":
    # Test execution / Verification
    from ingestion import DataIngestionEngine

    # 1. Prepare sample text & chunking
    sample_file = "sample.txt"
    ingestion_engine = DataIngestionEngine(chunk_size=100, chunk_overlap=20)
    chunks = ingestion_engine.process_and_chunk(sample_file)

    # 2. Store in ChromaDB
    vs_engine = VectorStoreEngine(persist_directory="./test_chroma_db")
    vs_engine.create_vector_store(chunks)

    # 3. Perform similarity search test
    query = "What is RAG?"
    search_results = vs_engine.similarity_search(query, k=1)

    print("\n--- TEST SUCCESSFUL ---")
    print(f"Search Query: '{query}'")
    print(f"Top Matching Chunk: {search_results[0].page_content}")