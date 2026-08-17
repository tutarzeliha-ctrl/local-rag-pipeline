from typing import List

from langchain_core.documents import Document

from ingestion import DataIngestionEngine
from vector_store import VectorStoreEngine


class LocalRAGPipeline:
    """
    Complete RAG Pipeline integration module that combines Data Ingestion, 
    Vector Storage, and Context Retrieval for QA tasks.
    """
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.ingestion_engine = DataIngestionEngine(chunk_size=500, chunk_overlap=100)
        self.vector_store_engine = VectorStoreEngine(persist_directory=persist_directory)

    def ingest_and_index(self, file_path: str):
        """
        Ingests a document and indexes its embeddings into ChromaDB.
        """
        print(f"\n[PIPELINE] Starting ingestion process for: {file_path}")
        chunks = self.ingestion_engine.process_and_chunk(file_path)
        self.vector_store_engine.create_vector_store(chunks)
        print("[PIPELINE] Ingestion and indexing completed successfully!\n")

    def retrieve_context(self, query: str, top_k: int = 2) -> List[Document]:
        """
        Retrieves relevant context chunks from the vector store for a user query.
        """
        print(f"[PIPELINE] Retrieving top-{top_k} contexts for query: '{query}'")
        results = self.vector_store_engine.similarity_search(query=query, k=top_k)
        return results

    def answer_question(self, query: str) -> str:
        """
        Simulates the final RAG generation step by augmenting prompt with retrieved context.
        """
        relevant_docs = self.retrieve_context(query)
        context_str = "\n---\n".join([doc.page_content for doc in relevant_docs])

        # Prompt Template structure for LLM
        formatted_prompt = (
            f"Context Information:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            f"Answer based strictly on the context provided above:"
        )

        return formatted_prompt


if __name__ == "__main__":
    # Test full end-to-end local RAG execution
    pipeline = LocalRAGPipeline(persist_directory="./pipeline_chroma_db")

    # Step 1: Index document
    pipeline.ingest_and_index("sample.txt")

    # Step 2: Query pipeline
    user_query = "What does RAG combine?"
    prompt_for_llm = pipeline.answer_question(user_query)

    print("\n--- GENERATED RAG PROMPT FOR LLM ---")
    print(prompt_for_llm)