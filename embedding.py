from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings


class EmbeddingEngine:
    """
    Embedding module responsible for converting text chunks into dense vector representations
    using local HuggingFace embedding models.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the embedding model.
        Default model: all-MiniLM-L6-v2 (fast, lightweight, 384-dimensional vectors).
        """
        print(f"[LOG] Loading embedding model: {model_name}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        print("[LOG] Embedding model loaded successfully!")

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Returns the initialized LangChain embedding object to be used by vector stores.
        """
        return self.embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generates vector representation for a single text query.
        """
        return self.embeddings.embed_query(query)


if __name__ == "__main__":
    # Test execution / Verification
    engine = EmbeddingEngine()
    
    sample_text = "Data Engineering for AI applications."
    vector = engine.embed_query(sample_text)
    
    print("\n--- TEST SUCCESSFUL ---")
    print(f"Sample Text: '{sample_text}'")
    print(f"Vector Length (Dimensions): {len(vector)}")
    print(f"First 5 Vector Values: {vector[:5]}")