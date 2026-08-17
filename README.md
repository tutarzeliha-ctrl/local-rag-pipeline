# Local RAG Data Pipeline

A modular, privacy-focused, and completely local **Retrieval-Augmented Generation (RAG)** pipeline built with **Python**, **LangChain**, **HuggingFace Embeddings**, and **ChromaDB**.

## 📌 Architecture & Features

- **Data Ingestion Engine (`ingestion.py`):** Efficiently loads raw PDF/TXT files and splits them into semantic chunks using `RecursiveCharacterTextSplitter`.
- **Embedding Engine (`embedding.py`):** Converts text chunks into 384-dimensional dense vectors locally using `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector Store Engine (`vector_store.py`):** Manages vector persistence and fast similarity searches using `ChromaDB`.
- **RAG Pipeline Orchestrator (`rag_pipeline.py`):** Integrates all components to construct optimized, context-augmented prompts for LLMs.

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone <YOUR-GITHUB-REPO-LINK>
   cd local-rag-data-pipeline