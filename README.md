# Local RAG Document Q&A

A completely local Retrieval-Augmented Generation (RAG) pipeline that allows you to ask questions based on the contents of any PDF. Runs entirely on consumer hardware without relying on external APIs.

## Tech Stack
* **LLM:** Qwen2.5 (via Ollama)
* **Embeddings:** `all-MiniLM-L6-v2` (HuggingFace)
* **Vector Database:** ChromaDB
* **Orchestration:** Python, `pypdf`

## How to Run
1. Install Ollama and pull the model: `ollama run qwen2.5`
2. Install dependencies: `pip install pypdf requests chromadb sentence-transformers`
3. Place your target document in the folder as `test.pdf`.
4. Run the script: `python rag_demo.py`
