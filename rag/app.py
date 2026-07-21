from pathlib import Path
from pydoc import doc
import sys
from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.search import RAGSearch
from src.vectorstore import FaissVectorStore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

"""
if __name__ == "__main__":
    docs = load_all_documents("data")
    print("docs***** -->\n",docs)
    chunks = EmbeddingPipeline().chunk_documents(docs)
    print("chunks***** -->\n",chunks)
    chunkVectors = EmbeddingPipeline().embed_chunks(chunks)
    print("chunkVectors***** -->\n",chunkVectors)
"""

"""
if __name__ == "__main__":
    docs = load_all_documents("data")
    print("docs***** -->\n",docs)
    store = FaissVectorStore()
    store.build_from_documents(docs)
"""
"""
if __name__ == "__main__":
    store = FaissVectorStore()
    store.load()
    print(store.query("what munir want?", top_k=3))
"""


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    rag_search = RAGSearch(
        data_dir=project_root / "data",
        persist_dir=project_root / "faiss_store",
    )
    query = "AI Engineer"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
