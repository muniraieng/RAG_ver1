import os
from pathlib import Path
from dotenv import load_dotenv
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(
        self,
        data_dir: str | Path,
        persist_dir: str | Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "openai/gpt-oss-20b",
    ):
        self.data_dir = Path(data_dir).resolve()
        self.vectorstore = FaissVectorStore(str(persist_dir), embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(self.vectorstore.persist_dir, "faiss.index")
        meta_path = os.path.join(self.vectorstore.persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            docs = load_all_documents(str(self.data_dir))
            if not docs:
                raise ValueError(f"No supported documents found in {self.data_dir}")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("Set GROQ_API_KEY in the project .env file.")
        self.llm = ChatGroq(api_key=groq_api_key, model=llm_model, temperature=0.1)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke(prompt)
        return response.content

# Example usage
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    rag_search = RAGSearch(project_root / "data", project_root / "data" / "faiss_store")
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
