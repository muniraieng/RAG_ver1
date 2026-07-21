# RAG Pipelines Learning Demo (Python)

## Contribute by
A learning project for building an RAG Pipelines by munir.ai.eng@gmail.com

This project is a hands-on introduction to **Retrieval-Augmented Generation (RAG)**. RAG lets a language model answer questions using your own documents instead of relying only on its built-in knowledge.

The repository demonstrates the same core idea in two different workflows:

1. **Jupyter notebooks (`.ipynb`)** — learn each RAG step interactively.
2. **Python application (`.py`)** — run a reusable, modular RAG pipeline.

## What happens in a RAG pipeline?

```text
Documents -> load -> split into chunks -> create embeddings -> vector store
                                                               |
User question -> create embedding -> similarity search -> relevant chunks
                                                               |
                                                     LLM answer with context
```

The local embedding model used here is `all-MiniLM-L6-v2`. The answer-generation model is accessed through Groq (`openai/gpt-oss-20b`).

## 1. Notebook workflow: learn one step at a time

The notebooks are designed for exploration. Run their cells in order, inspect the documents and chunks, change a question, and see how retrieval changes the answer.

| Notebook | What it demonstrates |
| --- | --- |
| `notebook/document.ipynb` | LangChain's `Document` structure: content plus metadata. |
| `notebook/csv_loader.ipynb` | Loading CSV rows, creating an in-memory FAISS store, retrieving chunks, and generating an answer. |
| `notebook/directory_loader.ipynb` | Loading all text files from `data/text_files` and answering from them. |
| `notebook/pdf_loader.ipynb` | Loading PDF content. |
| `notebook/webbase_loader.ipynb` | Loading content from a web page. |
| `typesense.ipynb` | A full RAG example using Typesense as the vector database instead of local FAISS. |

Most notebooks use **FAISS in memory**: the vectors exist only while the notebook kernel is running. `typesense.ipynb` stores vectors in a Typesense collection, which is useful for learning how a hosted/persistent vector database fits into RAG.

To open the notebooks:

```powershell
.\.venv\Scripts\Activate.ps1
jupyter notebook
```

Open a notebook and run cells from top to bottom. Start with `notebook/directory_loader.ipynb` or `notebook/csv_loader.ipynb` for the clearest end-to-end example.

## 2. Python workflow: reusable RAG application

The Python pipeline puts the RAG steps into separate files so they can be reused in an application.

| File | Responsibility |
| --- | --- |
| `src/data_loader.py` | Recursively loads supported files from `data/` (PDF, TXT, CSV, Excel, Word, and JSON). |
| `src/embedding.py` | Splits documents into overlapping chunks and creates embeddings. |
| `src/vectorstore.py` | Builds, saves, loads, and searches a local FAISS index. |
| `src/search.py` | Retrieves the most similar chunks and sends them to the Groq LLM in a grounded prompt. |
| `app.py` | Entry point that runs a sample question. |

Unlike the basic notebook examples, this workflow saves its vector index in `faiss_store/`. On later runs it reuses `faiss_store/faiss.index` and `faiss_store/metadata.pkl` instead of embedding all documents again.

Run it from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Change these lines in `app.py` to try your own question or retrieve a different number of chunks:

```python
query = "AI Engineer"
summary = rag_search.search_and_summarize(query, top_k=3)
```

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root. The Python pipeline and most answer-generation notebook cells require a Groq key:

```env
GROQ_API_KEY=your_groq_api_key
```

The Typesense notebook also requires these settings:

```env
TYPESENSE_HOST=your-typesense-host
TYPESENSE_API_KEY=your-typesense-api-key
TYPESENSE_PORT=443
TYPESENSE_PROTOCOL=https
TYPESENSE_COLLECTION=rag_text_chunks
```

Keep `.env` private; do not commit API keys to Git.

## Add your own knowledge

Put files in the matching folders under `data/`:

```text
data/
  text_files/     # .txt files
  pdf/            # .pdf files
  csv/            # .csv files
```

Then run `python app.py`. If the FAISS index already exists, delete or move only the generated `faiss_store/` directory before rerunning so the application rebuilds the index with your new content.

## Key learning difference

- Use the **notebooks** when you want to understand and experiment with individual stages such as loading, chunking, embeddings, and retrieval.
- Use the **Python pipeline** when you want a repeatable structure that can grow into a CLI, web app, API, or larger project.

Both approaches follow the same RAG principle: retrieve relevant information first, then ask the LLM to answer using that information.

