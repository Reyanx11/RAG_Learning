# RAG System (LangChain + Gemini + Chroma)

## Overview

This project implements a Retrieval-Augmented Generation (RAG) workflow with document ingestion, retrieval, and history-aware response generation.

* `ingestion_pipeline.py` – processes documents and creates the vector database
* `retrieval_pipeline.py` – retrieves relevant information based on a user query
* `history_aware_generation.py` – handles conversational queries with context awareness

---

## ingestion_pipeline.py

This script is responsible for preparing the data.

* Loads `.txt` files from the `docs/` directory
* Splits the content into smaller chunks
* Converts each chunk into embeddings using Gemini
* Stores embeddings in a Chroma vector database

This step is run once unless new data is added.

---

## retrieval_pipeline.py

This script is used at runtime.

* Takes a user query
* Converts the query into an embedding
* Searches the vector database for similar chunks
* Sends query + retrieved context to Gemini
* Returns the final answer to user

---

## history_aware_generation.py

This script extends retrieval with conversation awareness.

* Accepts user queries in a loop
* Uses chat history to rewrite follow-up questions into standalone queries
* Retrieves relevant chunks from the database
* Sends query + context to Gemini for answer generation
* Maintains chat history for better responses

---

## How it works

Documents → Chunks → Embeddings → Vector DB
Query → Retrieve relevant chunks → Gemini → Final answer

For conversational queries:
Chat history → Rewrite query → Retrieve → Gemini → Answer

---

## How to run

1. Run ingestion:

```
python ingestion_pipeline.py
```

2. Run retrieval:

```
python retrieval_pipeline.py
```

3. Run history-aware chat:

```
python history_aware_generation.py
```

---

## Notes

* The `db/` folder contains the vector database and should not be deleted after ingestion
* Re-running ingestion repeatedly will consume API requests
* Retrieval does not reprocess documents, it only queries existing data
* Chat history is session-based and not stored permanently
