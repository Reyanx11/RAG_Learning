# RAG System (LangChain + Gemini + Chroma)

## Overview

This project implements a basic Retrieval-Augmented Generation (RAG) workflow using two scripts:

* `ingestion_pipeline.py` – processes documents and creates the vector database
* `retrieval_pipeline.py` – retrieves relevant information based on a user query

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

## How it works

Documents → Chunks → Embeddings → Vector DB
Query → Retrieve relevant chunks → Gemini → Final answer

## How to run

1. Run ingestion:

```
python ingestion_pipeline.py
```

2. Run retrieval:

```
python retrieval_pipeline.py
```

---

## Notes

* The `db/` folder contains the vector database and should not be deleted after ingestion
* Re-running ingestion repeatedly will consume API requests
* Retrieval does not reprocess documents, it only queries existing data
