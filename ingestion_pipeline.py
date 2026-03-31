import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import time

load_dotenv()

# Loading files
def load_documents(docs_path = "docs"):
    print(f'Loading documents from {docs_path}.....')

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f'The directory {docs_path} does not exist. Please create it')
    
    loader = DirectoryLoader(
        path = docs_path,
        glob= "*.txt",
        loader_cls= TextLoader,
        loader_kwargs= {"encoding": "utf-8"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f'No .txt files found in {docs_path}. Please add files.')
    
    for i,doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f" source: {doc.metadata['source']}")
        print(f" Content Length: {len(doc.page_content)} Characters")
        print(f" Content Preview: {doc.page_content[:100]}....")
        print(f" metadata: {doc.metadata}")

    return documents

# Chunking

def split_documents(documents,chunk_size=1000,chunk_overlap=0):
    print('Splitting documents into chunks')

    text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content: ")
            print(chunk.page_content)
            print("-"*50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks)-5} more chunks")

    return chunks


    #Vector store
def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print(f"Creating embeddings for {len(chunks)} chunks...")

    embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",  
    task_type="retrieval_document"
)

    batch_size = 50 
    
    print(f"--- Processing batch 1 (0 to {min(batch_size, len(chunks))}) ---")
    initial_batch = chunks[:batch_size]
    vectorstore = Chroma.from_documents(
        documents=initial_batch,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    if len(chunks) > batch_size:
        for i in range(batch_size, len(chunks), batch_size):
            print(f"Waiting 60 seconds...")
            time.sleep(60) 
            
            batch = chunks[i : i + batch_size]
            print(f"--- Processing batch ({i} to {min(i + batch_size, len(chunks))}) ---")
            vectorstore.add_documents(batch)

    print("--- Finished creating vector store ---")
    return vectorstore

def main():
    #Loading files
    documents = load_documents(docs_path="docs")

    #Chunking
    chunks = split_documents(documents)

    # Vector store
    vectorstore = create_vector_store(chunks)



if __name__ == '__main__':
    main()