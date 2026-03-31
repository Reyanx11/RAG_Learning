from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = 'db/chroma_db' 

embedding_model = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")

db = Chroma(
    persist_directory= persistent_directory,
    embedding_function= embedding_model,
    collection_metadata= {"hnsw:space":"cosine"}
)

# query = "Who founded Tesla"
query = "When Elon musk joined Tesla"

retriever = db.as_retriever(search_kwargs={"k":3})

# retriever = db.as_retriever(
#     search_type = "similarity_score_threshold",
#     search_kwargs = {
#         "k":5,
#         "score_threshold": 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"User query: {query}")
print("--Context--")

for i,doc in enumerate(relevant_docs,1):
    print(f"Document {i}: \n{doc.page_content}\n")