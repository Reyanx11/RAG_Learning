from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,SystemMessage



load_dotenv()

persistent_directory = 'db/chroma_db' 

embedding_model = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")

db = Chroma(
    persist_directory= persistent_directory,
    embedding_function= embedding_model,
    collection_metadata= {"hnsw:space":"cosine"}
)

query = "Who founded Tesla"
#query = "who is Reyan"
#query = "when did elon musk joined Tesla"

#retriever = db.as_retriever(search_kwargs={"k":3})

retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {
        "k":3,
        "score_threshold": 0.4
    }
)

relevant_docs = retriever.invoke(query)

# Checking if the query is relevant to the docs

print(f"User query: {query}")
print("--Context--")

for i,doc in enumerate(relevant_docs,1):
    print(f"Document {i}: \n{doc.page_content}\n")

combined_input = f"""Based on following documents answer this question {query}

documents : {'\n'.join([f"-{doc.page_content}" for doc in relevant_docs])}

Please provide a clear answer using only the informatin from the documents. If you can't find answer in the document
say "i dont have enought information to answer the question based on the documents."
"""

model = ChatGoogleGenerativeAI(model = 'models/gemini-2.5-flash')

messages = [
    SystemMessage(content="Your are a helpful assistant"),
    HumanMessage(content= combined_input),
]

result = model.invoke(messages)

print("\n-------Generated Response----------------")
print(result.content)