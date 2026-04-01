from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*'],
)

embedding_model  = GoogleGenerativeAIEmbeddings(
    model = "models/gemini-embedding-001",
    task_type= 'retrieval_query'
)

db= Chroma(
    persist_directory='db/chroma_db',
    embedding_function= embedding_model
)

model = ChatGoogleGenerativeAI(model = 'models/gemini-2.5-flash')

class ChatRequest(BaseModel):
    message:str

def get_ans(query:str):
    retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {
        "k":3,
        "score_threshold": 0.4
    }
)
    
    relevant_docs = retriever.invoke(query)

    context = '\n'.join([doc.page_content for doc in relevant_docs])

    if not relevant_docs:
        return "I dont't have enought information to answer this question"

    prompt = f"""Answer the question based on following context 
    question: {query}

    context : {context}

    Please provide a clear answer using only the information from the documents. If you can't find answer in the document
    say "i dont have enough information to answer the question based on the documents."
    """

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=prompt)
    ]

    response = model.invoke(messages)

    return response.content

@app.get("/")
def root():
    return {"status": "RAG API running"}

@app.post("/chat")
def chat(req:ChatRequest):
    answer = get_ans(req.message)
    return {'response':answer}