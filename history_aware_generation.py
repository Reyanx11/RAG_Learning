from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

persistent_directory = 'db/chroma_db'
embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

model = ChatGoogleGenerativeAI(model = 'models/gemini-2.5-flash')

chat_history = []

def ask_question(user_question):
    print(f"\nYour question {user_question}")

    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for {search_question}")

    else:
        search_question = user_question


    retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {
        "k":3,
        "score_threshold": 0.4
    }
)

    relevant_docs = retriever.invoke(search_question)

    # Checking if the query is relevant to the docs

    print("found answer...")

    for i,doc in enumerate(relevant_docs,1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f'Doc {i}: {preview}')

    combined_input = f"""Based on following documents answer this question 

    documents : {'\n'.join([f"-{doc.page_content}" for doc in relevant_docs])}

    Please provide a clear answer using only the informatin from the documents. If you can't find answer in the document
    say "i dont have enought information to answer the question based on the documents."
    """

    messages = [
        SystemMessage(content="Your are a helpful assistant, answer question based from these documents and conversation."),
    ] + chat_history +[
        HumanMessage(content= combined_input)
    ]

    result = model.invoke(messages)
    answer = result.content

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content = answer))

    print(f"Answer: \n{answer}")
    return answer

def start_chat():
    print("Ask me question type 'quit' to exit")

    while True:
        question = input("\nYour question: ")

        if question.lower() == 'quit':
            print('Bye')
            break

        ask_question(question)

if __name__ == '__main__':
    start_chat()