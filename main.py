import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

from langchain import hub
from langchain_community.document_loaders import  NotionDBLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.docstore.document import Document
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_community.chat_message_histories import RedisChatMessageHistory

load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
NOTION_KEY = os.environ.get("NOTION_KEY")
NOTION_DB_ID=os.environ.get("NOTION_DB_ID")
REDIS_URL=os.environ.get("REDIS_URL")


llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)

set_llm_cache(InMemoryCache())


# Function to check if ChromaDB is empty
def is_chromadb_empty(vectorstore):
    return bool(vectorstore._collection.get(include=['embeddings']))

# Initialize ChromaDB
vectorstore = Chroma()

if is_chromadb_empty(vectorstore):
    loader = NotionDBLoader(
        integration_token=NOTION_KEY,
        database_id=NOTION_DB_ID,
        request_timeout_sec=30,  # optional, defaults to 10
    )

    loader.load()
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

    # Split the document into chunks
    split_docs = []
    for doc in docs:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            split_docs.append(Document(page_content=chunk, metadata={}))

    vectorstore = Chroma.from_documents(documents=split_docs, embedding=OpenAIEmbeddings())
else:
    print("chroma db is already populated")

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\
You are asistance focus on helping project management and software development. \
You can use bullet points or list and also you can use emoticon if it helps to answer the question. \
Also try to provide further questions if there is any revelent information present. \

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def get_message_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(session_id, url=REDIS_URL)


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str
    session_id: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/question/")
def ask_question(question: Question):
    answer = conversational_rag_chain.invoke(
    {"input": question.question},
    config={
        "configurable": {"session_id": question.session_id}
    },)["answer"]

    return {"answer": answer}

