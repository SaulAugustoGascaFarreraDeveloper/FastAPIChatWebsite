from dotenv import load_dotenv
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain.chains import history_aware_retriever,create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
import uuid
import logging

load_dotenv()


app = FastAPI()

model = ChatOpenAI(model="gpt-4o")

llm = OllamaLLM(model="llama3.2")

embed = OllamaEmbeddings(model="nomic-embed-text")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tech-bot-saas.vercel.app/","*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500,chunk_overlap=200)

doc_pages = []

doc_chunks = []

vector_store = None

document_ids = []

chat_history = [AIMessage(content="Hello I'm a Bot How I Can Help You ?")]

last_url = None

class UrlModel(BaseModel):
    url: str

class ChatModel(BaseModel):
    message: str    


@app.get('/')
async def root():
    return {"message" : "The Chat Website API work succesfully"}


@app.post("/api/scrape")
async def scrape(item:UrlModel):
    global vector_store,chat_history,document_ids,doc_chunks,doc_pages

    if vector_store is not None and len(document_ids) > 0:
        vector_store.delete(ids=document_ids)
        print("vector store boorado --> ",document_ids)
        

    chat_history.clear()
    doc_chunks.clear()
    doc_pages.clear()


    url = item.url

    
    loader = WebBaseLoader(url)
    docs = loader.load()
    doc_pages.extend(docs)
    doc_chunks = text_splitter.split_documents(doc_pages)

    document_ids = [str(uuid.uuid4()) for _ in doc_chunks]
    vector_store = Chroma.from_documents(doc_chunks,embed,ids=document_ids)
    
    print(document_ids)
 
    return {"message":f"Vetor Store Initialized with {url}"}


@app.post('/api/chat')
async def chat(request:ChatModel):
    global vector_store,chat_history
    if vector_store is None:
        return {"error vector": "Vectore Store Is Not Found !!, Try a Vali Url to continue..."}
    user_message = HumanMessage(request.message)

    retriever_chain = get_context_retriever_chain(vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke( {"chat_history": chat_history, "input": user_message.content})
   
    chat_history.append(user_message)
    #return{"chat result 2":user_message.content}
    #return{"chat result 3":response["answer"]}
    return response["answer"]


def get_context_retriever_chain(vector_store: Chroma):
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})

    messages= [
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}"),
        ("user","Given the above conversation, generate  a slef conversation")
    ]

    prompt = ChatPromptTemplate.from_messages(messages=messages)

    retriever_chain = create_history_aware_retriever(llm,retriever,prompt)

    return retriever_chain


def get_conversational_rag_chain(retriever_chain):

    messages = [
        ("system","Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}")
    ]

    prompt = ChatPromptTemplate.from_messages(messages=messages)

    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)

    return create_retrieval_chain(retriever_chain,stuff_documents_chain)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app,host="0.0.0.0",port=8000)
