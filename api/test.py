from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import uuid


app = FastAPI()
load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

class UrlModel2(BaseModel):
    url: str
    question: str



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

vector_store = None

document_ids = []

@app.post('/api/scrape2')
async def scrape2(item:UrlModel2):
    global vector_store,document_ids

    if vector_store is not None:
        vector_store.delete(ids=document_ids)
        
        


    url = item.url
    question = item.question

    loader = WebBaseLoader(web_path=url)
    docs = loader.load()
    splits = text_splitter.split_documents(docs)


    document_ids = [str(uuid.uuid4()) for _ in splits]
    vector_store = Chroma.from_documents(documents=splits,embedding=OpenAIEmbeddings(),ids=document_ids)

    
    
    retriever = vector_store.as_retriever()

    

    # 2. Incorporate the retriever into a question-answering chain.
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm,prompt)
    rag_chain = create_retrieval_chain(retriever,question_answer_chain)

    response = rag_chain.invoke({"input":question})

    return {"message": response['answer']}






