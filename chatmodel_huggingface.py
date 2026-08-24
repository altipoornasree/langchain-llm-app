
import streamlit as st
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint,HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough
from pydantic import BaseModel,Field
from typing import Literal
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()
def model():
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
    )
    return ChatHuggingFace(llm=llm)
# step1 to load the document in memory
def load_pdf(data_dir):
    loader=DirectoryLoader(
        path=data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()
# to split the data
def splitting(docs):
    text_split=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    text_chunks=text_split.split_documents(docs)
    return text_chunks
# creating embedding
def embedding():
    embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
    return embeddings
#storing the chunks in form of vector base
def create_vector_store(chunks,embedding,store_path="vectorstore/db_faiss"):
    db=FAISS.from_documents(
        chunks,embedding
    )
    db.save_local(store_path)
    return db
# creating the retrival chain
def retrival_chain(llm,db):
    retriver=db.as_retriever(search_kwargs={"k":3})
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the context below.
    Context: {context}
    Question: {question}
    """)
    
    chain = (
        {"context": retriver, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


st.set_page_config(page_title="MedBot", page_icon="🩺")

@st.cache_resource
def init_chain():
    llm = model()
    docs = load_pdf("data/")
    chunks = splitting(docs)
    embed = embedding()
    db = create_vector_store(chunks, embed)
    return retrival_chain(llm, db)

chain = init_chain()

st.title("🩺 MedBot - Medical PDF Assistant")

col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input("Enter your question", label_visibility="collapsed")
with col2:
    submitted = st.button("Ask")

if submitted and question.strip():
    answer = chain.invoke(question)
    st.write(answer)
