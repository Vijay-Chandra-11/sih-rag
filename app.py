# app.py

import streamlit as st
import ollama
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

# --- App Title and Configuration ---
st.set_page_config(page_title="Offline RAG Assistant", layout="wide")
st.title("📄 Offline Multimodal RAG Assistant")
st.write("Upload a document and ask questions about its content. All processing is done locally on your machine.")

# --- Caching the Vector Store ---
# This is a key optimization. The function will only rerun if the input (file path) changes.
@st.cache_resource
def create_vector_store(file_path):
    """Loads a PDF, splits it into chunks, creates embeddings, and stores them in a vector store."""
    st.info(f"Creating vector store for {os.path.basename(file_path)}... This may take a moment.")
    
    # 1. Load the document
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # 2. Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # 3. Create embeddings and store in ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    st.success(f"Vector store created successfully for {os.path.basename(file_path)}!")
    return vectorstore

# --- Main App Logic ---
with st.sidebar:
    st.header("1. Upload Your Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    st.markdown("---")
    st.header("2. Configure LLM")
    st.info("Make sure Ollama is running with the 'phi3' model pulled.")

vectorstore = None
if uploaded_file is not None:
    # Use a temporary file to store the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    # Create (or load from cache) the vector store
    vectorstore = create_vector_store(tmp_file_path)

# --- Chat Interface ---
if vectorstore:
    st.header("3. Ask a Question")
    question = st.text_input("Enter your question about the document:", placeholder="What is this document about?")

    if question:
        with st.spinner("Thinking..."):
            # 1. Setup the Retriever from the vector store
            retriever = vectorstore.as_retriever()
            
            # 2. Setup the LLM and RAG Chain
            llm = Ollama(model="phi3")
            
            template = """
            You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Keep the answer concise.

            Context: {context} 

            Question: {question} 

            Answer:
            """
            prompt = PromptTemplate.from_template(template)

            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            # 3. Get the response
            response = rag_chain.invoke(question)
            
            # 4. Display the response
            st.success("Here is the answer:")
            st.write(response)

else:
    st.warning("Please upload a PDF document in the sidebar to get started.")