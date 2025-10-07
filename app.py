# --- Force Hugging Face Transformers to be offline ---
import os
os.environ['HF_HUB_OFFLINE'] = '1'

import streamlit as st
import ollama
import tempfile
import hashlib
from TTS.api import TTS

# --- LANGCHAIN AND AI LIBRARIES (MODERN IMPORTS) ---
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Advanced RAG Assistant", layout="wide")
st.title("📄 Advanced RAG Assistant")

# --- GLOBAL CONFIGURATION ---
CHROMA_DB_PATH = "./chroma_db_folder"
KNOWLEDGE_BASE_PATH = r"C:\Users\vijay\OneDrive\Desktop\RAG-Base"

# --- CORE FUNCTIONS ---

@st.cache_resource
def get_tts_model():
    return TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=True)

@st.cache_data
def generate_audio_file(text):
    tts = get_tts_model()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
        tts.tts_to_file(text=text, file_path=fp.name)
        return fp.name

def get_file_hash(file):
    hasher = hashlib.md5()
    for chunk in iter(lambda: file.read(4096), b""):
        hasher.update(chunk)
    file.seek(0)
    return hasher.hexdigest()

# Function for single, temporary uploads
@st.cache_resource
def create_single_file_vector_store(file_hash, file_path, file_name):
    st.info(f"Processing {file_name} on GPU...")
    loader = PyPDFLoader(file_path) if file_name.endswith('.pdf') else Docx2txtLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    for split in splits:
        split.metadata['source'] = file_name
    model_kwargs = {'device': 'cuda'}
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs=model_kwargs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    st.success(f"File '{file_name}' processed and ready!")
    return vectorstore

# Function for the persistent, multi-file folder
@st.cache_resource
def load_or_create_folder_vector_store(folder_path):
    st.info(f"Loading knowledge base from '{folder_path}'...")
    model_kwargs = {'device': 'cuda'}
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs=model_kwargs)
    vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    
    indexed_files = [metadata['source'] for metadata in vectorstore.get()['metadatas']]
    current_files = [f for f in os.listdir(folder_path) if f.endswith(('.pdf', '.docx'))]
    new_files = [f for f in current_files if f not in indexed_files]
    
    if new_files:
        st.info(f"New documents found: {', '.join(new_files)}. Processing...")
        all_new_docs = []
        for filename in new_files:
            file_path = os.path.join(folder_path, filename)
            try:
                loader = PyPDFLoader(file_path) if filename.endswith('.pdf') else Docx2txtLoader(file_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                for split in splits:
                    split.metadata['source'] = filename
                all_new_docs.extend(splits)
                st.write(f"Processed '{filename}'.")
            except Exception as e:
                st.error(f"Error processing file {filename}: {e}")
        
        if all_new_docs:
            st.info("Creating embeddings for new documents on GPU...")
            vectorstore.add_documents(all_new_docs)
            st.success("Knowledge base updated successfully!")
    else:
        st.success("Knowledge base is up-to-date.")

    return vectorstore

def chat_interface(vectorstore):
    st.header("Ask a Question")
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        question = st.text_input("Enter your question...", key="question_input", label_visibility="collapsed")
    with col2:
        search_button = st.button("🔍", use_container_width=True)

    if search_button or question:
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching, thinking, and generating answer..."):
                retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 20})
                retrieved_docs = retriever.invoke(question)
                if not retrieved_docs:
                    st.warning("Could not find any relevant information in the document(s).")
                else:
                    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
                    try:
                        llm = OllamaLLM(model="phi3")
                        template = "Use the following context to answer the question concisely... \n\nContext: {context}\nQuestion: {question}\nAnswer:"
                        prompt = PromptTemplate.from_template(template)
                        rag_chain = prompt | llm | StrOutputParser()
                        
                        st.success("Here is the answer:")
                        response_stream = rag_chain.stream({"context": context, "question": question})
                        response = st.write_stream(response_stream)

                        with st.spinner("Generating audio..."):
                            audio_file_path = generate_audio_file(response)
                            if audio_file_path:
                                st.audio(audio_file_path)
                        
                        # --- MODIFIED: Replaced the expander with a single source line ---
                        if retrieved_docs:
                            top_source_filename = retrieved_docs[0].metadata['source']
                            st.markdown(f"**Source:** `{top_source_filename}`")

                    except Exception as e:
                        st.error(f"An error occurred with the LLM: {e}. Is the Ollama server running?")

# --- MAIN APPLICATION LOGIC ---
def main():
    with st.sidebar:
        st.header("Select Mode")
        app_mode = st.radio(
            "Choose your interaction mode:",
            ("Upload a Single Document", "Chat with a Folder")
        )
        st.markdown("---")

    if app_mode == "Upload a Single Document":
        st.write("Upload a single PDF or DOCX file for a focused chat session.")
        with st.sidebar:
            st.header("Upload Your Document")
            uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx"], label_visibility="collapsed")

        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            file_hash = get_file_hash(uploaded_file)
            vectorstore = create_single_file_vector_store(file_hash, tmp_file_path, uploaded_file.name)
            chat_interface(vectorstore)
        else:
            st.warning("Please upload a file to begin.")

    elif app_mode == "Chat with a Folder":
        st.write(f"Ask questions across all documents located in: `{KNOWLEDGE_BASE_PATH}`")
        if not os.path.exists(KNOWLEDGE_BASE_PATH):
            os.makedirs(KNOWLEDGE_BASE_PATH)
        
        vectorstore = load_or_create_folder_vector_store(KNOWLEDGE_BASE_PATH)
        
        if vectorstore and vectorstore._collection.count() > 0:
            chat_interface(vectorstore)
        else:
            st.warning(f"The folder '{KNOWLEDGE_BASE_PATH}' is empty. Please add PDF or DOCX files and refresh the page.")

if __name__ == "__main__":
    main()