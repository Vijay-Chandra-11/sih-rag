# --- NEW: Force Hugging Face Transformers to be offline ---
import os
os.environ['HF_HUB_OFFLINE'] = '1'
# --- END NEW ---

import streamlit as st
import ollama
import tempfile
import hashlib
from TTS.api import TTS

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

# --- App Title and Configuration ---
st.set_page_config(page_title="Advanced RAG Assistant", layout="wide")
st.title("📄 Advanced RAG Assistant")
st.write("Using intelligent chunking, GPU acceleration, and offline TTS to provide accurate sources.")

# --- All functions are defined first ---

@st.cache_resource
def get_tts_model():
    """Initializes the TTS model and caches it."""
    return TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=True)

@st.cache_data
def generate_audio_file(text):
    """Generates an audio file from text and returns its path."""
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

@st.cache_resource
def create_vector_store(file_hash, file_path, file_name):
    st.info(f"Parsing and indexing {file_name}... This may take a moment.")
    
    if file_name.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else: # .docx
        loader = Docx2txtLoader(file_path)
    
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    for split in splits:
        split.metadata['source'] = file_name

    model_kwargs = {'device': 'cuda'}
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs=model_kwargs
    )
    
    st.info("Creating embeddings on GPU...")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    st.success(f"File '{file_name}' processed and added to the knowledge base!")
    return vectorstore

# --- Main execution block ---
def main():
    with st.sidebar:
        st.header("Upload Your Document")
        uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])

    vectorstore = None
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        file_hash = get_file_hash(uploaded_file)
        vectorstore = create_vector_store(file_hash, tmp_file_path, uploaded_file.name)

    if vectorstore:
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
                with st.spinner("Searching and thinking..."):
                    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 20})
                    retrieved_docs = retriever.invoke(question)
                    
                    if not retrieved_docs:
                        st.warning("Could not find any relevant information in the document.")
                    else:
                        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
                        llm = Ollama(model="phi3")
                        template = "Use the following context to answer the question concisely. If you don't know, say that. \n\nContext: {context} \nQuestion: {question} \nAnswer:"
                        prompt = PromptTemplate.from_template(template)
                        rag_chain = prompt | llm | StrOutputParser()
                        
                        st.success("Here is the answer:")
                        response_stream = rag_chain.stream({"context": context, "question": question})
                        response = st.write_stream(response_stream)

                        with st.spinner("Generating audio..."):
                            audio_file_path = generate_audio_file(response)
                            st.audio(audio_file_path)
                        
                        # with st.expander("View Sources"):
                        #     st.info("The answer was based on the following sources, ranked by relevance.")
                        #     for i, doc in enumerate(retrieved_docs):
                        #         st.markdown(f"**Source {i+1} (from `{doc.metadata['source']}`)**")
                        #         st.write(doc.page_content)
                        #         st.markdown("---")
    else:
        st.warning("Please upload a file in the sidebar to get started.")

if __name__ == "__main__":
    main()