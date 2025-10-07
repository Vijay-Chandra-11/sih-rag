# Advanced Offline RAG Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?style=for-the-badge&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green?style=for-the-badge)
![CUDA](https://img.shields.io/badge/NVIDIA%20CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia)

An advanced, private, and completely offline Retrieval-Augmented Generation (RAG) system built with Streamlit and LangChain. This application allows users to interact with their local documents through a conversational AI, with responses delivered in both text and high-quality audio.

***

## 📖 Table of Contents
- [About The Project](#-about-the-project)
- [✨ Key Features](#-key-features)
- [🏛️ System Architecture](#️-system-architecture)
- [🛠️ Tech Stack](#-tech-stack)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation Guide](#installation-guide)
- [💻 Usage](#-usage)
- [Future Work](#-future-work)

***

## 🏛️ About The Project

This project addresses the challenge of creating a powerful, conversational AI for documents that can operate in a completely secure, offline environment. Unlike cloud-based solutions that require data to be uploaded to third-party servers, this RAG Assistant runs entirely on a local machine. It leverages local Large Language Models (LLMs) and GPU acceleration to provide a fast, private, and robust experience.

The application offers two modes: a quick chat with a single uploaded document, and a powerful "knowledge base" mode that creates a persistent, searchable index of an entire folder of documents.



***

## ✨ Key Features

* **Dual Interaction Modes:**
    * **Single Document Chat:** Quickly upload a single PDF or DOCX file for a temporary, session-based Q&A.
    * **Persistent Folder Chat:** Point the app to a local folder to create a permanent, multi-file knowledge base that loads instantly and updates incrementally as you add new files.
* **100% Offline Operation:** After an initial setup, the entire application runs without any internet connection. All AI models are local, guaranteeing absolute data privacy.
* **GPU Acceleration (CUDA):** Leverages an NVIDIA GPU to dramatically speed up document processing (embeddings) and Text-to-Speech (TTS) generation.
* **Advanced RAG Pipeline:** Uses reliable document loaders (`PyPDFLoader`) and an advanced retrieval strategy (`MMR`) to find the most accurate and diverse sources for any question.
* **Grounded & Verifiable Answers:** Every answer is generated based *only* on the documents you provide. The **"View Sources"** feature shows the exact text chunks used, providing full transparency and trust.
* **Streaming Text Response:** The AI's answer appears word-by-word as it's being generated, creating a highly responsive user experience.
* **High-Quality Offline Audio:** Generates a natural-sounding audio version of every answer using a local, GPU-accelerated Text-to-Speech engine.

***

## 🏛️ System Architecture

The application follows a modern, local-first RAG architecture:

1.  **Frontend (Streamlit):** The user interacts with the web UI, selecting a mode and uploading documents or asking questions.
2.  **Document Processing:** LangChain uses `PyPDFLoader` and `Docx2txtLoader` to read documents. The text is split into chunks.
3.  **Embedding & Indexing:** `HuggingFaceEmbeddings` (running on the CUDA GPU) converts the text chunks into vectors. These vectors are stored in a local `Chroma` vector database. In "Folder Mode," this database is saved to disk for persistence.
4.  **RAG Loop:**
    * A user's question is embedded.
    * The `MMR` retriever finds the most relevant document chunks from ChromaDB.
    * These chunks, along with the original question, are passed to the `phi3` LLM running in `Ollama`.
    * Ollama (using the GPU) generates a text answer based on the provided context.
5.  **Response Delivery:**
    * The text answer is streamed back to the Streamlit UI.
    * `Coqui TTS` (using the GPU) converts the final text into an audio file.
    * Streamlit displays the text, audio player, and source citations.

***

## 🛠️ Tech Stack

* **Framework:** Streamlit
* **AI Orchestration:** LangChain
* **Local LLM:** Ollama (serving the `phi3` model)
* **Vector Database:** ChromaDB
* **Embeddings:** Sentence-Transformers (via `langchain-huggingface`)
* **Document Loading:** `PyPDFLoader`, `Docx2txtLoader`
* **Text-to-Speech:** Coqui TTS

***

## 🚀 Getting Started

### Prerequisites

* **Python 3.11**
* An **NVIDIA GPU** with at least 6GB of VRAM
* **Git** for cloning the repository

### Installation Guide

This is a multi-step process that needs to be followed carefully.

**1. Setup GPU Drivers & CUDA Toolkit**
* Ensure your NVIDIA drivers are up to date.
* Download and install the **CUDA Toolkit 12.1.1** from the [NVIDIA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).
    * During installation, select the **"Express (Recommended)"** option.
    * Restart your PC after installation.
    * Verify the installation by opening a terminal and running `nvcc --version`. It should succeed.

**2. Setup Python Environment**
* Clone your project repository and navigate into the folder.
* Create and activate a Python 3.11 virtual environment:
    ```sh
    # On Windows
    py -3.11 -m venv venv_py11
    .\venv_py11\Scripts\activate
    ```

**3. Install Dependencies**
* First, install the correct version of PyTorch for CUDA 12.1:
    ```sh
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```
* Next, create a file named `requirements.txt` in your project folder and paste the following content into it:
    ```
    streamlit
    ollama
    TTS
    langchain-chroma
    langchain-huggingface
    langchain-ollama
    langchain-community
    langchain-core
    pypdf
    docx2txt
    sentence-transformers
    chromadb
    ```
* Now, install all other libraries from this file:
    ```sh
    pip install -r requirements.txt
    ```

**4. Setup Local LLM & AI Models**
* Install Ollama from the [official website](https://ollama.com/) if you haven't already.
* Pull the required LLM:
    ```sh
    ollama pull phi3
    ```
* Create a file named `setup.py` in your project folder and paste this code into it. This will pre-download the embedding and TTS models for offline use.
    ```python
    # setup.py
    from sentence_transformers import SentenceTransformer
    from TTS.api import TTS

    print("--- Starting Pre-download of All Required AI Models ---")

    print("\n[1/2] Downloading Embedding Model...")
    SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Embedding Model is ready.")

    print("\n[2/2] Downloading TTS Model...")
    TTS("tts_models/en/ljspeech/tacotron2-DDC")
    print("✅ TTS Model is ready.")

    print("\n--- All models have been downloaded and cached. ---")
    ```
* Run the setup script (this requires internet and may take a long time):
    ```sh
    python setup.py
    ```

***

## 💻 Usage

1.  Make sure your **Ollama application is running** in the background.
2.  Start the Streamlit app from your terminal:
    ```sh
    streamlit run app.py
    ```
3.  Your browser will open to the application.
4.  **To use "Chat with a Folder" mode:**
    * Make sure the path in `app.py` for `KNOWLEDGE_BASE_PATH` points to your desired folder.
    * Place your PDF and DOCX files in that folder.
    * The app will automatically process new files.
5.  **To use "Upload a Single Document" mode:**
    * Select the mode in the sidebar and use the file uploader.

***

## 🛣️ Future Work

* **Integrate Image Q&A:** Re-add the "Ask Questions About an Image" mode using a Vision-Language Model like `bakllava` or `llava-phi3`.
* **Add Audio Ingestion:** Implement a speech-to-text pipeline (e.g., using Whisper) to allow the system to process audio recordings.