# Offline Multimodal RAG Assistant (SIH 2025)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?style=for-the-badge&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-lightgrey?style=for-the-badge)

An offline, multimodal Retrieval-Augmented Generation (RAG) system designed for the Smart India Hackathon (SIH) 2025, based on Problem Statement **ID25231**.

This application allows users to chat with their documents and images in a completely private, offline environment, ensuring data security and accessibility without an internet connection.

***

## 📖 Table of Contents
- [About The Project](#-about-the-project)
- [✨ Features](#-features)
- [🛠️ Tech Stack](#-tech-stack)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [💻 Usage](#-usage)
- [🛣️ Future Work](#️-future-work)

***

## 🏛️ About The Project

Traditional search tools struggle to understand and query diverse data formats like documents and images simultaneously. This project solves that problem by building a unified semantic retrieval framework using a Large Language Model (LLM).

The core of the system is a **Retrieval-Augmented Generation (RAG)** pipeline that can ingest PDFs, DOCX files, and images. It uses intelligent, context-aware chunking to process text and multimodal embeddings for images, storing everything in a local vector database. When a user asks a question, the system retrieves the most relevant text and image sources and feeds them to a locally-run LLM to generate a grounded, accurate answer with citations.

The entire process, from data processing to model inference, runs **100% offline**, making it ideal for secure and sensitive environments.



***

## ✨ Features

* **100% Offline:** All models (LLM, embeddings) and data processing run locally. No internet connection or API keys are required for operation.
* **Multimodal Ingestion:** Natively supports PDF, DOCX, and image formats (PNG, JPG, JPEG).
* **Intelligent Chunking:** Uses the `unstructured` library to parse documents based on their logical structure (paragraphs, titles, lists), leading to highly accurate context retrieval.
* **Accurate Retrieval:** Employs advanced search strategies (like MMR) to find the most relevant text chunks and images for a given query.
* **Grounded AI Answers:** Leverages a local LLM (Ollama with Phi-3) to generate answers based directly on the content from the uploaded files.
* **Citation Transparency:** Every answer is accompanied by a "View Sources" section, showing the exact text chunks used to formulate the response.
* **Simple & Interactive UI:** A clean and user-friendly web interface built with Streamlit.

***

## 🛠️ Tech Stack

This project is built with a modern, open-source stack:

* **Backend & Framework:** Python, Streamlit
* **AI & NLP:**
    * **LLM Engine:** [Ollama](https://ollama.com/) (running the Phi-3 model)
    * **RAG Framework:** [LangChain](https://www.langchain.com/)
    * **Embeddings:** [Sentence-Transformers](https://sbert.net/) (`clip-ViT-B-32` for multimodal understanding)
    * **Vector Database:** [ChromaDB](https://www.trychroma.com/)
    * **Document Processing:** [Unstructured.io](https://unstructured.io/)
* **Image Processing:** Pillow

***

## 🚀 Getting Started

Follow these steps to get the application running on your local machine.

### Prerequisites

1.  **Python:** Ensure you have Python 3.10 or newer installed.
2.  **Ollama:** Download and install Ollama from the [official website](https://ollama.com/).
3.  **System Dependencies (for `unstructured`):** To process PDFs, you may need to install `poppler`.
    * Follow the installation guide here: [How to install Poppler](https://github.com/oschwartz10612/poppler-windows/releases/)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://your-repository-url/sih_multimodal_rag.git
    cd sih_multimodal_rag
    ```

2.  **Create and activate a Python virtual environment:**
    ```sh
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required Python libraries:**
    ```sh
    pip install streamlit langchain-community "unstructured[pdf,docx]" chromadb sentence-transformers Pillow ollama
    ```

4.  **Pull the local LLM using Ollama:**
    * Open a new terminal and run the following command. This will download the Phi-3 model (this is a one-time step).
    ```sh
    ollama pull phi3
    ```
    * Make sure the Ollama application is running in the background.

***

## 💻 Usage

1.  **Run the Streamlit app:**
    * In your terminal (with the virtual environment activated), navigate to the project directory and run:
    ```sh
    streamlit run app.py
    ```

2.  **Upload a file:**
    * Your web browser will open with the application.
    * Use the sidebar to upload a PDF, DOCX, or image file. The app will process it and create the knowledge base.

3.  **Ask a question:**
    * Once the file is processed, the main chat interface will appear.
    * Type your question into the text box and press **Enter**.
    * The app will retrieve relevant information, generate a text-based answer, and provide the sources it used.

***

## 🛣️ Future Work

* **Add Audio Ingestion:** Integrate a speech-to-text model (like Whisper) to support querying audio files.
* **Persistent Knowledge Base:** Modify the app to load all files from a directory on startup, creating a persistent knowledge base.
* **Advanced Cross-Modal Linking:** Implement logic to explicitly link a specific image to a relevant text chunk in the answer.