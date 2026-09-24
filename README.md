# Advanced Offline RAG Assistant: The INDUS AI

An advanced, **Zero-Trust**, and completely **offline Retrieval-Augmented Generation (RAG)** system. This application enables secure, local conversational AI capable of processing universal document types—including **Word, PowerPoint, PDFs, and Images**—and providing answers grounded in the provided context with audio output.

-----

## 📖 Table of Contents

  - [About The Project](https://www.google.com/search?q=%23-about-the-project)
  - [✨ Key Features](https://www.google.com/search?q=%23-key-features)
  - [🏛️ System Architecture](https://www.google.com/search?q=%23%EF%B8%8F-system-architecture)
  - [🛠️ Tech Stack](https://www.google.com/search?q=%23-tech-stack)
  - [🚀 Getting Started](https://www.google.com/search?q=%23-getting-started)
  - [💻 Usage](https://www.google.com/search?q=%23-usage)

-----

## 🏛️ About The Project

The **INDUS AI** addresses the need for powerful conversational AI in environments demanding **Zero-Trust Data Privacy**. Unlike cloud-dependent solutions, this system runs entirely locally using local Small Language Models (SLMs like Phi-3), local vector storage (ChromaDB), and local OCR processing.

It features a specialized **ETL Pipeline** that pre-processes low-quality images and structurally parses complex Office documents to ensure high-quality answers.

-----

## ✨ Key Features

  * **Universal File Support:** Natively processes **.pdf**, **.docx** (Word), **.pptx** (PowerPoint), and Images (\*\* .png, .jpg\*\*).
  * **Computer Vision & OCR:** Integrates **Tesseract OCR** to "read" pixels. It automatically detects text in images, screenshots, or scanned PDFs.
  * **Advanced Pre-Processing (Level 1):** Includes an image enhancement pipeline (Upscaling + Grayscale + Binarization) to significantly improve OCR accuracy on low-quality inputs.
  * **Structural Parsing (Level 2):** Explicitly preserves **Table structures** in Word and PowerPoint documents (formatting them with pipes `|`), ensuring the AI understands rows and columns.
  * **Zero-Trust Privacy:** 100% Offline. Your data never leaves `localhost`.
  * **Session Isolation:** The "Chat with File" feature creates a **unique, isolated vector collection** for every uploaded file (hashed), preventing data leakage between uploads.
  * **Deterministic Sourcing:** explicitly cites the source file and page number for every answer, prioritizing the most relevant document.
  * **GPU Acceleration:** Leverages NVIDIA CUDA for faster embedding generation (`all-MiniLM-L6-v2`) and offline **Text-to-Speech (TTS)** via Coqui.

-----

## 🏛️ System Architecture

1.  **Ingestion & Routing:**
      * **PDFs:** Converted to images via Poppler $\to$ Enhanced $\to$ OCR.
      * **Images:** Pre-processed (Contrast/Upscale) $\to$ OCR.
      * **DOCX/PPTX:** Parsed via `python-docx` / `python-pptx` to extract text and tables.
2.  **Vectorization:**
      * Text is chunked (1000 chars) and embedded using `SentenceTransformers`.
      * Stored in **ChromaDB** (Persistent for Folder mode, Ephemeral for File mode).
3.  **Retrieval (RAG):**
      * Uses **MMR (Maximal Marginal Relevance)** to find diverse, accurate context.
4.  **Generation:**
      * **Phi-3** (via Ollama) generates the text response.
      * **Tacotron2** (Local TTS) generates the audio response.

-----

## 🛠️ Tech Stack

### Backend & AI

| Category | Tools Used |
| :--- | :--- |
| **Language** | Python 3.11+ |
| **API Framework** | FastAPI, Uvicorn |
| **Orchestration** | LangChain (Core, Chroma, Ollama) |
| **LLM** | Ollama (Phi-3) |
| **Vector DB** | ChromaDB (Local) |
| **OCR / Vision** | Tesseract, Poppler, Pillow (PIL), pdf2image |
| **Office Parsing** | `python-docx`, `python-pptx` |
| **Audio** | Coqui TTS, PyTorch (CUDA) |

-----

## 🚀 Getting Started

### Prerequisites

  * **Python 3.10+**
  * **Ollama** installed and running (`ollama serve`).
  * **Tesseract OCR** installed.
  * **Poppler** installed.

### Installation

1.  **Clone the Repo**

    ```bash
    git clone https://github.com/your-username/INDUS-AI-agent.git
    cd INDUS-AI-agent
    ```

2.  **Setup Virtual Environment**

    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Paths**
    Open `main.py` and update the `AgentConfig` class:

    ```python
    class AgentConfig:
        TESSERACT_CMD = r'C:\Path\To\tesseract.exe'
        POPPLER_PATH = r'C:\Path\To\poppler\bin'
        KNOWLEDGE_BASE_PATH = r'C:\Path\To\MyDocs'
    ```

-----

## 💻 Usage

1.  **Start Ollama**

    ```bash
    ollama pull phi3
    ollama serve
    ```

2.  **Start the Backend**

    ```bash
    python -m uvicorn main:app --reload
    ```

    *On startup, the system will auto-scan and ingest your Knowledge Base folder.*

3.  **API Endpoints**

      * `POST /chat-folder`: Chat with your permanent document base.
      * `POST /chat-file`: Upload a file (.pdf, .docx, .pptx, .png) and chat with it instantly.

-----

## 📜 License

MIT License