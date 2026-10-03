
# AI Document Assistant

An AI-powered document question-answering application built using
Retrieval-Augmented Generation (RAG).

Upload a PDF, ask questions about its contents, and receive
AI-generated answers grounded in relevant document passages.

## Features

- PDF text extraction using pypdf
- Overlapping text chunking
- Semantic embeddings using Sentence Transformers
- Vector similarity search using FAISS
- LLM answer generation using Groq
- Interactive Streamlit interface
- Retrieved source passages and similarity scores
- Loading indicators and error handling

## Tech Stack

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- Groq API
- python-dotenv

## Project Structure

```text
ai-document-assistant/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── pdf_loader.py
│   ├── text_chunker.py
│   ├── embeddings.py
│   ├── retriever.py
│   └── answer_generator.py
└── data/
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd ai-document-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your Groq API key

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_supported_groq_model
```

Replace the example values with your own credentials
and a model supported by your Groq account.

Never commit your `.env` file.

### 5. Run the application

```bash
python -m streamlit run app.py
```

## How It Works

1. Upload a PDF.
2. Extract its text.
3. Split the text into overlapping chunks.
4. Generate embeddings for the chunks.
5. Index the embeddings in FAISS.
6. Retrieve relevant passages for a question.
7. Send the question and retrieved context to Groq.
8. Display the generated answer and source passages.

## Current Status

Completed: Days 1–7, including the initial end-to-end RAG
pipeline and Streamlit interface.

## Future Improvements

- Source citations
- Retrieval similarity thresholds
- Improved handling of unanswerable questions
- Retrieval evaluation
- Support for scanned PDFs using OCR
