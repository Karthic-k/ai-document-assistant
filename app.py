
import hashlib

import streamlit as st

from src.pdf_loader import extract_pages_from_pdf
from src.text_chunker import create_page_aware_chunks
from pypdf import PdfReader
from pypdf.errors import PdfReadError
# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Document Assistant")
st.write(
    "Upload a PDF and ask questions about its contents. "
    "Answers are generated using relevant passages from your document."
)


# --------------------------------------------------
# 2. Initialize session state
# --------------------------------------------------

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "answer" not in st.session_state:
    st.session_state.answer = None

if "sources" not in st.session_state:
    st.session_state.sources = []

if "last_question" not in st.session_state:
    st.session_state.last_question = ""


# --------------------------------------------------
# 3. Upload a PDF
# --------------------------------------------------

st.header("1. Upload your document")

uploaded_pdf = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Upload a text-based PDF to get started."
)

current_document_id = None

if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.getvalue()
    current_document_id = hashlib.sha256(pdf_bytes).hexdigest()

    st.info(f"Selected document: {uploaded_pdf.name}")

    # Prevent using an old document's index after a new upload.
    if current_document_id != st.session_state.document_id:
        st.session_state.chunks = []
        st.session_state.index = None
        st.session_state.document_id = None
        st.session_state.answer = None
        st.session_state.sources = []
        st.session_state.last_question = ""


# --------------------------------------------------
# 4. Process the PDF and build the search index
# --------------------------------------------------

if uploaded_pdf is not None:
    if st.button("Process PDF", type="primary"):
        try:
            with st.spinner(
                "Reading PDF, creating embeddings, and building the search index..."
            ):
                # Import here so model loading happens during processing.
                from src.embeddings import generate_embeddings
                from src.retriever import build_index
                # Validate the uploaded file before processing.
                uploaded_pdf.seek(0)

                if not uploaded_pdf.getvalue().strip():
                    st.error("The uploaded file is empty.")
                    st.stop()

                try:
                    PdfReader(uploaded_pdf, strict=True)
                    uploaded_pdf.seek(0)
                except (PdfReadError, ValueError, OSError):
                    st.error("Invalid or corrupted PDF. Please upload a valid PDF.")
                    st.stop()
                # Extract text from the uploaded PDF.
                pages = extract_pages_from_pdf(uploaded_pdf)

                if not pages:
                    st.error(
                        "No readable text was found. "
                        "This may be a scanned PDF requiring OCR."
                    )
                else:
                    chunks = create_page_aware_chunks(
                        pages,
                        chunk_size=500,
                        overlap_sentences=1
                    )

                    if not chunks:
                        st.error(
                            "Could not create text chunks from this PDF."
                        )
                    else:
                        chunk_texts = [
                            chunk["text"]
                            for chunk in chunks
                        ]

                        embeddings = generate_embeddings(
                            chunk_texts
                        )

                        index = build_index(
                            chunks,
                            embeddings
                        )

                        st.session_state.chunks = chunks
                        st.session_state.index = index
                        st.session_state.document_id = current_document_id
                        st.session_state.answer = None
                        st.session_state.sources = []
                        st.session_state.last_question = ""

                        st.success(
                            f"PDF processed successfully! "
                            f"{len(chunks)} chunks indexed."
                        )

        except Exception as e:
            st.error("The PDF could not be processed.")
            st.write("DEBUG: Exception handler reached")
            st.code(repr(e))

# --------------------------------------------------
# 5. Ask a question
# --------------------------------------------------

document_ready = (
    uploaded_pdf is not None
    and current_document_id == st.session_state.document_id
    and st.session_state.index is not None
)

st.divider()
st.header("2. Ask a question")

if not document_ready:
    st.info("Upload and process a PDF before asking questions.")
else:
    with st.form("question_form"):
        question = st.text_input(
            "What would you like to know?",
            placeholder="Example: What are the main findings?"
        )

        submitted = st.form_submit_button("Generate answer")

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                from src.retriever import search_chunks
                from src.answer_generator import generate_answer

                with st.spinner(
                    "Searching your document and generating an answer..."
                ):
                    # Retrieve the most relevant passages.
                    results = search_chunks(
                    query=question,
                    index=st.session_state.index,
                    chunks=st.session_state.chunks,
                    top_k=3,
                    similarity_threshold=0.35
                )

                if not results:
                    st.warning(
                        "I couldn't find sufficiently relevant information "
                        "in this document to answer that question."
                    )

                    st.session_state.answer = None
                    st.session_state.sources = []

                else:
                    answer = generate_answer(
                        question,
                        results
                    )

                    st.session_state.answer = answer
                    st.session_state.sources = results
                    st.session_state.last_question = question

            except Exception as e:
                import traceback

                st.error(
                        f"PDF processing failed: {type(e).__name__}: {e}"
                    )

                st.code(
                        traceback.format_exc()
                    )


# --------------------------------------------------
# 6. Display the answer
# --------------------------------------------------

if st.session_state.answer is not None:
    st.divider()
    st.header("3. AI-generated answer")

    st.markdown(f"**Question:** {st.session_state.last_question}")
    st.markdown(st.session_state.answer)

    # Show the passages used to generate the answer.
    with st.expander("📚 View supporting source passages"):

        for number, result in enumerate(
            st.session_state.sources,
            start=1
        ):
            st.markdown(
                f"### 📌 Passage {number}"
            )

            st.caption(
                f"📄 Page: {result['page_number']} "
                f"| 🎯 Similarity: {result['score']:.3f} "
                f"| 🔹 Chunk: {result['index']}"
            )

            st.write(result["chunk"])

            st.divider()


# --------------------------------------------------
# 7. Footer
# --------------------------------------------------

st.caption(
    "AI-generated answers may contain errors. "
    "Verify important information against the original PDF."
)
