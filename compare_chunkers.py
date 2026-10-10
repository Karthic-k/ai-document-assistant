from src.pdf_loader import extract_text_from_pdf
from src.text_chunker import (
    chunk_text,
    sentence_aware_chunk_text
)
from src.embeddings import generate_embeddings
from src.retriever import (
    build_index,
    search_chunks
)


# --------------------------------------------------
# Load PDF
# --------------------------------------------------

with open("sample.pdf", "rb") as pdf_file:
    text = extract_text_from_pdf(pdf_file)


# --------------------------------------------------
# Evaluation questions
# --------------------------------------------------

evaluation_questions = [
    {
        "question": "What is the main purpose of machine learning?",
        "expected": "predictions"
    },
    {
        "question": "What is supervised learning?",
        "expected": "labeled examples"
    },
    {
        "question": "What is unsupervised learning?",
        "expected": "structure"
    },
    {
        "question": "What is reinforcement learning?",
        "expected": "rewards"
    },
    {
        "question": "What are the steps in the machine learning workflow?",
        "expected": "training"
    },
    {
        "question": "What is a feature in machine learning?",
        "expected": "input variable"
    },
    {
        "question": "What is regression?",
        "expected": "estimate"
    }
]


# --------------------------------------------------
# Function to evaluate a chunking strategy
# --------------------------------------------------

def evaluate_chunks(name, chunks):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Number of chunks: {len(chunks)}"
    )

    embeddings = generate_embeddings(chunks)

    index = build_index(
        chunks,
        embeddings
    )

    for item in evaluation_questions:

        question = item["question"]
        expected = item["expected"].lower()

        results = search_chunks(
            query=question,
            index=index,
            chunks=chunks,
            top_k=3,
            similarity_threshold=0.15
        )

        print("\nQuestion:")
        print(question)

        if not results:

            print("❌ No results")
            continue

        retrieved_text = " ".join(
            result["chunk"].lower()
            for result in results
        )

        if expected in retrieved_text:
            print("✅ Expected information found in Top-3")
        else:
            print("❌ Expected information NOT found in Top-3")

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"  Rank {rank} | "
                f"Score: {result['score']:.3f} | "
                f"Chunk: {result['index']}"
            )

# --------------------------------------------------
# OLD CHUNKER
# --------------------------------------------------

old_chunks = chunk_text(
    text,
    chunk_size=500,
    overlap=50
)

evaluate_chunks(
    "OLD CHARACTER-BASED CHUNKER",
    old_chunks
)


# --------------------------------------------------
# NEW CHUNKER
# --------------------------------------------------

new_chunks = sentence_aware_chunk_text(
    text,
    chunk_size=500,
    overlap_sentences=1
)

evaluate_chunks(
    "NEW SENTENCE-AWARE CHUNKER",
    new_chunks
)