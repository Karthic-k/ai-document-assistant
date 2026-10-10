from src.pdf_loader import extract_pages_from_pdf
from src.text_chunker import create_page_aware_chunks
from src.embeddings import generate_embeddings
from src.retriever import build_index, search_chunks


# --------------------------------------------------
# 1. Load PDF
# --------------------------------------------------

with open("sample.pdf", "rb") as pdf:
    pages = extract_pages_from_pdf(pdf)


# --------------------------------------------------
# 2. Create page-aware chunks
# --------------------------------------------------

chunks = create_page_aware_chunks(
    pages,
    chunk_size=500,
    overlap_sentences=1
)


# --------------------------------------------------
# 3. Create embeddings
# --------------------------------------------------

chunk_texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = generate_embeddings(
    chunk_texts
)


# --------------------------------------------------
# 4. Build FAISS index
# --------------------------------------------------

index = build_index(
    chunks,
    embeddings
)


# --------------------------------------------------
# 5. Test question
# --------------------------------------------------

questions = [
    "What is machine learning?",
    "What is supervised learning?",
    "What is reinforcement learning?",
    "What is regression?",
    "What are the steps in the machine learning workflow?"
]

# --------------------------------------------------
# 6. Compare different Top-K values
# --------------------------------------------------

for question in questions:

    print("\n" + "#" * 60)
    print(f"QUESTION: {question}")
    print("#" * 60)

    for threshold in [0.15, 0.30, 0.40, 0.50]:

        print("\n" + "=" * 60)
        print(f"THRESHOLD = {threshold}")
        print("=" * 60)

        results = search_chunks(
            query=question,
            index=index,
            chunks=chunks,
            top_k=3,
            similarity_threshold=threshold
        )

        for rank, result in enumerate(
            results,
            start=1
        ):
            print(
                f"Rank {rank} | "
                f"Score {result['score']:.3f} | "
                f"Page {result['page_number']} | "
                f"Chunk {result['index']}"
            )