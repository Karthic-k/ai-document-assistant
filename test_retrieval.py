from src.pdf_loader import extract_text_from_pdf
from src.text_chunker import chunk_text
from src.embeddings import generate_embeddings
from src.retriever import build_index, search_chunks


PDF_PATH = "sample.pdf"

with open(PDF_PATH, "rb") as pdf_file:

    text = extract_text_from_pdf(pdf_file)


chunks = chunk_text(
    text,
    chunk_size=500,
    overlap=50
)

embeddings = generate_embeddings(chunks)

index = build_index(
    chunks,
    embeddings
)


test_questions = [
    "What is the main topic of this document?",
    "What are the main findings?",
    "What methodology is discussed?"
]


for question in test_questions:

    print("\n" + "=" * 70)

    print(f"QUESTION: {question}")

    print("=" * 70)

    results = search_chunks(
        query=question,
        index=index,
        chunks=chunks,
        top_k=3,
        similarity_threshold=0.15
    )

    if not results:

        print("NO RELEVANT CHUNKS FOUND.")
        continue

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRank {rank} "
            f"| Score: {result['score']:.3f} "
            f"| Chunk: {result['index']}"
        )

        print(result["chunk"][:500])