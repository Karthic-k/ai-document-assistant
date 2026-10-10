evaluation_questions = [
    {
        "question": "What is machine learning?",
        "relevant_chunks": [0]
    },
    {
        "question": "What is supervised learning?",
        "relevant_chunks": [1]
    },
    {
        "question": "What is reinforcement learning?",
        "relevant_chunks": [2]
    },
    {
        "question": "What is regression?",
        "relevant_chunks": [6]
    },
    {
        "question": "What are the steps in the machine learning workflow?",
        "relevant_chunks": [3]
    }
]

from src.pdf_loader import extract_pages_from_pdf
from src.text_chunker import create_page_aware_chunks
from src.embeddings import generate_embeddings
from src.retriever import build_index, search_chunks


# Load PDF
with open("sample.pdf", "rb") as pdf:
    pages = extract_pages_from_pdf(pdf)


# Create page-aware chunks
chunks = create_page_aware_chunks(
    pages,
    chunk_size=500,
    overlap_sentences=1
)


# Extract chunk text
chunk_texts = [
    chunk["text"]
    for chunk in chunks
]


# Generate embeddings
embeddings = generate_embeddings(chunk_texts)


# Build FAISS index
index = build_index(
    chunks,
    embeddings
)

recall_at_1 = []
recall_at_3 = []


for item in evaluation_questions:

    question = item["question"]
    relevant_chunks = item["relevant_chunks"]

    print("\n" + "=" * 60)
    print(f"QUESTION: {question}")
    print(f"EXPECTED CHUNKS: {relevant_chunks}")
    print("=" * 60)

    for k in [1, 3]:

        results = search_chunks(
            query=question,
            index=index,
            chunks=chunks,
            top_k=k,
            similarity_threshold=0.15
        )

        retrieved_chunks = [
            result["index"]
            for result in results
        ]

        found = any(
            chunk in retrieved_chunks
            for chunk in relevant_chunks
        )

        recall = 1 if found else 0

        print(
            f"Recall@{k}: {recall} "
            f"| Retrieved: {retrieved_chunks}"
        )

        # Store result for final calculation
        if k == 1:
            recall_at_1.append(recall)

        elif k == 3:
            recall_at_3.append(recall)


print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

recall_1 = sum(recall_at_1) / len(recall_at_1)
recall_3 = sum(recall_at_3) / len(recall_at_3)

print(f"Recall@1: {recall_1:.2%}")
print(f"Recall@3: {recall_3:.2%}")