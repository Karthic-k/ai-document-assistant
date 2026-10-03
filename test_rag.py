
from src.pdf_loader import extract_text_from_pdf
from src.text_chunker import chunk_text
from src.embeddings import generate_embeddings
from src.retriever import build_index, search_chunks
from src.answer_generator import generate_answer


# Step 1: Load and extract the PDF.
with open("sample.pdf", "rb") as pdf_file:
    text = extract_text_from_pdf(pdf_file)

# Step 2: Split the extracted text.
chunks = chunk_text(
    text,
    chunk_size=500,
    overlap=50
)

if not chunks:
    raise ValueError(
        "No extractable text found in sample.pdf."
    )

# Step 3: Generate document embeddings.
embeddings = generate_embeddings(chunks)

# Step 4: Build the FAISS index.
index = build_index(chunks, embeddings)

print(f"\nIndexed {index.ntotal} chunks.")

# Step 5: Ask a question.
question = input("\nAsk a question about your PDF: ")

# Step 6: Retrieve relevant passages.
results = search_chunks(
    query=question,
    index=index,
    chunks=chunks,
    top_k=3
)

print("\n--- Retrieved passages ---")

for i, result in enumerate(results, start=1):
    print(f"\n[Passage {i}]")
    print(result["chunk"])
    print(f"Similarity: {result['score']:.3f}")

# Step 7: Generate an answer using Groq.
print("\n--- AI Answer ---")

answer = generate_answer(question, results)
print(answer)
