
from src.pdf_loader import extract_text_from_pdf
from src.text_chunker import chunk_text
from src.retriever import build_index, search_chunks


# Step 1: Read the PDF
with open("sample.pdf", "rb") as pdf_file:
    text = extract_text_from_pdf(pdf_file)

# Step 2: Create chunks
chunks = chunk_text(
    text,
    chunk_size=500,
    overlap=50
)

if not chunks:
    raise ValueError(
        "No text extracted. Check sample.pdf."
    )

# Step 3: Build the vector index
index = build_index(chunks)

print("Chunks indexed:", index.ntotal)

# Step 4: Ask a question
query = input("\nAsk a question about your PDF: ")

# Step 5: Retrieve relevant chunks
results = search_chunks(
    query=query,
    chunks=chunks,
    index=index,
    top_k=3
)

# Step 6: Display the results
print("\n--- SEARCH RESULTS ---")

for rank, result in enumerate(results, start=1):
    print(f"\nResult {rank}")
    print(f"Similarity: {result['score']:.4f}")
    print(f"Chunk index: {result['index']}")
    print(result["chunk"])
