
from src.pdf_loader import extract_text_from_pdf
from src.text_chunker import chunk_text
from src.embeddings import generate_embeddings


pdf_path = "sample.pdf"

# Step 1: Extract text
with open(pdf_path, "rb") as pdf_file:
    extracted_text = extract_text_from_pdf(pdf_file)

# Step 2: Split the text into chunks
chunks = chunk_text(
    extracted_text,
    chunk_size=500,
    overlap=50
)

if not chunks:
    raise ValueError(
        "No text was extracted. Check your PDF."
    )

# Step 3: Generate embeddings
embeddings = generate_embeddings(chunks)

# Step 4: Inspect the results
print("\nTotal chunks:", len(chunks))
print("Embedding matrix shape:", embeddings.shape)

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}:")
    print(chunk[:150])

    print("Embedding dimensions:", embeddings[i].shape)

print("\nPDF embedding pipeline completed!")