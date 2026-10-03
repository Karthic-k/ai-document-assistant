
from src.pdf_loader import extract_text_from_pdf
from src.text_chunker import chunk_text

pdf_path = "sample.pdf"

with open(pdf_path, "rb") as pdf_file:
    extracted_text = extract_text_from_pdf(pdf_file)

chunks = chunk_text(
    extracted_text,
    chunk_size=500,
    overlap=50
)

print("\nTotal characters:", len(extracted_text))
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)
    print("Length:", len(chunk))