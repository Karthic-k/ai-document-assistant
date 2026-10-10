from src.pdf_loader import extract_pages_from_pdf
from src.text_chunker import create_page_aware_chunks


with open("sample.pdf", "rb") as pdf:
    pages = extract_pages_from_pdf(pdf)


chunks = create_page_aware_chunks(
    pages,
    chunk_size=500,
    overlap_sentences=1
)


print(f"Total chunks: {len(chunks)}")


for chunk in chunks:

    print("\n" + "=" * 60)

    print(f"Chunk index: {chunk['chunk_index']}")
    print(f"Page number: {chunk['page_number']}")
    print(f"Characters: {len(chunk['text'])}")

    print("\nText:")
    print(chunk["text"][:200])
    