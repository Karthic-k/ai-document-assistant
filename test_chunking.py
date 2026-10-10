from src.pdf_loader import extract_text_from_pdf

from src.text_chunker import (
    chunk_text,
    sentence_aware_chunk_text
)


# --------------------------------------------------
# Load PDF
# --------------------------------------------------

with open("sample.pdf", "rb") as pdf_file:

    text = extract_text_from_pdf(pdf_file)


# --------------------------------------------------
# Old chunking
# --------------------------------------------------

old_chunks = chunk_text(
    text,
    chunk_size=500,
    overlap=50
)


# --------------------------------------------------
# New chunking
# --------------------------------------------------

new_chunks = sentence_aware_chunk_text(
    text,
    chunk_size=500
)


# --------------------------------------------------
# Display old chunks
# --------------------------------------------------

print("\n" + "=" * 70)
print("OLD CHARACTER-BASED CHUNKING")
print("=" * 70)

print(
    f"Number of chunks: {len(old_chunks)}"
)


for i, chunk in enumerate(
    old_chunks[:3],
    start=1
):

    print(f"\n--- Chunk {i} ---")
    print(chunk)


# --------------------------------------------------
# Display new chunks
# --------------------------------------------------

print("\n" + "=" * 70)
print("NEW SENTENCE-AWARE CHUNKING")
print("=" * 70)

print(
    f"Number of chunks: {len(new_chunks)}"
)


for i, chunk in enumerate(
    new_chunks[:3],
    start=1
):

    print(f"\n--- Chunk {i} ---")
    print(chunk)