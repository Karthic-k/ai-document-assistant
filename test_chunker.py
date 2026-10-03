
from src.text_chunker import chunk_text

sample_text = (
    "Machine learning helps computers learn patterns from data. "
    "Supervised learning uses labeled examples. "
    "Unsupervised learning discovers patterns without labels. "
    "Model evaluation helps measure prediction quality."
)

chunks = chunk_text(
    sample_text,
    chunk_size=100,
    overlap=20
)

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)
    print("Characters:", len(chunk))

print("\nTotal chunks:", len(chunks))