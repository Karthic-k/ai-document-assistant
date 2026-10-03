
import faiss
import numpy as np

from src.embeddings import generate_embeddings


def build_index(chunks, embeddings):
    """Build a FAISS index from document chunk embeddings."""
    if len(chunks) == 0:
        raise ValueError("No chunks provided.")

    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk must have one embedding.")

    embeddings = np.ascontiguousarray(
        embeddings, dtype=np.float32
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def search_chunks(query, chunks, index, top_k=3):
    """Return the most similar chunks for a user query."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not chunks:
        return []

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if index.ntotal != len(chunks):
        raise ValueError("Index and chunks do not match.")

    # Convert the question into a normalized embedding.
    query_embedding = generate_embeddings([query])

    query_embedding = np.ascontiguousarray(
        query_embedding, dtype=np.float32
    )

    # Retrieve no more results than the index contains.
    k = min(top_k, index.ntotal)

    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        results.append({
            "chunk": chunks[int(idx)],
            "score": float(score),
            "index": int(idx)
        })

    return results
