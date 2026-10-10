
import faiss
import numpy as np

from src.embeddings import generate_embeddings,model


def build_index(chunks, embeddings):
    """Build a FAISS index from chunk embeddings."""

    if len(chunks) == 0:
        raise ValueError("No chunks provided.")

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Each chunk must have one embedding."
        )

    embeddings = np.ascontiguousarray(
        embeddings,
        dtype=np.float32
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index

def search_chunks(
    query,
    index,
    chunks,
    top_k=3,
    similarity_threshold=0.15
):
    """Retrieve relevant chunks with metadata."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if not 0 <= similarity_threshold <= 1:
        raise ValueError(
            "similarity_threshold must be between 0 and 1."
        )

    if index.ntotal != len(chunks):
        raise ValueError(
            "Index and chunks do not match."
        )

    if index.ntotal == 0:
        return []

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.ascontiguousarray(
        query_embedding,
        dtype=np.float32
    )

    k = min(top_k, index.ntotal)

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        score = float(score)

        if score < similarity_threshold:
            continue

        chunk = chunks[int(idx)]

        results.append({
            "chunk": chunk["text"],
            "score": score,
            "index": chunk["chunk_index"],
            "page_number": chunk["page_number"]
        })

    return results