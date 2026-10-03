
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Load the model once when this module is imported
model = SentenceTransformer(MODEL_NAME)


def generate_embeddings(text_chunks):
    """Convert a list of text chunks into numerical vectors."""

    if not text_chunks:
        return model.encode([])

    embeddings = model.encode(
        text_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings