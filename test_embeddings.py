
import numpy as np

from src.embeddings import generate_embeddings


sentences = [
    "Machine learning helps computers learn from data.",
    "AI models discover patterns in training examples.",
    "I enjoy playing cricket on weekends."
]

embeddings = generate_embeddings(sentences)

print("Embedding shape:", embeddings.shape)
print("Data type:", embeddings.dtype)

print("\nFirst sentence embedding:")
print(embeddings[0])

print("\nVector length:", len(embeddings[0]))

# Check that the vectors are normalized
norms = np.linalg.norm(embeddings, axis=1)
print("\nVector norms:", norms)

assert embeddings.shape == (3, 384)
assert np.allclose(norms, 1.0, atol=1e-5)

print("\nEmbedding test passed!")