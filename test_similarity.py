
import numpy as np

from src.embeddings import generate_embeddings


sentences = [
    "How can I learn Python programming?",
    "What is the best way to study Python?",
    "The weather is rainy today."
]

embeddings = generate_embeddings(sentences)

# Since the embeddings are normalized,
# their dot product gives cosine similarity.
similarity_matrix = embeddings @ embeddings.T

print("Similarity matrix:\n")
print(np.round(similarity_matrix, 3))

print("\nPython sentence similarity:")
print(round(float(similarity_matrix[0, 1]), 3))

print("\nPython vs weather similarity:")
print(round(float(similarity_matrix[0, 2]), 3))