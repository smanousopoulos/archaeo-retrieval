
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "smanousopoulos/archaeo-bge"

model = SentenceTransformer(MODEL_NAME)

query = "prehistoric drinking vessel"

documents = [
    "A Late Helladic kylix with a shallow bowl and two handles.",
    "A Roman bronze coin depicting the emperor on the obverse.",
    "Palestinian amphora with broad, bag-shaped body and short collar rim with plain rounded lip.",
    "A Mycenaean goblet dating to the 12th century BCE.",
]

query_embedding = model.encode(
    query,
    normalize_embeddings=True,
)

document_embeddings = model.encode(
    documents,
    normalize_embeddings=True,
)

# Cosine similarity for normalised vectors.
scores = document_embeddings @ query_embedding

ranked_indices = np.argsort(scores)[::-1]

print(f"\nQuery: {query}\n")
print("Ranked results:\n")

for rank, index in enumerate(ranked_indices, start=1):
    print(f"{rank}. Score: {scores[index]:.4f}")
    print(f"   {documents[index]}\n")
