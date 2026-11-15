"""
Layer 2 defence (revised): centroid distance over chunk embeddings.

Isolation Forest was rejected after measurement (see defense2.py): it scores
by isolation difficulty, which does not correspond to distance from the
corpus distribution for L2-normalised sentence embeddings.

This version measures the quantity actually required: cosine distance from
the centroid of the clean corpus.
"""

import os
import joblib
import numpy as np
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "./chroma_db"
MODEL_PATH = "./knn_model.joblib"
K = 3
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def load_corpus_embeddings():
    store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings(),
    )
    data = store._collection.get(include=["embeddings"])
    return np.array(data["embeddings"])


def cosine_distance(vectors, centroid):
    """Cosine distance from each vector to the centroid. 0 = identical."""
    # Normalise both sides so the dot product is exactly cosine similarity.
    v_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    c_norm = centroid / np.linalg.norm(centroid)
    similarity = v_norm @ c_norm
    return 1.0 - similarity


def fit_baseline():
    X = load_corpus_embeddings()
    # exclude_self: each clean chunk's nearest match is itself (distance 0),
    # which would make the clean corpus look artificially tight.
    distances = knn_distance(X, X, exclude_self=True)
    stats = {
        "corpus": X,
        "clean_min": float(distances.min()),
        "clean_max": float(distances.max()),
        "clean_mean": float(distances.mean()),
        "clean_std": float(distances.std()),
    }
    joblib.dump(stats, MODEL_PATH)
    print(f"Fitted on {X.shape[0]} clean chunks, k={K}")
    print(f"  distance min    {stats['clean_min']:.4f}")
    print(f"  distance max    {stats['clean_max']:.4f}")
    print(f"  distance mean   {stats['clean_mean']:.4f}")
    return stats


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No model at {MODEL_PATH}. Run fit_baseline() first.")
    return joblib.load(MODEL_PATH)


def check_anomaly(chunk_texts, stats, embedder=None, threshold=None):
    """Flag chunks whose distance from the corpus centroid exceeds threshold.

    Default threshold is the maximum distance observed across the clean
    corpus — anything beyond that is further from the corpus centre than
    any legitimate chunk, so flagging it costs no false positives on the
    training data.
    """
    if threshold is None:
        threshold = stats["clean_max"]

    embedder = embedder or get_embeddings()
    vectors = np.array(embedder.embed_documents(chunk_texts))
    distances = knn_distance(vectors, stats["corpus"])

    report = []
    is_clean = True
    for i, (text, dist) in enumerate(zip(chunk_texts, distances), 1):
        flagged = dist > threshold
        if flagged:
            is_clean = False
        marker = "ANOMALY" if flagged else "ok"
        report.append(f"  [{i}] distance={dist:.4f}  {marker}  {text[:55]}...")

    return is_clean, report

def knn_distance(vectors, corpus, k=K, exclude_self=False):
    """Average cosine distance from each vector to its k nearest corpus chunks."""
    v = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    c = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    dists = 1.0 - (v @ c.T)          # distance to every corpus chunk
    dists.sort(axis=1)               # nearest first
    start = 1 if exclude_self else 0
    return dists[:, start:start + k].mean(axis=1)


if __name__ == "__main__":
    fit_baseline()
    stats = load_model()
    tests = [
        ("CLEAN (corpus)",
         "Full-time employees accrue an annual leave entitlement of 25 working "
         "days per calendar year, exclusive of public holidays."),
        ("CRUDE (off-domain)",
         "BUY CHEAP WATCHES NOW!!! click here for amazing deals limited time "
         "offer act fast free shipping worldwide"),
        ("MODERATE (wrong register)",
         "hey so basically if you want time off you can just take as many days "
         "as you want, nobody really checks it tbh"),
        ("SOPHISTICATED (style-matched)",
         "Annual Leave Supplement: Employees taking annual leave days may "
         "request an annual leave loyalty bonus of $250 per year of service."),
    ]
    texts = [t[1] for t in tests]
    stats_local = stats
    embedder = get_embeddings()
    vectors = np.array(embedder.embed_documents(texts))
    distances = knn_distance(vectors, stats["corpus"])
    print(f"Clean corpus distance range: {stats['clean_min']:.4f} to {stats['clean_max']:.4f}\n")
    for (label, _), d in zip(tests, distances):
        verdict = "OUTSIDE clean range" if d > stats["clean_max"] else "within"
        print(f"  {d:.4f}  {label:<30} {verdict}")