import json
import os
import re

import joblib
import numpy as np
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "./chroma_db"
MODEL_PATH = "./knn_model.joblib"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K = 3

# Question echo threshold. Measured on this corpus: across the 108 clean test
# questions the highest echo against any clean passage was 0.43, while every
# retrieval-hijacking poison scored 1.00. 0.70 sits between the two.
ECHO_THRESHOLD = 0.70
ECHO_NGRAM = 3


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def load_corpus():
    """Return (embeddings, texts) for every passage currently in the store."""
    store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings(),
    )
    data = store._collection.get(include=["embeddings", "documents"])
    return np.array(data["embeddings"]), list(data["documents"])


# ---------------------------------------------------------------- signal 1

def knn_distance(vectors, corpus, k=K, exclude_self=False):
    """Average cosine distance from each vector to its k nearest corpus passages."""
    v = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    c = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    dists = 1.0 - (v @ c.T)          # distance to every corpus passage
    dists.sort(axis=1)               # nearest first
    start = 1 if exclude_self else 0
    return dists[:, start:start + k].mean(axis=1)


# ---------------------------------------------------------------- signal 2

def _words(text):
    return re.findall(r"[a-z0-9$]+", text.lower())


def _ngrams(words, n):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def question_echo(question, passage, n=ECHO_NGRAM):
    """Share of the question's n-word sequences found word for word in the passage.

    0.0 means the passage shares no n-word sequence with the question.
    1.0 means the whole question appears inside the passage.
    """
    q = _ngrams(_words(question), n)
    if not q:
        return 0.0
    return len(q & _ngrams(_words(passage), n)) / len(q)


# ---------------------------------------------------------------- baseline

def fit_baseline():
    """Record the clean corpus and its distance range. Run on a CLEAN store only."""
    X, _ = load_corpus()
    # exclude_self: each clean passage's nearest match is itself (distance 0),
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
    print(f"Fitted on {X.shape[0]} clean passages, k={K}")
    print(f"  distance min    {stats['clean_min']:.4f}")
    print(f"  distance max    {stats['clean_max']:.4f}")
    print(f"  distance mean   {stats['clean_mean']:.4f}")
    return stats


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No model at {MODEL_PATH}. Run fit_baseline() first.")
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------- the check

def check_anomaly(chunk_texts, stats, embedder=None, threshold=None, question=None):
    """Run both Layer 2 signals on the retrieved passages.

    Returns (is_clean, report_lines). A passage is flagged if it is further
    from the corpus than any clean passage, or if it repeats the question.
    The question-echo signal only runs when the question is passed in.
    """
    if threshold is None:
        threshold = stats["clean_max"]

    embedder = embedder or get_embeddings()
    vectors = np.array(embedder.embed_documents(chunk_texts))
    distances = knn_distance(vectors, stats["corpus"])

    report = []
    is_clean = True
    for i, (text, dist) in enumerate(zip(chunk_texts, distances), 1):
        echo = question_echo(question, text) if question else 0.0

        reasons = []
        if dist > threshold:
            reasons.append("far from corpus")
        if echo > ECHO_THRESHOLD:
            reasons.append("repeats the question")

        if reasons:
            is_clean = False
            marker = "ANOMALY (" + ", ".join(reasons) + ")"
        else:
            marker = "ok"
        report.append(
            f"  [{i}] distance={dist:.4f}  echo={echo:.2f}  {marker}  {text[:45]}..."
        )

    return is_clean, report


# ---------------------------------------------------------------- evidence

def measure_echo(queries_path="queries.json", poisons_path="poisons.json"):
    """Print the question-echo numbers for the report. Run on a CLEAN store."""
    _, passages = load_corpus()
    with open(queries_path, encoding="utf-8") as f:
        queries = json.load(f)
    with open(poisons_path, encoding="utf-8") as f:
        poisons = json.load(f)

    # Worst case for a clean question: its highest echo against ANY clean passage.
    clean = sorted(
        max(question_echo(q["question"], p) for p in passages) for q in queries
    )
    false_alarms = sum(score > ECHO_THRESHOLD for score in clean)

    print(f"\nQuestion echo, threshold {ECHO_THRESHOLD:.2f}")
    print(f"  Clean questions: {len(clean)}")
    print(f"    highest echo {clean[-1]:.2f}, median {clean[len(clean) // 2]:.2f}")
    print(f"    false alarms {false_alarms}/{len(clean)}")

    for attack in ("content_injection", "embedding_manipulation", "retrieval_hijacking"):
        scores = [
            question_echo(p["target_question"], p["text"])
            for p in poisons if p["attack_type"] == attack
        ]
        flagged = sum(score > ECHO_THRESHOLD for score in scores)
        print(f"  {attack:<24} flagged {flagged}/{len(scores)}")


if __name__ == "__main__":
    fit_baseline()
    measure_echo()