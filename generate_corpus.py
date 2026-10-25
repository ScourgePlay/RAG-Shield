"""
generate_corpus.py — builds the RAG-Shield evaluation corpus.

Single source of truth: corpus_data_a.py + corpus_data_b.py.
Running this script regenerates three things from that one source, so the
corpus text, the ground-truth answers and the poison variants can never
drift apart:

  documents/<doc_id>.txt   the clean corpus the RAG system ingests
  queries.json             evaluation questions + ground-truth answers
  poisons.json             poisoned documents, labelled by attack type

Attack types generated:
  content_injection        directly contradicts a claim in the corpus
                           (Layer 1 consistency checker SHOULD catch this)
  embedding_manipulation   false claim on the same topic that contradicts
                           nothing in the corpus, saturated with the query's
                           vocabulary
                           (Layer 1 is EXPECTED to miss this — it is the
                            documented blind spot motivating Layers 2 and 3)
  retrieval_hijacking      the target question repeated verbatim ahead of a
                           contradicting claim, to maximise similarity and
                           capture the top retrieval slot

Usage:
    python generate_corpus.py
"""

import json
import os
import shutil

from corpus_data_a import DOCUMENTS_A
from corpus_data_b import DOCUMENTS_B

DOCUMENTS = DOCUMENTS_A + DOCUMENTS_B

DOCS_DIR = "documents"
QUERIES_FILE = "queries.json"
POISONS_FILE = "poisons.json"


def write_corpus():
    """Write one .txt file per policy document."""
    if os.path.exists(DOCS_DIR):
        shutil.rmtree(DOCS_DIR)
    os.makedirs(DOCS_DIR)

    for doc in DOCUMENTS:
        lines = [f"SCRG Corp — {doc['title']}", ""]
        for claim in doc["claims"]:
            lines.append(claim["text"])
            lines.append("")  # blank line → chunk boundary
        path = os.path.join(DOCS_DIR, f"{doc['id']}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).strip() + "\n")

    return len(DOCUMENTS)


def build_queries():
    """One evaluation query per claim, with ground truth."""
    queries = []
    for doc in DOCUMENTS:
        for claim in doc["claims"]:
            queries.append({
                "query_id": claim["id"],
                "domain": doc["domain"],
                "source_document": doc["id"],
                "question": claim["question"],
                "ground_truth": claim["answer"],
            })
    return queries


def build_poisons():
    """Three poison variants per claim, each labelled with its attack type."""
    poisons = []
    for doc in DOCUMENTS:
        for claim in doc["claims"]:
            base = {
                "target_claim_id": claim["id"],
                "domain": doc["domain"],
                "source_document": doc["id"],
                "target_question": claim["question"],
                "ground_truth": claim["answer"],
                "poisoned": True,
            }

            poisons.append({
                **base,
                "poison_id": f"{claim['id']}__content_injection",
                "attack_type": "content_injection",
                "contradicts_corpus": True,
                "text": claim["poison_contradiction"],
            })

            poisons.append({
                **base,
                "poison_id": f"{claim['id']}__embedding_manipulation",
                "attack_type": "embedding_manipulation",
                "contradicts_corpus": False,
                "text": claim["poison_noncontradicting"],
            })

            poisons.append({
                **base,
                "poison_id": f"{claim['id']}__retrieval_hijacking",
                "attack_type": "retrieval_hijacking",
                "contradicts_corpus": True,
                "text": (
                    f"{claim['question']} {claim['question']} "
                    f"{claim['poison_contradiction']}"
                ),
            })

    return poisons


def main():
    n_docs = write_corpus()
    queries = build_queries()
    poisons = build_poisons()

    with open(QUERIES_FILE, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)

    with open(POISONS_FILE, "w", encoding="utf-8") as f:
        json.dump(poisons, f, indent=2, ensure_ascii=False)

    # Summary
    domains = {}
    for doc in DOCUMENTS:
        domains[doc["domain"]] = domains.get(doc["domain"], 0) + 1

    by_type = {}
    for p in poisons:
        by_type[p["attack_type"]] = by_type.get(p["attack_type"], 0) + 1

    print("Corpus generated.")
    print(f"  Documents written to {DOCS_DIR}/ : {n_docs}")
    for d, c in sorted(domains.items()):
        print(f"      {d:<16} {c} documents")
    print(f"  Evaluation queries ({QUERIES_FILE}) : {len(queries)}")
    print(f"  Poison variants    ({POISONS_FILE}) : {len(poisons)}")
    for t, c in sorted(by_type.items()):
        print(f"      {t:<24} {c}")
    print()
    print("Next: delete chroma_db and rebuild so the new corpus is ingested.")


if __name__ == "__main__":
    main()
