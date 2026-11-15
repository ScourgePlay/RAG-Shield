"""
Integrity check: is every retrieved chunk part of the approved corpus?

When the corpus is built from the source documents, a fingerprint (SHA-256
hash) of every chunk is saved to approved_chunks.json. At question time, any
retrieved chunk whose fingerprint is not in that file was added to the vector
store after the approved build, and is flagged.

Limit: a legitimate new document is also flagged until the corpus is rebuilt
with it included. This check verifies where a chunk came from. It does not
judge whether the content is true.
"""

import hashlib
import json
import os

BASELINE_PATH = "./approved_chunks.json"


def fingerprint(text):
    """A fixed-length code that changes if even one character of the text changes."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def save_baseline(chunk_texts):
    """Record the fingerprints of every approved chunk. Called once, at build time."""
    hashes = sorted({fingerprint(t) for t in chunk_texts})
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(hashes, f)
    return len(hashes)


def load_baseline():
    if not os.path.exists(BASELINE_PATH):
        raise FileNotFoundError(
            f"No baseline at {BASELINE_PATH}. Delete the chroma_db folder and "
            "run rag_system.py once to rebuild the corpus and create it."
        )
    with open(BASELINE_PATH, encoding="utf-8") as f:
        return set(json.load(f))


def check_integrity(chunk_texts, approved):
    """Return (all_approved, positions of any chunks not in the baseline)."""
    unapproved = [
        i for i, text in enumerate(chunk_texts, 1)
        if fingerprint(text) not in approved
    ]
    return len(unapproved) == 0, unapproved
