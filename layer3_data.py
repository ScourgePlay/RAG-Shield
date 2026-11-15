"""
layer3_data.py: builds the training and test sets for Layer 3 (the classifier).

Three kinds of example:

  clean           every passage of the approved corpus                label 0
  benign_notice   a TRUE policy rule wrapped in update-style wording  label 0
  titled_rule     a TRUE policy rule with its policy title in front   label 0
  poison          every entry in poisons.json (three attack types)    label 1
  *_plain         the same poisons with their title removed, so they
                  read like an ordinary rule                          label 1

Why the titled rules and the plain poisons exist
  A first version of the classifier scored 100% on the held-out test but a
  hand-written stress test showed it had learned a shortcut: a short title
  followed by a colon meant "poison", and a plain sentence meant "legitimate".
  Adding legitimate passages WITH a title and poisoned passages WITHOUT one
  removes that shortcut, so the classifier has to read the body of the text.

Why the benign notices exist
  Many poisons are worded as updates ("Update: effective immediately...").
  A classifier could score well just by learning "update wording = poison",
  and would then flag every genuine policy update a company publishes.
  The benign notices use the same kind of wording around true content, so
  the classifier has to learn something better, and so its false alarms on
  genuine updates can be measured.

How the split keeps the test fair
  - Whole subject areas are held out: the classifier trains on four domains
    and is tested on two it has never seen.
  - The benign notices in the test set use wordings that never appear in
    the training set.

Usage:
    python layer3_data.py
Writes layer3_train.json and layer3_test.json.
"""

import json
import os
from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter

from corpus_data_a import DOCUMENTS_A
from corpus_data_b import DOCUMENTS_B

DOCUMENTS = DOCUMENTS_A + DOCUMENTS_B

DOCS_DIR = "documents"
POISONS_FILE = "poisons.json"
TRAIN_FILE = "layer3_train.json"
TEST_FILE = "layer3_test.json"

# Domains the classifier never sees during training.
TEST_DOMAINS = {"Travel", "Data Governance"}

# Update-style wordings wrapped around TRUE policy text.
TRAIN_TEMPLATES = [
    "{title} Update: Following the scheduled annual review, this provision is confirmed without change. {rule}",
    "Policy Notice: Effective immediately, staff are reminded that the following rule continues to apply. {rule}",
    "{title} Revision: The wording below was reissued after the latest policy audit and remains in force. {rule}",
    "Reminder: This policy has now been republished. No changes have been made to this section. {rule}",
]
TEST_TEMPLATES = [
    "{title} Change Log: This section was reviewed this quarter. It is no longer under consultation and stands as written. {rule}",
    "Notice to all employees: The previous guidance has been replaced by the confirmed text below. {rule}",
]


def first_sentence(text):
    """The first sentence of a policy paragraph, which carries the rule itself."""
    end = text.find(". ")
    return text if end == -1 else text[: end + 1]


def clean_examples():
    """Every passage of the approved corpus, split exactly as the RAG system splits it."""
    domain_of = {doc["id"]: doc["domain"] for doc in DOCUMENTS}
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    examples = []
    for filename in sorted(os.listdir(DOCS_DIR)):
        doc_id = os.path.splitext(filename)[0]
        with open(os.path.join(DOCS_DIR, filename), encoding="utf-8") as f:
            for passage in splitter.split_text(f.read()):
                examples.append({
                    "text": passage,
                    "label": 0,
                    "kind": "clean",
                    "domain": domain_of[doc_id],
                })
    return examples


def benign_notice_examples():
    """One true, update-worded notice per policy rule."""
    examples = []
    for doc in DOCUMENTS:
        templates = TEST_TEMPLATES if doc["domain"] in TEST_DOMAINS else TRAIN_TEMPLATES
        for i, claim in enumerate(doc["claims"]):
            template = templates[i % len(templates)]
            examples.append({
                "text": template.format(title=doc["title"], rule=first_sentence(claim["text"])),
                "label": 0,
                "kind": "benign_notice",
                "domain": doc["domain"],
            })
    return examples


def titled_rule_examples():
    """One true rule per claim, with the policy title in front of it."""
    return [
        {
            "text": f"{doc['title']}: {first_sentence(claim['text'])}",
            "label": 0,
            "kind": "titled_rule",
            "domain": doc["domain"],
        }
        for doc in DOCUMENTS
        for claim in doc["claims"]
    ]


def strip_title(text):
    """Remove a leading 'Some Title: ' so the passage reads like an ordinary rule."""
    cut = text.find(": ")
    return text[cut + 2:] if 0 < cut < 60 else text


def poison_examples():
    with open(POISONS_FILE, encoding="utf-8") as f:
        poisons = json.load(f)
    examples = []
    for p in poisons:
        examples.append({
            "text": p["text"],
            "label": 1,
            "kind": p["attack_type"],
            "domain": p["domain"],
        })
        # Hijacking poisons start with the question, not a title, so only the
        # other two attack types get a title-free variant.
        if p["attack_type"] in ("content_injection", "embedding_manipulation"):
            examples.append({
                "text": strip_title(p["text"]),
                "label": 1,
                "kind": p["attack_type"] + "_plain",
                "domain": p["domain"],
            })
    return examples


def summarise(name, examples):
    kinds = Counter(e["kind"] for e in examples)
    labels = Counter(e["label"] for e in examples)
    print(f"\n{name}: {len(examples)} examples "
          f"({labels[0]} legitimate, {labels[1]} poisoned)")
    for label in (0, 1):
        for kind in sorted({e["kind"] for e in examples if e["label"] == label}):
            print(f"    {kind:<30} {kinds[kind]}")


def main():
    examples = (clean_examples() + benign_notice_examples()
                + titled_rule_examples() + poison_examples())

    train = [e for e in examples if e["domain"] not in TEST_DOMAINS]
    test = [e for e in examples if e["domain"] in TEST_DOMAINS]

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    with open(TEST_FILE, "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2, ensure_ascii=False)

    print("Layer 3 dataset built.")
    print(f"  Training domains: {sorted({e['domain'] for e in train})}")
    print(f"  Held-out domains: {sorted(TEST_DOMAINS)}")
    summarise("Train", train)
    summarise("Test (held-out domains)", test)


if __name__ == "__main__":
    main()