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


def poison_examples():
    with open(POISONS_FILE, encoding="utf-8") as f:
        poisons = json.load(f)
    return [
        {
            "text": p["text"],
            "label": 1,
            "kind": p["attack_type"],
            "domain": p["domain"],
        }
        for p in poisons
    ]


def summarise(name, examples):
    kinds = Counter(e["kind"] for e in examples)
    labels = Counter(e["label"] for e in examples)
    print(f"\n{name}: {len(examples)} examples "
          f"({labels[0]} legitimate, {labels[1]} poisoned)")
    for kind in ("clean", "benign_notice", "content_injection",
                 "embedding_manipulation", "retrieval_hijacking"):
        print(f"    {kind:<24} {kinds[kind]}")


def main():
    examples = clean_examples() + benign_notice_examples() + poison_examples()

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
