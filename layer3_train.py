"""
layer3_train.py: trains and evaluates the Layer 3 classifier.

The classifier is DistilBERT, a small pre-trained language model, fine-tuned
to read one passage and output the probability that it is poisoned.

  Training set   layer3_train.json   four subject areas
  Test set       layer3_test.json    two subject areas never seen in training

The test reports two things separately, because both matter:
  - how many poisoned passages are caught       (detection rate)
  - how many legitimate passages are flagged    (false alarm rate)

Usage:
    python layer3_data.py      (once, to build the two json files)
    python layer3_train.py

Writes the trained model to ./layer3_model/ and the scores to
layer3_results.json.
"""

import json
import os
import random
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_MODEL = os.getenv("LAYER3_BASE_MODEL", "distilbert-base-uncased")
MODEL_DIR = "./layer3_model"
TRAIN_FILE = "layer3_train.json"
TEST_FILE = "layer3_test.json"
RESULTS_FILE = "layer3_results.json"

SEED = 42            # fixed so the run can be repeated
EPOCHS = 4           # passes over the training set
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
MAX_LENGTH = 128     # tokens; every passage in the data is shorter than this
THRESHOLD = 0.5      # probability above which a passage is flagged



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_examples(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run layer3_data.py first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def encode(tokenizer, texts):
    """Turn a list of passages into the number arrays the model reads."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )


def train(model, tokenizer, examples):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    for epoch in range(1, EPOCHS + 1):
        order = list(range(len(examples)))
        random.shuffle(order)
        total_loss = 0.0

        for start in range(0, len(order), BATCH_SIZE):
            batch = [examples[i] for i in order[start:start + BATCH_SIZE]]
            inputs = encode(tokenizer, [e["text"] for e in batch])
            labels = torch.tensor([e["label"] for e in batch])

            loss = model(**inputs, labels=labels).loss   # how wrong the model was
            loss.backward()                              # work out how to adjust
            optimizer.step()                             # adjust the model
            optimizer.zero_grad()

            total_loss += loss.item() * len(batch)

        print(f"  epoch {epoch}/{EPOCHS}  average loss {total_loss / len(examples):.4f}")


def poison_probabilities(model, tokenizer, texts):
    """Probability (0 to 1) that each passage is poisoned."""
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(texts), 32):
            inputs = encode(tokenizer, texts[start:start + 32])
            logits = model(**inputs).logits
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].tolist())
    return probabilities


def evaluate(model, tokenizer, examples):
    probabilities = poison_probabilities(model, tokenizer, [e["text"] for e in examples])

    flagged = {}
    total = {}
    for example, p in zip(examples, probabilities):
        kind = example["kind"]
        total[kind] = total.get(kind, 0) + 1
        flagged[kind] = flagged.get(kind, 0) + (1 if p > THRESHOLD else 0)

    # Work out which kinds are legitimate and which are poisoned from the labels.
    legitimate_kinds = sorted({e["kind"] for e in examples if e["label"] == 0})
    poison_kinds = sorted({e["kind"] for e in examples if e["label"] == 1})

    print("\nLegitimate passages (should NOT be flagged)")
    for kind in legitimate_kinds:
        print(f"  {kind:<30} flagged {flagged[kind]:>3}/{total[kind]:<3} "
              f"false alarm rate {flagged[kind] / total[kind]:.1%}")

    print("\nPoisoned passages (should be flagged)")
    for kind in poison_kinds:
        print(f"  {kind:<30} caught  {flagged[kind]:>3}/{total[kind]:<3} "
              f"detection rate  {flagged[kind] / total[kind]:.1%}")

    legit_total = sum(total[k] for k in legitimate_kinds)
    legit_flagged = sum(flagged[k] for k in legitimate_kinds)
    poison_total = sum(total[k] for k in poison_kinds)
    poison_flagged = sum(flagged[k] for k in poison_kinds)

    detection_rate = poison_flagged / poison_total if poison_total else 0.0
    false_alarm_rate = legit_flagged / legit_total if legit_total else 0.0

    print(f"\nOverall detection rate    {detection_rate:.1%}  ({poison_flagged}/{poison_total})")
    print(f"Overall false alarm rate  {false_alarm_rate:.1%}  ({legit_flagged}/{legit_total})")

    return {
        "per_kind": {k: {"flagged": flagged[k], "total": total[k]} for k in total},
        "detection_rate": detection_rate,
        "false_alarm_rate": false_alarm_rate,
    }


def main():
    set_seed(SEED)
    train_examples = load_examples(TRAIN_FILE)
    test_examples = load_examples(TEST_FILE)

    print(f"Base model: {BASE_MODEL}")
    print(f"Training on {len(train_examples)} examples, testing on {len(test_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    started = time.perf_counter()
    train(model, tokenizer, train_examples)
    minutes = (time.perf_counter() - started) / 60
    print(f"Training finished in {minutes:.1f} minutes")

    print("\n=== Results on the held-out test set ===")
    results = evaluate(model, tokenizer, test_examples)

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    results["settings"] = {
        "base_model": BASE_MODEL,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "threshold": THRESHOLD,
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nModel saved to {MODEL_DIR}/")
    print(f"Scores saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()