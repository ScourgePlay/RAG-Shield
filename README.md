# RAG-Shield

RAG-Shield detects poisoned documents in a retrieval-augmented generation (RAG) system. It sits between the passages a policy assistant retrieves and the answer it writes. If any check raises an alert, no answer is generated.

The checks treat the language model as a black box. They need no access to model internals and no retraining of the answering model.

This is a final-year project. The corpus, the test questions and the poisoned passages are synthetic and were generated with an AI assistant (Claude).

## How it works

For every question the system:

1. Retrieves the three most similar passages from the vector store.
2. Runs four checks on those passages.
3. Writes an answer only if every check passes.

| Check | File | What it looks for | Calls Gemini |
|---|---|---|---|
| Layer 1 | `defense.py` | Passages that contradict each other | Yes |
| Layer 2 | `defense2b.py` | A passage far from the corpus, or one that repeats the question | No |
| Layer 3 | `defense3.py` | A passage written like a poisoned one (fine-tuned DistilBERT) | No |
| Integrity | `integrity.py` | A passage that is not on the approved list | No |

## The three attacks

| Attack | What the poison does |
|---|---|
| Content injection | States a false rule that contradicts a real one |
| Embedding manipulation | Invents a claim that contradicts nothing, using the topic's vocabulary |
| Retrieval hijacking | Copies the target question into the poison so it ranks first |

## Project files

| File | Purpose |
|---|---|
| `rag_system.py` | The pipeline and the command-line prompt |
| `app.py` | The Streamlit demo |
| `attack.py` | Adds a single poison to the store by hand |
| `benchmark.py` | Runs every clean question and every poison through the pipeline |
| `make_figures.py` | Turns the benchmark results into charts and tables |
| `reset.py` | Rebuilds the clean corpus |
| `count.py` | Prints how many passages are in the store |
| `layer3_data.py`, `layer3_train.py` | Build the Layer 3 dataset and train the classifier |
| `defense2.py` | Isolation Forest attempt for Layer 2, kept as a negative result |
| `generate_corpus.py`, `corpus_data_a.py`, `corpus_data_b.py` | Generate the corpus, questions and poisons |
| `documents/` | 18 policy documents (107 passages) |
| `queries.json` | 108 test questions with correct answers |
| `poisons.json` | 324 poisoned passages, 108 per attack |
| `benchmark_results.csv`, `benchmark_summary.json` | Saved benchmark results |

These are created locally and are not in the repository: `chroma_db/`, `approved_chunks.json`, `knn_model.joblib`, `layer3_model/`, `.env`.

## Setup on a fresh copy

Requires Python 3.12 and a free Gemini API key from Google AI Studio.

```bash
python -m venv .rag-venv
source .rag-venv/bin/activate
pip install -r requirements.txt
```

Create a file named `.env` containing:

```
GOOGLE_API_KEY=your_key_here
```

Then build the local files, in this order:

```bash
python reset.py           # builds the vector store and the approved list
python defense2b.py       # fits the Layer 2 baseline
python layer3_data.py     # builds the Layer 3 training and test sets
python layer3_train.py    # trains the classifier (a few minutes on a laptop)
```

The first run downloads two models from Hugging Face: `all-MiniLM-L6-v2` for embeddings and `distilbert-base-uncased` for Layer 3.

## Running it

| Task | Command |
|---|---|
| Demo in the browser | `streamlit run app.py` |
| Command-line prompt | `python rag_system.py` |
| Rebuild a clean corpus | `python reset.py` |
| Trial benchmark (18 cases) | `python benchmark.py --quick` |
| Full benchmark (432 cases, about 30 minutes) | `python benchmark.py` |
| Print the result tables again | `python benchmark.py --summary` |
| List false alarms and misses | `python benchmark.py --details` |
| Charts and tables for the report | `python make_figures.py` |
| Layer 3 stress test | `python defense3.py` |

The demo, the benchmark and `attack.py` all use the same vector store. Run one at a time, and run `python reset.py` before a benchmark or a demo.

The full benchmark makes about 450 Gemini calls. It saves after every case, so it can be stopped with Ctrl+C and continued by running the same command again.

## Results

Full benchmark, 16 November 2026: 108 clean questions and 324 poisons. Detection is counted over the poisons that reached the top three passages.

| | Contradiction | Invented fact | Hijacking | False alarms |
|---|---|---|---|---|
| Layer 1 | 98.1% | 2.4% | 99.1% | 4.6% |
| Layer 2 | 0% | 0% | 100% | 0% |
| Layer 3 | 100% | 98.8% | 100% | 0% |
| Integrity | 100% | 100% | 100% | 0% |
| Layers 1 to 3 together | 100% | 98.8% | 100% | 4.6% |
| All four checks | 100% | 100% | 100% | 4.6% |

Median time per question: Layer 1 872 ms, Layer 2 26 ms, Layer 3 64 ms, integrity under 1 ms. Writing an answer with no checks takes about 1,140 ms.

## Known limits

- The integrity check catches every benchmark poison by design, because each one was added after the approved list was made. It does not catch a poison that enters through normal approval.
- Layer 3 was trained on poisons from four of the six subject areas, written in one style. On the two areas it never saw it caught 97 of 98 poisons with no false alarms. On hand-written passages it was weaker: it flagged 3 of 6 legitimate new rules and all 3 genuine policy changes.
- Layer 3 judges how a passage is written. It cannot tell whether a claim is true.
- Layer 2's question-echo signal is avoided by an attacker who rewords the question.
- Layer 1 produced all five false alarms and adds about 0.9 seconds per question.
- The corpus is small (107 passages) and synthetic, and the benchmark is a single run.
