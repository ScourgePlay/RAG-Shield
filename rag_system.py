"""
RAG-Shield pipeline: retrieve -> check -> answer.

For every question the system:
  1. retrieves the top passages from the vector store
  2. runs every check on those passages
       Layer 1    contradiction between passages        (defense.py)
       Layer 2    corpus distance and question echo     (defense2b.py)
       Layer 3    classifier: written like a poison?    (defense3.py)
       Integrity  passage is in the approved baseline   (integrity.py)
  3. generates an answer ONLY if every check passes

If any check raises an alert, no answer is generated, so poisoned content
never reaches the language model that writes the answer.
"""

import logging
import os
import time

try:
    import readline  # noqa: F401  (makes arrow keys and backspace work at the prompt)
except ImportError:
    pass

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from defense import check_consistency
from defense2b import load_model as load_layer2, check_anomaly, get_embeddings
from defense3 import load_classifier, check_poison_style
from integrity import save_baseline, load_baseline, check_integrity

load_dotenv()

CHROMA_PATH = "./chroma_db"
DOCS_DIR = "documents"
LLM_MODEL = "gemini-flash-lite-latest"
TOP_K = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("rag_system.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Layer 1 auditor: a separate LLM instance from the one that writes answers.
auditor_llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

# Layer 2: nearest-neighbour baseline and the embedding model, loaded once.
layer2_stats = load_layer2()
layer2_embedder = get_embeddings()

# Layer 3: the fine-tuned DistilBERT classifier, loaded once.
layer3_classifier = load_classifier()

ANSWER_PROMPT = (
    "Answer the question using only the context below. If the context does "
    "not contain the answer, say that you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def build_rag_system():
    """Load the vector store (or build it once) and return what a question needs."""
    # The same embedding model is used for the store and for Layer 2.
    embeddings = layer2_embedder

    vectorstore = None
    if os.path.exists(CHROMA_PATH):
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
        )
        stored = vectorstore._collection.count()
        if stored == 0:
            # Any script that opens the store creates an empty folder. An
            # empty folder is not a built corpus, so build it now.
            logger.info("Vector store folder exists but holds 0 passages, rebuilding.")
            vectorstore = None
        else:
            logger.info("Existing vector store found with %d passages, loading it.", stored)

    if vectorstore is None:
        logger.info("Building the vector store from %s/", DOCS_DIR)
        loader = DirectoryLoader(
            DOCS_DIR,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH,
        )
        # Everything in the store right now came from the source documents,
        # so this is the moment to record the approved baseline.
        saved = save_baseline([c.page_content for c in chunks])
        logger.info(
            "Built store with %d chunks; %d fingerprints saved as the approved baseline.",
            len(chunks), saved,
        )

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    return {
        "vectorstore": vectorstore,
        "retriever": vectorstore.as_retriever(search_kwargs={"k": TOP_K}),
        "llm": llm,
        "approved": load_baseline(),
    }


def _response_text(response):
    """Return the text of an LLM response whether it arrives as a string or as parts."""
    content = response.content
    if isinstance(content, str):
        return content.strip()
    parts = []
    for part in content:
        parts.append(part.get("text", "") if isinstance(part, dict) else str(part))
    return "".join(parts).strip()


def generate_answer(llm, question, chunk_texts):
    """Ask the LLM to answer from the retrieved passages only."""
    prompt = ANSWER_PROMPT.format(
        context="\n\n".join(chunk_texts),
        question=question,
    )
    return _response_text(llm.invoke(prompt))


def ask_question(system, question, verbose=True, generate=True, use_layer1=True):
    """Retrieve, run every check, and answer only if all checks pass.

    Returns a dictionary of results so the benchmark and the web app can use
    the same function as the interactive prompt.

    generate=False    run the checks but do not write an answer (benchmark)
    use_layer1=False  skip Layer 1, the only check that calls Gemini (benchmark)
    """
    # 1. Retrieve
    docs = system["retriever"].invoke(question)
    chunk_texts = [d.page_content for d in docs]

    # 2. Check (each check is timed so the benchmark can report latency)
    t0 = time.perf_counter()
    if use_layer1:
        layer1_ok, layer1_reason = check_consistency(chunk_texts, auditor_llm)
    else:
        layer1_ok, layer1_reason = True, "skipped"
    t1 = time.perf_counter()
    layer2_ok, layer2_report = check_anomaly(
        chunk_texts, layer2_stats, embedder=layer2_embedder, question=question
    )
    t2 = time.perf_counter()
    layer3_ok, layer3_report = check_poison_style(chunk_texts, layer3_classifier)
    t3 = time.perf_counter()
    integrity_ok, unapproved = check_integrity(chunk_texts, system["approved"])
    t4 = time.perf_counter()

    # 3. Answer only if nothing was flagged
    blocked = not (layer1_ok and layer2_ok and layer3_ok and integrity_ok)
    if blocked or not generate:
        answer = None
    else:
        answer = generate_answer(system["llm"], question, chunk_texts)

    result = {
        "question": question,
        "chunks": chunk_texts,
        "layer1_ran": use_layer1,
        "layer1_ok": layer1_ok,
        "layer1_reason": layer1_reason,
        "layer2_ok": layer2_ok,
        "layer2_report": layer2_report,
        "layer3_ok": layer3_ok,
        "layer3_report": layer3_report,
        "integrity_ok": integrity_ok,
        "unapproved_chunks": unapproved,
        "blocked": blocked,
        "answer": answer,
        "seconds": {
            "layer1": t1 - t0,
            "layer2": t2 - t1,
            "layer3": t3 - t2,
            "integrity": t4 - t3,
        },
    }

    if verbose:
        print_result(result)
    return result


def print_result(r):
    """Show one result at the interactive prompt."""
    def status(ok):
        return "pass" if ok else "ALERT"

    print("\nChecks:")
    print(f"  Layer 1 (contradiction)    {status(r['layer1_ok'])}")
    print(f"  Layer 2 (distance, echo)   {status(r['layer2_ok'])}")
    print(f"  Layer 3 (classifier)       {status(r['layer3_ok'])}")
    print(f"  Integrity (approved list)  {status(r['integrity_ok'])}")

    if not r["layer1_ok"]:
        print("\n[!] Layer 1: retrieved passages contradict each other.")
        print(r["layer1_reason"])

    if not r["layer2_ok"]:
        print("\n[!] Layer 2: a retrieved passage is statistically unusual.")
        print("\n".join(r["layer2_report"]))

    if not r["layer3_ok"]:
        print("\n[!] Layer 3: a retrieved passage is written like a poisoned one.")
        print("\n".join(r["layer3_report"]))

    if not r["integrity_ok"]:
        positions = ", ".join(str(i) for i in r["unapproved_chunks"])
        print(f"\n[!] Integrity: passage(s) not in the approved corpus: {positions}")

    if r["blocked"]:
        print("\nAnswer withheld: at least one check raised an alert.")
    elif r["answer"] is not None:
        print("\nAnswer:", r["answer"])

    print("\nPassages retrieved:")
    for i, text in enumerate(r["chunks"], 1):
        print(f"  [{i}] {text[:80].strip()}...")


if __name__ == "__main__":
    try:
        system = build_rag_system()
    except Exception:
        logger.exception("RAG system build failed")
        raise

    print("\n--- RAG-Shield ready. Type a question, or 'quit' to exit. ---")
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if not question:
            continue
        try:
            ask_question(system, question)
        except Exception as e:
            print(f"\nThat question failed ({type(e).__name__}). Wait a moment and ask again.")