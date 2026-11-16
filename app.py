"""
app.py: the RAG-Shield demo.

Run with:
    streamlit run app.py

What it shows
  - Ask the policy assistant a question and see what each check decided.
  - Add a poisoned passage to the corpus from the sidebar, ask again, and
    watch the checks respond.
  - Turn protection off to see what the assistant would have answered
    without RAG-Shield.
  - The benchmark results, read from benchmark_summary.json.

The app calls rag_system.ask_question(), the same function the command-line
prompt and the benchmark use.
"""

import json
import os
import uuid

import pandas as pd
import streamlit as st

POISONS_FILE = "poisons.json"
SUMMARY_FILE = "benchmark_summary.json"
DEMO_SOURCE = "demo_poison"          # marks passages added from this app
DEFAULT_QUESTION = "What is the maximum expense claim without director approval?"

ATTACKS = {
    "content_injection": "Contradiction: replaces a real rule with a false one",
    "embedding_manipulation": "Invented fact: adds a claim that contradicts nothing",
    "retrieval_hijacking": "Hijacking: copies the question so it ranks first",
}

CHECKS = [
    ("layer1", "Layer 1", "Do the passages contradict each other?"),
    ("layer2", "Layer 2", "Is a passage off-topic, or does it repeat the question?"),
    ("layer3", "Layer 3", "Is a passage written like a poisoned one?"),
    ("integrity", "Integrity", "Is every passage on the approved list?"),
]

st.set_page_config(page_title="RAG-Shield", layout="wide")


# ------------------------------------------------------------------ loading

@st.cache_resource(show_spinner="Loading the models and the document store...")
def load_system():
    """Load everything once. Streamlit reruns this script on every click."""
    import rag_system as rs
    return rs, rs.build_rag_system()


@st.cache_data
def load_scenarios():
    """Map each target question to its three ready-made poisons."""
    with open(POISONS_FILE, encoding="utf-8") as f:
        poisons = json.load(f)
    scenarios = {}
    for p in poisons:
        scenarios.setdefault(p["target_question"], {})[p["attack_type"]] = p["text"]
    return scenarios


# ------------------------------------------------------------------ corpus

def injected_passages(store):
    """Texts of the passages this app has added to the corpus."""
    found = store._collection.get(where={"source": DEMO_SOURCE}, include=["documents"])
    return found["documents"]


def add_poison(store, attack, text):
    from langchain_core.documents import Document
    store.add_documents(
        [Document(page_content=text,
                  metadata={"source": DEMO_SOURCE, "poisoned": True, "attack": attack})],
        ids=["demo-" + uuid.uuid4().hex[:8]],
    )


def remove_all_poison(store):
    ids = store._collection.get(where={"source": DEMO_SOURCE})["ids"]
    if ids:
        store.delete(ids=ids)
    return len(ids)


# ------------------------------------------------------------------ asking

def ask(rs, system, question, protected):
    if protected:
        return rs.ask_question(system, question, verbose=False)

    # Protection off: retrieve and answer with no checks at all.
    docs = system["retriever"].invoke(question)
    chunks = [d.page_content for d in docs]
    return {
        "unprotected": True,
        "question": question,
        "chunks": chunks,
        "answer": rs.generate_answer(system["llm"], question, chunks),
    }


def check_detail(result, key):
    """The explanation a check gave when it raised an alert."""
    if key == "layer1":
        return result["layer1_reason"]
    if key == "layer2":
        return "\n".join(result["layer2_report"])
    if key == "layer3":
        return "\n".join(result["layer3_report"])
    positions = ", ".join(str(i) for i in result["unapproved_chunks"])
    return f"Not on the approved list: passage {positions}"


def show_result(result, poison_texts):
    if result.get("unprotected"):
        st.warning("Protection is off. No checks were run on these passages.")
    else:
        columns = st.columns(len(CHECKS))
        for column, (key, name, what) in zip(columns, CHECKS):
            with column:
                st.markdown(f"**{name}**")
                st.caption(what)
                if result[f"{key}_ok"]:
                    st.success("Pass")
                else:
                    st.error("Alert")
                    with st.expander("Why"):
                        st.text(check_detail(result, key))
                st.caption(f"{result['seconds'][key] * 1000:.0f} ms")

    st.subheader("Answer")
    if result.get("blocked"):
        st.error(
            "Answer withheld. At least one check raised an alert, so the "
            "passages were not passed to the model that writes the answer."
        )
    else:
        st.write(result["answer"])

    st.subheader("Passages retrieved")
    for i, text in enumerate(result["chunks"], 1):
        with st.container(border=True):
            if text in poison_texts:
                st.markdown(f"**Passage {i}** :red[(the poison you added)]")
            else:
                st.markdown(f"**Passage {i}**")
            st.write(text)


# ------------------------------------------------------------------ benchmark tab

def percent(entry):
    return round(100 * entry["alerts"] / entry["total"], 1) if entry["total"] else None


def show_benchmark():
    if not os.path.exists(SUMMARY_FILE):
        st.info("No benchmark results yet. Run `python benchmark.py` to produce them.")
        return

    with open(SUMMARY_FILE, encoding="utf-8") as f:
        summary = json.load(f)

    names = {"layer1": "Layer 1", "layer2": "Layer 2", "layer3": "Layer 3",
             "integrity": "Integrity", "layers": "Layers 1-3 together",
             "any": "All four checks"}
    attack_names = {"content_injection": "Contradiction",
                    "embedding_manipulation": "Invented fact",
                    "retrieval_hijacking": "Hijacking"}

    st.caption(f"{summary['cases']} cases, read from {summary['results_file']}")

    st.subheader("Poisons detected (%)")
    st.caption("Counted over the poisons that reached the top three passages.")
    detection = pd.DataFrame({
        names[key]: {
            attack_names[attack]: percent(values[key])
            for attack, values in summary["detection"].items() if key in values
        }
        for key in names
    }).dropna(axis=1, how="all")
    st.dataframe(detection, width="stretch")
    chart = detection[[c for c in ("Layer 1", "Layer 2", "Layer 3", "Integrity")
                       if c in detection.columns]]
    try:
        st.bar_chart(chart, stack=False)
    except TypeError:                      # older Streamlit without stack=
        st.bar_chart(chart)

    left, right = st.columns(2)
    with left:
        st.subheader("False alarms on clean questions")
        false_alarms = pd.DataFrame([
            {"Check": names[key],
             "False alarms": f"{value['alerts']} of {value['total']}",
             "Rate (%)": percent(value)}
            for key, value in summary["false_alarms"].items() if key in names
        ])
        st.dataframe(false_alarms, hide_index=True, width="stretch")
    with right:
        st.subheader("Time per question (median)")
        labels = {"layer1": "Layer 1", "layer2": "Layer 2", "layer3": "Layer 3",
                  "integrity": "Integrity", "local checks": "Layers 2, 3 and integrity",
                  "all checks": "All four checks",
                  "answer generation": "Writing the answer"}
        times = pd.DataFrame([
            {"Step": labels.get(key, key), "Milliseconds": value}
            for key, value in summary["milliseconds_median"].items()
        ])
        st.dataframe(times, hide_index=True, width="stretch")

    st.caption(
        "Reading these numbers: every benchmark poison was added after the "
        "approved list was made, so the integrity check catches all of them by "
        "design. Layer 3 saw two thirds of these poisons during training; its "
        "score on the two subject areas it never trained on is in "
        "benchmark_summary.json."
    )


# ------------------------------------------------------------------ page

rs, system = load_system()
store = system["vectorstore"]
scenarios = load_scenarios()
questions = list(scenarios)

with st.sidebar:
    st.header("Poison the corpus")
    target = st.selectbox(
        "Question to attack",
        questions,
        index=questions.index(DEFAULT_QUESTION) if DEFAULT_QUESTION in questions else 0,
    )
    attack = st.radio("Kind of attack", list(ATTACKS), format_func=ATTACKS.get)
    st.caption("Passage that will be added")
    st.info(scenarios[target][attack])

    if st.button("Add poison", type="primary", width="stretch"):
        add_poison(store, attack, scenarios[target][attack])
        st.session_state.pop("result", None)
    if st.button("Remove all poison", width="stretch"):
        remove_all_poison(store)
        st.session_state.pop("result", None)

    st.divider()
    protected = st.toggle(
        "Protection on", value=True,
        help="Off: the assistant answers from whatever it retrieves, with no checks.",
    )

    poison_texts = injected_passages(store)
    approved = len(system["approved"])
    st.metric("Passages in the corpus", store._collection.count())
    if poison_texts:
        st.caption(f"{approved} approved, {len(poison_texts)} added as poison")
    else:
        st.caption(f"{approved} approved, no poison")

st.title("RAG-Shield")
st.caption(
    "A company policy assistant with four checks between the documents it "
    "retrieves and the answer it writes."
)

ask_tab, benchmark_tab = st.tabs(["Ask a question", "Benchmark results"])

with ask_tab:
    question = st.text_input("Question", value=target, key=f"question-{target}")
    if st.button("Ask", type="primary"):
        try:
            with st.spinner("Retrieving passages and running the checks..."):
                st.session_state["result"] = ask(rs, system, question.strip(), protected)
        except Exception as error:
            st.session_state.pop("result", None)
            st.error(
                f"The question could not be answered ({type(error).__name__}). "
                "This is usually Gemini being busy. Wait a moment and ask again."
            )

    if "result" in st.session_state:
        show_result(st.session_state["result"], set(poison_texts))
    else:
        st.info(
            "Ask the question above to see a normal answer. Then add a poison "
            "from the sidebar and ask again."
        )

with benchmark_tab:
    show_benchmark()
