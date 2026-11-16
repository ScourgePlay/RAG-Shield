"""
benchmark.py: measures RAG-Shield on the full test set.

What it runs
  Clean questions   every question in queries.json against the clean corpus.
                    Any alert here is a FALSE ALARM.
  Poisons           every entry in poisons.json, one at a time: the poison is
                    added to the store, its target question is asked, the
                    checks run, and the poison is removed again.
                    An alert here is a DETECTION.

Every case goes through rag_system.ask_question(), the same function the
interactive prompt and the web app use, so what is measured is what is demoed.
No answers are generated during the benchmark except for a small sample used
to time answer generation.

Layer 1 is the only check that calls Gemini (one call per case). The run is
paced to stay under the free-tier rate limit, saves each result as it goes,
and can be stopped with Ctrl+C and resumed by running the same command again.

Usage
    python benchmark.py --quick        small trial run, about two minutes
    python benchmark.py                full run (stop and resume at any time)
    python benchmark.py --no-layer1    full run without Layer 1 (no Gemini calls,
                                       saved to a separate results file)
    python benchmark.py --summary      print the result tables again
    python benchmark.py --details      list the false alarms and missed poisons
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time

QUERIES_FILE = "queries.json"
POISONS_FILE = "poisons.json"
RESULTS_FILE = "benchmark_results.csv"
QUICK_FILE = "benchmark_quick.csv"
NO_LAYER1_FILE = "benchmark_results_no_layer1.csv"
SUMMARY_FILE = "benchmark_summary.json"

# Layer 3 was trained on poisons from the other four domains, so its scores
# are also reported on these two alone.
HELD_OUT_DOMAINS = {"Travel", "Data Governance"}

SECONDS_BETWEEN_CASES = 4.0   # pacing when Layer 1 is on (about 15 Gemini calls a minute)
GENERATION_SAMPLE = 20        # clean questions for which answer generation is timed
MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 30
POISON_ID = "benchmark-poison"

CHECKS = ["layer1", "layer2", "layer3", "integrity"]
ATTACKS = ["content_injection", "embedding_manipulation", "retrieval_hijacking"]

FIELDS = [
    "case_id", "kind", "domain", "question",
    "poison_retrieved", "poison_rank",
    "layer1_ran", "layer1_alert", "layer2_alert", "layer3_alert", "integrity_alert",
    "any_alert",
    "t_layer1", "t_layer2", "t_layer3", "t_integrity", "t_generate",
]


# ------------------------------------------------------------------ cases

def load_cases(quick):
    with open(QUERIES_FILE, encoding="utf-8") as f:
        queries = json.load(f)
    with open(POISONS_FILE, encoding="utf-8") as f:
        poisons = json.load(f)

    if quick:
        # One clean question per subject area, and four poisons of each type.
        queries = queries[::18]
        keep = {q["query_id"] for q in queries[:4]}
        poisons = [p for p in poisons if p["target_claim_id"] in keep]

    cases = []
    for q in queries:
        cases.append({
            "case_id": "clean__" + q["query_id"],
            "kind": "clean",
            "domain": q["domain"],
            "question": q["question"],
            "poison_text": "",
        })
    for p in poisons:
        cases.append({
            "case_id": p["poison_id"],
            "kind": p["attack_type"],
            "domain": p["domain"],
            "question": p["target_question"],
            "poison_text": p["text"],
        })
    return cases


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_row(path, row):
    is_new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ------------------------------------------------------------------ running

def run_case(rs, system, case, use_layer1, time_generation):
    """Run one case through the real pipeline and return its result row."""
    from langchain_core.documents import Document

    store = system["vectorstore"]
    added = False
    try:
        if case["poison_text"]:
            store.add_documents(
                [Document(
                    page_content=case["poison_text"],
                    metadata={"source": "benchmark_poison", "poisoned": True},
                )],
                ids=[POISON_ID],
            )
            added = True

        result = rs.ask_question(
            system, case["question"],
            verbose=False, generate=False, use_layer1=use_layer1,
        )

        t_generate = ""
        if time_generation:
            started = time.perf_counter()
            rs.generate_answer(system["llm"], case["question"], result["chunks"])
            t_generate = round(time.perf_counter() - started, 4)
    finally:
        if added:
            store.delete(ids=[POISON_ID])     # always leave the store clean

    rank = 0
    if case["poison_text"]:
        for i, text in enumerate(result["chunks"], 1):
            if text.strip() == case["poison_text"].strip():
                rank = i
                break

    alerts = {
        "layer1": not result["layer1_ok"],
        "layer2": not result["layer2_ok"],
        "layer3": not result["layer3_ok"],
        "integrity": not result["integrity_ok"],
    }
    return {
        "case_id": case["case_id"],
        "kind": case["kind"],
        "domain": case["domain"],
        "question": case["question"],
        "poison_retrieved": int(rank > 0),
        "poison_rank": rank,
        "layer1_ran": int(use_layer1),
        "layer1_alert": int(alerts["layer1"]) if use_layer1 else "",
        "layer2_alert": int(alerts["layer2"]),
        "layer3_alert": int(alerts["layer3"]),
        "integrity_alert": int(alerts["integrity"]),
        "any_alert": int(any(alerts.values())),
        "t_layer1": round(result["seconds"]["layer1"], 4) if use_layer1 else "",
        "t_layer2": round(result["seconds"]["layer2"], 4),
        "t_layer3": round(result["seconds"]["layer3"], 4),
        "t_integrity": round(result["seconds"]["integrity"], 6),
        "t_generate": t_generate,
    }


def run_case_with_retries(rs, system, case, use_layer1, time_generation):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return run_case(rs, system, case, use_layer1, time_generation)
        except KeyboardInterrupt:
            raise
        except Exception as error:
            print(f"    attempt {attempt}/{MAX_RETRIES} failed: {type(error).__name__}")
            if attempt == MAX_RETRIES:
                break
            time.sleep(RETRY_WAIT_SECONDS)
    sys.exit(
        "\nStopped: a case kept failing (often a Gemini limit). Your progress "
        "is saved. Run the same command again later to continue."
    )


def run(quick, use_layer1):
    if quick:
        path = QUICK_FILE
    elif use_layer1:
        path = RESULTS_FILE
    else:
        path = NO_LAYER1_FILE
    if quick and os.path.exists(path):
        os.remove(path)                      # the trial run always starts fresh

    cases = load_cases(quick)
    done = {row["case_id"] for row in read_rows(path)}
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"{len(cases)} cases in total, {len(done)} already done, {len(todo)} to run.")
    if not todo:
        return path

    import rag_system as rs
    system = rs.build_rag_system()
    store = system["vectorstore"]

    # The benchmark must start from the approved, poison-free corpus.
    try:
        store.delete(ids=[POISON_ID])        # left over from an interrupted run
    except Exception:
        pass
    stored = store._collection.count()
    approved = len(system["approved"])
    if stored != approved:
        sys.exit(
            f"\nThe store holds {stored} passages but the approved baseline has "
            f"{approved}. Delete the chroma_db folder, run rag_system.py once "
            "to rebuild, then start the benchmark again."
        )

    timed_generation = 0
    try:
        for number, case in enumerate(todo, len(done) + 1):
            started = time.perf_counter()
            time_generation = (
                use_layer1 and case["kind"] == "clean" and timed_generation < GENERATION_SAMPLE
            )
            row = run_case_with_retries(rs, system, case, use_layer1, time_generation)
            if time_generation:
                timed_generation += 1
            append_row(path, row)

            flags = " ".join(
                name for name, key in (("L1", "layer1_alert"), ("L2", "layer2_alert"),
                                       ("L3", "layer3_alert"), ("INT", "integrity_alert"))
                if row[key] == 1
            ) or "none"
            print(f"[{number}/{len(cases)}] {case['kind']:<24} alerts: {flags}")

            if use_layer1:
                pace = SECONDS_BETWEEN_CASES * (2 if time_generation else 1)
                wait = pace - (time.perf_counter() - started)
                if wait > 0:
                    time.sleep(wait)
    except KeyboardInterrupt:
        print("\nStopped. Progress is saved; run the same command again to continue.")
        sys.exit(0)
    return path


# ------------------------------------------------------------------ summary

def _rate(rows, key):
    """(count, total) of rows where the 0/1 column `key` is 1, ignoring blanks."""
    values = [row[key] for row in rows if row[key] != ""]
    return sum(v == "1" for v in values), len(values)


def _layers_alert(row):
    """True if any of Layers 1-3 raised an alert (the integrity check is ignored)."""
    return any(row[f"{c}_alert"] == "1" for c in ("layer1", "layer2", "layer3"))


def _layers_rate(rows):
    return sum(_layers_alert(r) for r in rows), len(rows)


def _show(count, total, empty="no cases"):
    return f"{count:>3}/{total:<3} {count / total:6.1%}" if total else f"   {empty}"


def _times(rows, key):
    return [float(row[key]) for row in rows if row[key] != ""]


def summarise(path):
    rows = read_rows(path)
    if not rows:
        sys.exit(f"No results in {path} yet.")

    clean = [r for r in rows if r["kind"] == "clean"]
    summary = {"results_file": path, "cases": len(rows)}

    labels = {
        "layer1": "Layer 1  contradiction",
        "layer2": "Layer 2  distance, echo",
        "layer3": "Layer 3  classifier",
        "integrity": "Integrity check",
        "layers": "Layers 1-3 together",
        "any": "All four checks",
    }
    keys = {c: f"{c}_alert" for c in CHECKS}
    keys["any"] = "any_alert"

    print(f"\nResults from {path}: {len(rows)} cases")

    # 1. false alarms
    print(f"\n1. FALSE ALARMS on {len(clean)} clean questions (lower is better)")
    summary["false_alarms"] = {}
    for name in CHECKS + ["any"]:
        count, total = _rate(clean, keys[name])
        empty = "not run" if name == "layer1" and clean else "no cases"
        if name == "any":
            lc, lt = _layers_rate(clean)
            print(f"   {labels['layers']:<26} {_show(lc, lt)}")
            summary["false_alarms"]["layers"] = {"alerts": lc, "total": lt}
        print(f"   {labels[name]:<26} {_show(count, total, empty)}")
        summary["false_alarms"][name] = {"alerts": count, "total": total}

    # 2. detection per attack type
    # A poison that never reaches the top 3 passages cannot affect the answer
    # and gives the checks nothing to see, so detection is counted over the
    # poisons that WERE retrieved.
    print("\n2. DETECTION by attack type (higher is better)")
    print("   counted over the poisons that reached the top 3 passages")
    summary["detection"] = {}
    for attack in ATTACKS:
        cases = [r for r in rows if r["kind"] == attack]
        if not cases:
            continue
        reached = [r for r in cases if r["poison_retrieved"] == "1"]
        print(f"\n   {attack}: {len(cases)} poisons, {len(reached)} reached the top 3")
        summary["detection"][attack] = {"poisons": len(cases), "retrieved": len(reached)}
        for name in CHECKS + ["any"]:
            count, n = _rate(reached, keys[name])
            empty = "not run" if name == "layer1" and reached else "no cases"
            if name == "any":
                lc, lt = _layers_rate(reached)
                print(f"     {labels['layers']:<26} {_show(lc, lt)}")
                summary["detection"][attack]["layers"] = {"alerts": lc, "total": lt}
            print(f"     {labels[name]:<26} {_show(count, n, empty)}")
            summary["detection"][attack][name] = {"alerts": count, "total": n}

    # 3. Layer 3 on domains it never trained on
    held_out = [r for r in rows if r["domain"] in HELD_OUT_DOMAINS]
    if held_out:
        print("\n3. LAYER 3 on the two domains it never trained on")
        summary["layer3_held_out"] = {}
        count, total = _rate([r for r in held_out if r["kind"] == "clean"], "layer3_alert")
        print(f"   {'false alarms, clean':<26} {_show(count, total)}")
        summary["layer3_held_out"]["clean"] = {"alerts": count, "total": total}
        for attack in ATTACKS:
            reached = [r for r in held_out
                       if r["kind"] == attack and r["poison_retrieved"] == "1"]
            count, total = _rate(reached, "layer3_alert")
            print(f"   {attack:<26} {_show(count, total)}")
            summary["layer3_held_out"][attack] = {"alerts": count, "total": total}

    # 4. time added by each check
    print("\n4. TIME per question (milliseconds, median)")
    summary["milliseconds_median"] = {}
    for name, key in (("layer1", "t_layer1"), ("layer2", "t_layer2"),
                      ("layer3", "t_layer3"), ("integrity", "t_integrity"),
                      ("answer generation", "t_generate")):
        values = _times(rows, key)
        if values:
            median = statistics.median(values) * 1000
            label = labels.get(name, "Answer generation (sample)")
            print(f"   {label:<26} {median:9.1f} ms   ({len(values)} measurements)")
            summary["milliseconds_median"][name] = round(median, 2)
        else:
            print(f"   {labels.get(name, 'Answer generation (sample)'):<26}   not measured")

    def _total(row, columns):
        return sum(float(row[c]) for c in columns if row[c] != "")

    local = [_total(r, ("t_layer2", "t_layer3", "t_integrity")) for r in rows]
    everything = [_total(r, ("t_layer1", "t_layer2", "t_layer3", "t_integrity")) for r in rows]
    local_ms = statistics.median(local) * 1000
    all_ms = statistics.median(everything) * 1000
    print(f"   {'Layers 2, 3 + integrity':<26} {local_ms:9.1f} ms   (no Gemini call)")
    print(f"   {'All four checks':<26} {all_ms:9.1f} ms")
    summary["milliseconds_median"]["local checks"] = round(local_ms, 2)
    summary["milliseconds_median"]["all checks"] = round(all_ms, 2)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {SUMMARY_FILE}")


def details(path):
    """List the individual cases worth reading: false alarms and misses."""
    rows = read_rows(path)
    if not rows:
        sys.exit(f"No results in {path} yet.")

    names = (("L1", "layer1_alert"), ("L2", "layer2_alert"),
             ("L3", "layer3_alert"), ("INT", "integrity_alert"))

    print("\nCLEAN QUESTIONS THAT RAISED AN ALERT (false alarms)")
    found = False
    for r in rows:
        if r["kind"] == "clean" and r["any_alert"] == "1":
            found = True
            flags = " ".join(n for n, key in names if r[key] == "1")
            print(f"  [{flags}] {r['domain']}: {r['question']}")
    if not found:
        print("  none")

    print("\nRETRIEVED POISONS THAT LAYERS 1-3 ALL MISSED")
    found = False
    for r in rows:
        if r["kind"] != "clean" and r["poison_retrieved"] == "1" and not _layers_alert(r):
            found = True
            print(f"  {r['kind']}, rank {r['poison_rank']}, {r['domain']}: {r['question']}")
    if not found:
        print("  none")


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description="Benchmark RAG-Shield.")
    parser.add_argument("--quick", action="store_true", help="small trial run")
    parser.add_argument("--no-layer1", action="store_true", help="skip Layer 1 (no Gemini calls)")
    parser.add_argument("--summary", action="store_true", help="only print the result tables")
    parser.add_argument("--details", action="store_true",
                        help="list the false alarms and the missed poisons")
    args = parser.parse_args()

    if args.details:
        for path in (RESULTS_FILE, NO_LAYER1_FILE, QUICK_FILE):
            if os.path.exists(path):
                details(path)
                return
        sys.exit("No benchmark results found yet.")

    if args.summary:
        for path in (RESULTS_FILE, NO_LAYER1_FILE, QUICK_FILE):
            if os.path.exists(path):
                summarise(path)
                return
        sys.exit("No benchmark results found yet.")
        return

    path = run(quick=args.quick, use_layer1=not args.no_layer1)
    summarise(path)


if __name__ == "__main__":
    main()