"""
make_figures.py: turns the benchmark results into charts and tables for the report.

Reads   benchmark_summary.json   (written by benchmark.py)
Writes  figures/fig1_detection.png        poisons detected, by attack and check
        figures/fig2_false_alarms.png     false alarms on clean questions
        figures/fig3_time.png             time each step takes
        figures/fig4_attack_reach.png     how often each attack reaches the top 3
        figures/results_tables.md         the same numbers as tables

Usage:
    pip install matplotlib
    python make_figures.py

Every number comes from the summary file, so re-running the benchmark and
then this script keeps the report's figures in step with the data.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")            # write image files without opening a window
import matplotlib.pyplot as plt

SUMMARY_FILE = "benchmark_summary.json"
OUT_DIR = "figures"

# The false-alarm target from the project proposal, drawn as a dashed line on
# figure 2. Set to None to leave the line out.
FALSE_ALARM_TARGET = 5.0

CHECKS = [
    ("layer1", "Layer 1: contradiction", "#0072B2"),
    ("layer2", "Layer 2: distance and echo", "#E69F00"),
    ("layer3", "Layer 3: classifier", "#009E73"),
    ("integrity", "Integrity check", "#8C8C8C"),
]
ATTACKS = [
    ("content_injection", "Contradiction"),
    ("embedding_manipulation", "Invented fact"),
    ("retrieval_hijacking", "Hijacking"),
]


def percent(entry):
    return 100.0 * entry["alerts"] / entry["total"] if entry["total"] else 0.0


def tidy(ax):
    """Shared look: no top/right border, light horizontal guide lines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", path)


# ------------------------------------------------------------------ figures

def figure_detection(summary):
    detection = summary["detection"]
    attacks = [(key, label) for key, label in ATTACKS if key in detection]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    width = 0.2
    for i, (check, label, colour) in enumerate(CHECKS):
        values = [percent(detection[a][check]) for a, _ in attacks]
        positions = [x + (i - 1.5) * width for x in range(len(attacks))]
        bars = ax.bar(positions, values, width * 0.92, label=label, color=colour)
        for bar, value in zip(bars, values):
            text = f"{value:.0f}%" if value in (0.0, 100.0) else f"{value:.1f}%"
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, text,
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels([
        f"{label}\n{detection[key]['retrieved']} poisons retrieved"
        for key, label in attacks
    ])
    ax.set_ylim(0, 118)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Poisons detected (%)")
    ax.set_title("Poisons detected by each check, by type of attack", loc="left")
    ax.grid(axis="y", color="#E3E3E3", linewidth=0.8)
    ax.legend(ncol=4, frameon=False, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    tidy(ax)
    save(fig, "fig1_detection.png")


def figure_false_alarms(summary):
    false_alarms = summary["false_alarms"]
    rows = [(label.split(":")[0], false_alarms[key], colour)
            for key, label, colour in CHECKS if key in false_alarms]
    if "any" in false_alarms:
        rows.append(("All four\nchecks", false_alarms["any"], "#3B3B3B"))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    values = [percent(entry) for _, entry, _ in rows]
    bars = ax.bar([name for name, _, _ in rows], values, 0.6,
                  color=[colour for _, _, colour in rows])
    top = max(values + [FALSE_ALARM_TARGET or 0]) + 2.5
    for bar, (_, entry, _), value in zip(bars, rows, values):
        label = f"{entry['alerts']} of {entry['total']}\n({value:.1f}%)"
        centre = bar.get_x() + bar.get_width() / 2
        if value >= 0.25 * top:
            # Tall bar: put the label inside it, clear of the target line.
            ax.text(centre, value - 0.15, label, ha="center", va="top",
                    fontsize=8.5, color="white")
        else:
            ax.text(centre, value + 0.15, label, ha="center", va="bottom", fontsize=8.5)

    if FALSE_ALARM_TARGET is not None:
        ax.axhline(FALSE_ALARM_TARGET, color="#B00020", linestyle="--", linewidth=1)
        ax.text((len(rows) - 1) / 2, FALSE_ALARM_TARGET + 0.12,
                f"Proposal target: under {FALSE_ALARM_TARGET:.0f}%",
                ha="center", va="bottom", fontsize=8.5, color="#B00020")
    ax.set_ylim(0, top)
    ax.set_ylabel("Clean questions that raised an alert (%)")
    total = rows[0][1]["total"]
    ax.set_title(f"False alarms on {total} clean questions", loc="left")
    ax.grid(axis="y", color="#E3E3E3", linewidth=0.8)
    tidy(ax)
    save(fig, "fig2_false_alarms.png")


def figure_time(summary):
    times = summary["milliseconds_median"]
    rows = [(label.split(":")[0], times[key], colour)
            for key, label, colour in CHECKS if key in times]
    if "answer generation" in times:
        rows.append(("Writing the answer\n(no checks)", times["answer generation"], "#3B3B3B"))
    rows.reverse()                                   # first row at the top

    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.barh([name for name, _, _ in rows], [v for _, v, _ in rows], 0.6,
                   color=[colour for _, _, colour in rows])
    longest = max(v for _, v, _ in rows)
    for bar, (_, value, _) in zip(bars, rows):
        text = f"{value:.0f} ms" if value >= 1 else "under 1 ms"
        ax.text(value + longest * 0.012, bar.get_y() + bar.get_height() / 2, text,
                va="center", fontsize=9)
    ax.set_xlim(0, longest * 1.15)
    ax.set_xlabel("Milliseconds per question (median)")
    ax.set_title("Time taken by each check, compared with writing the answer", loc="left")
    ax.grid(axis="x", color="#E3E3E3", linewidth=0.8)
    tidy(ax)
    save(fig, "fig3_time.png")


def figure_attack_reach(summary):
    detection = summary["detection"]
    attacks = [(key, label) for key, label in ATTACKS if key in detection]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    values = [100.0 * detection[k]["retrieved"] / detection[k]["poisons"] for k, _ in attacks]
    bars = ax.bar([label for _, label in attacks], values, 0.55, color="#5B6C8F")
    for bar, (key, _), value in zip(bars, attacks, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5,
                f"{detection[key]['retrieved']} of {detection[key]['poisons']}\n({value:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 122)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Poisons retrieved (%)")
    ax.set_title("How often each attack reached the top three passages", loc="left")
    ax.grid(axis="y", color="#E3E3E3", linewidth=0.8)
    tidy(ax)
    save(fig, "fig4_attack_reach.png")


# ------------------------------------------------------------------ tables

def cell(entry):
    if not entry["total"]:
        return "not run"
    return f"{entry['alerts']}/{entry['total']} ({percent(entry):.1f}%)"


def write_tables(summary):
    detection = summary["detection"]
    false_alarms = summary["false_alarms"]
    columns = [(key, label.split(":")[0]) for key, label, _ in CHECKS]
    columns += [("layers", "Layers 1-3 together"), ("any", "All four checks")]

    lines = [
        "# RAG-Shield benchmark results",
        "",
        f"Source: {summary['results_file']}, {summary['cases']} cases.",
        "",
        "## Table 1. Poisons detected, by attack type and check",
        "",
        "Counted over the poisons that reached the top three passages.",
        "",
        "| Attack | Poisons | Retrieved | " + " | ".join(name for _, name in columns) + " |",
        "|---|---|---|" + "---|" * len(columns),
    ]
    for key, label in ATTACKS:
        if key not in detection:
            continue
        d = detection[key]
        cells = [cell(d[c]) if c in d else "" for c, _ in columns]
        lines.append(f"| {label} | {d['poisons']} | {d['retrieved']} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Table 2. False alarms on clean questions",
        "",
        "| Check | False alarms |",
        "|---|---|",
    ]
    for key, name in columns:
        if key in false_alarms:
            lines.append(f"| {name} | {cell(false_alarms[key])} |")

    held_out = summary.get("layer3_held_out")
    if held_out:
        lines += [
            "",
            "## Table 3. Layer 3 on the two subject areas it never trained on",
            "",
            "| Passages | Flagged by Layer 3 |",
            "|---|---|",
            f"| Clean questions (false alarms) | {cell(held_out['clean'])} |",
        ]
        for key, label in ATTACKS:
            if key in held_out:
                lines.append(f"| {label} poisons (detected) | {cell(held_out[key])} |")

    names = {"layer1": "Layer 1", "layer2": "Layer 2", "layer3": "Layer 3",
             "integrity": "Integrity check",
             "local checks": "Layers 2, 3 and integrity together",
             "all checks": "All four checks together",
             "answer generation": "Writing the answer (no checks)"}
    lines += [
        "",
        "## Table 4. Time per question (median)",
        "",
        "| Step | Milliseconds |",
        "|---|---|",
    ]
    for key, value in summary["milliseconds_median"].items():
        lines.append(f"| {names.get(key, key)} | {value:.1f} |")

    path = os.path.join(OUT_DIR, "results_tables.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("  wrote", path)


def main():
    if not os.path.exists(SUMMARY_FILE):
        sys.exit(f"{SUMMARY_FILE} not found. Run `python benchmark.py --summary` first.")
    with open(SUMMARY_FILE, encoding="utf-8") as f:
        summary = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11.5})

    print("Writing figures and tables:")
    figure_detection(summary)
    figure_false_alarms(summary)
    figure_time(summary)
    figure_attack_reach(summary)
    write_tables(summary)


if __name__ == "__main__":
    main()
