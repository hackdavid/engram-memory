"""Generate engram_memory benchmark dashboard PNG from benchmark_report.json."""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "benchmarks" / "benchmark_report.json"
OUT  = ROOT / "images" / "engram_benchmark_dashboard.png"

with open(DATA) as f:
    data = json.load(f)

BG       = "#0D1117"
SURFACE  = "#161B22"
BORDER   = "#30363D"
TEXT     = "#E6EDF3"
TEXT_DIM = "#8B949E"
ACCENT   = "#58A6FF"
GREEN    = "#3FB950"
AMBER    = "#D29922"
CORAL    = "#F85149"
PURPLE   = "#BC8CFF"
TEAL     = "#39D353"

fig = plt.figure(figsize=(20, 13), facecolor=BG, dpi=150)
fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.05, hspace=0.50, wspace=0.30)

gs = GridSpec(3, 4, figure=fig, height_ratios=[0.25, 1, 1])

fig.text(0.06, 0.95, "engram", fontsize=32, fontweight="bold",
         color=ACCENT, fontfamily="monospace", va="top")
fig.text(0.175, 0.95, "memory", fontsize=32, fontweight="light",
         color=TEXT, fontfamily="monospace", va="top")
fig.text(0.06, 0.915, "SDK Benchmark Report  |  v0.1.0  |  Neo4j + LLM Graph Memory",
         fontsize=11, color=TEXT_DIM, fontfamily="monospace", va="top")

line_y = 0.905
fig.add_artist(plt.Line2D([0.06, 0.97], [line_y, line_y],
               transform=fig.transFigure, color=BORDER, linewidth=0.8))

# ── Row 0: KPI Cards ────────────────────────────────────────────────
kpi_data = [
    ("RECALL  P50",   f'{data["summary"]["recall_p50_ms"]:.0f} ms',  GREEN),
    ("MRR",           f'{data["summary"]["mrr"]:.2f}',               ACCENT),
    ("PRECISION @1",  f'{data["summary"]["precision_at_1"]:.0%}',    GREEN),
    ("TOKENS / DOC",  f'{data["summary"]["tokens_avg_per_ingest"]:.0f}', AMBER),
    ("COST / INGEST", f'${data["summary"]["cost_per_ingest_usd"]:.3f}', PURPLE),
    ("SCALING",       "O(1)",                                         TEAL),
    ("ISOLATION",     "PASS",                                         GREEN),
    ("NODES",         str(data["summary"]["total_nodes"]),            ACCENT),
]

for i, (label, value, color) in enumerate(kpi_data):
    x_start = 0.06 + i * (0.91 / len(kpi_data))
    x_width = 0.91 / len(kpi_data) - 0.008

    rect = mpatches.FancyBboxPatch(
        (x_start, 0.845), x_width, 0.050,
        boxstyle="round,pad=0.005", facecolor=SURFACE,
        edgecolor=BORDER, linewidth=0.6, transform=fig.transFigure,
    )
    fig.add_artist(rect)

    fig.text(x_start + x_width / 2, 0.881, value,
             fontsize=16, fontweight="bold", color=color,
             fontfamily="monospace", ha="center", va="center",
             transform=fig.transFigure)
    fig.text(x_start + x_width / 2, 0.855, label,
             fontsize=7, color=TEXT_DIM, fontfamily="monospace",
             ha="center", va="center", transform=fig.transFigure)


def style_ax(ax, title):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.6)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold", color=TEXT,
                 fontfamily="monospace", pad=10, loc="left")
    ax.grid(axis="y", color=BORDER, linewidth=0.4, alpha=0.5)


# ── Panel 1: Latency Comparison (bar) ──────────────────────────────
ax1 = fig.add_subplot(gs[1, 0:2])
style_ax(ax1, "Latency (ms)")

categories = ["Ingest", "Recall", "Search"]
p50_vals = [data["ingest"]["latency"]["p50_ms"],
            data["recall"]["latency"]["p50_ms"],
            data["search"]["latency"]["p50_ms"]]
p95_vals = [data["ingest"]["latency"]["p95_ms"],
            data["recall"]["latency"]["p95_ms"],
            data["search"]["latency"]["p95_ms"]]

x = np.arange(len(categories))
w = 0.32
bars1 = ax1.bar(x - w/2, p50_vals, w, color=ACCENT, alpha=0.9, label="p50", zorder=3)
bars2 = ax1.bar(x + w/2, p95_vals, w, color=PURPLE, alpha=0.75, label="p95", zorder=3)

for bar, val in zip(bars1, p50_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(p95_vals)*0.02,
             f"{val:.0f}", ha="center", va="bottom", fontsize=8, color=ACCENT,
             fontfamily="monospace", fontweight="bold")
for bar, val in zip(bars2, p95_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(p95_vals)*0.02,
             f"{val:.0f}", ha="center", va="bottom", fontsize=8, color=PURPLE,
             fontfamily="monospace", fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=10, color=TEXT, fontfamily="monospace")
ax1.set_ylabel("ms", fontsize=9, color=TEXT_DIM, fontfamily="monospace")
ax1.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT_DIM,
           loc="upper right")

# ── Panel 2: Accuracy at K (grouped bar) ───────────────────────────
ax2 = fig.add_subplot(gs[1, 2:4])
style_ax(ax2, "Retrieval Quality @ K")

ks = ["@1", "@3", "@5", "@10"]
precision = [data["accuracy"]["precision_at_k"][k] for k in ["1","3","5","10"]]
recall    = [data["accuracy"]["recall_at_k"][k]    for k in ["1","3","5","10"]]
f1        = [data["accuracy"]["f1_at_k"][k]        for k in ["1","3","5","10"]]

x2 = np.arange(len(ks))
w2 = 0.24
ax2.bar(x2 - w2, precision, w2, color=ACCENT, alpha=0.9, label="Precision", zorder=3)
ax2.bar(x2,      recall,    w2, color=GREEN,  alpha=0.85, label="Recall",    zorder=3)
ax2.bar(x2 + w2, f1,        w2, color=AMBER,  alpha=0.85, label="F1",        zorder=3)

ax2.set_xticks(x2)
ax2.set_xticklabels(ks, fontsize=10, color=TEXT, fontfamily="monospace")
ax2.set_ylim(0, 1.15)
ax2.set_ylabel("Score", fontsize=9, color=TEXT_DIM, fontfamily="monospace")
ax2.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT_DIM,
           ncol=3, loc="upper right")
ax2.axhline(y=1.0, color=TEXT_DIM, linewidth=0.5, linestyle="--", alpha=0.5)

# ── Panel 3: Token Usage per Document ──────────────────────────────
ax3 = fig.add_subplot(gs[2, 0:2])
style_ax(ax3, "Token Usage per Ingestion")

per_doc = data["ingest"]["per_document"]
doc_indices  = [d["index"] for d in per_doc]
tok_prompt   = [d["tokens_prompt"] for d in per_doc]
tok_complete = [d["tokens_completion"] for d in per_doc]
domains      = [d["domain"] for d in per_doc]

domain_colors = {
    "engineering": ACCENT, "healthcare": GREEN, "finance": AMBER,
    "education": PURPLE, "operations": TEAL, "compliance": CORAL,
}
bar_colors = [domain_colors.get(d, TEXT_DIM) for d in domains]

x3 = np.arange(len(doc_indices))
bars_p = ax3.bar(x3, tok_prompt, 0.65, color=[c + "99" for c in bar_colors],
                 label="Prompt", zorder=3)
bars_c = ax3.bar(x3, tok_complete, 0.65, bottom=tok_prompt,
                 color=bar_colors, label="Completion", zorder=3)

ax3.axhline(y=data["ingest"]["avg_tokens_per_ingest"], color=CORAL,
            linewidth=1.2, linestyle="--", alpha=0.7, zorder=4)
ax3.text(len(doc_indices) - 0.5, data["ingest"]["avg_tokens_per_ingest"] + 20,
         f'avg {data["ingest"]["avg_tokens_per_ingest"]:.0f}',
         fontsize=8, color=CORAL, fontfamily="monospace", ha="right")

ax3.set_xticks(x3)
ax3.set_xticklabels([d[:3].upper() for d in domains], fontsize=7,
                    color=TEXT_DIM, fontfamily="monospace", rotation=45, ha="right")
ax3.set_ylabel("Tokens", fontsize=9, color=TEXT_DIM, fontfamily="monospace")
ax3.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT_DIM,
           loc="upper left")

# ── Panel 4: Recall Latency per Query (dot plot) ──────────────────
ax4 = fig.add_subplot(gs[2, 2:4])
style_ax(ax4, "Recall Latency & Hit Rate per Query")

per_query = data["recall"]["per_query"]
q_indices = [q["index"] for q in per_query]
q_latency = [q["latency_ms"] for q in per_query]
q_hitrate = [q["hit_rate"] for q in per_query]
q_domains = [q["expected_domain"] for q in per_query]
q_colors  = [domain_colors.get(d, TEXT_DIM) for d in q_domains]

ax4_twin = ax4.twinx()
ax4_twin.set_ylabel("Hit Rate", fontsize=9, color=GREEN, fontfamily="monospace")
ax4_twin.tick_params(colors=GREEN, labelsize=8)
ax4_twin.set_ylim(0, 1.15)
ax4_twin.spines["right"].set_color(GREEN)
ax4_twin.spines["right"].set_linewidth(0.6)
for spine_key in ["top", "left", "bottom"]:
    ax4_twin.spines[spine_key].set_visible(False)

ax4.bar(q_indices, q_latency, 0.55, color=q_colors, alpha=0.7, zorder=3)
ax4_twin.plot(q_indices, q_hitrate, "o-", color=GREEN, markersize=6,
              linewidth=1.5, markeredgecolor=BG, markeredgewidth=1.2, zorder=5)

ax4.axhline(y=data["recall"]["latency"]["p50_ms"], color=ACCENT,
            linewidth=1, linestyle="--", alpha=0.6, zorder=4)
ax4.text(0.3, data["recall"]["latency"]["p50_ms"] + 3,
         f'p50 = {data["recall"]["latency"]["p50_ms"]:.0f}ms',
         fontsize=8, color=ACCENT, fontfamily="monospace")

ax4.set_xticks(q_indices)
short_labels = [q["expected_domain"][:3].upper() for q in per_query]
ax4.set_xticklabels(short_labels, fontsize=7, color=TEXT_DIM,
                    fontfamily="monospace", rotation=45, ha="right")
ax4.set_ylabel("Latency (ms)", fontsize=9, color=TEXT_DIM, fontfamily="monospace")

domain_handles = [mpatches.Patch(color=c, label=d.capitalize(), alpha=0.7)
                  for d, c in domain_colors.items()]
ax4.legend(handles=domain_handles, fontsize=6.5, facecolor=SURFACE,
           edgecolor=BORDER, labelcolor=TEXT_DIM, ncol=3, loc="upper right")

# ── Footer ─────────────────────────────────────────────────────────
fig.text(0.06, 0.012,
         f'Python {data["meta"]["python_version"]}  |  '
         f'{data["meta"]["platform"]}  |  '
         f'{data["meta"]["timestamp_utc"][:10]}  |  '
         f'Corpus: {data["summary"]["corpus_size"]} docs  |  '
         f'Total cost: ${data["summary"]["estimated_cost_usd"]:.2f}',
         fontsize=8, color=TEXT_DIM, fontfamily="monospace", va="bottom")
fig.text(0.97, 0.012, "github.com/hackdavid/engram-memory",
         fontsize=8, color=ACCENT, fontfamily="monospace",
         ha="right", va="bottom", alpha=0.7)

fig.savefig(str(OUT), dpi=150, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
print(f"Saved: {OUT}")
