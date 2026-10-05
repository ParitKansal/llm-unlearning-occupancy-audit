"""Results figure for the README (light + dark). Numbers from experiments/kill_v1/RESULTS.md. Run: python make_figure.py"""
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
# rows: label, truth (relearning), occupancy Mh psi-hat, CI lo, CI hi, single probe, any-of-K
ROWS = [
    ("GradDiff", 0.615, 0.654, 0.580, 0.751, 0.380, 0.720),
    ("NPO", 0.235, 0.251, 0.085, 0.342, 0.075, 0.370),
    ("RMU", 0.6475, 0.320, 0.267, 0.396, 0.193, 0.530),
    ("Control: never\nlearned the facts", 0.0, 0.081, 0.021, 0.117, 0.038, 0.160),
]
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e0",
                  ours="#2a78d6", single="#eb6834", anyk="#1baf7a"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#3a3936",
                 ours="#3987e5", single="#d95926", anyk="#199e70"),
}


def draw(theme, path):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    fig.patch.set_facecolor(t["surface"]); ax.set_facecolor(t["surface"])
    n = len(ROWS)
    for i, (name, truth, psi, lo, hi, single, anyk) in enumerate(ROWS):
        y = n - 1 - i
        ax.plot([truth, truth], [y - 0.36, y + 0.36], color=t["ink"], lw=3, solid_capstyle="round", zorder=2)
        ax.plot([lo, hi], [y, y], color=t["ours"], lw=2, solid_capstyle="round", zorder=3)
        ax.scatter([psi], [y], s=70, color=t["ours"], edgecolor=t["surface"], linewidth=2, zorder=4, marker="o")
        ax.scatter([single], [y + 0.22], s=58, color=t["single"], edgecolor=t["surface"], linewidth=2, zorder=4, marker="s")
        ax.scatter([anyk], [y - 0.22], s=95, color=t["anyk"], edgecolor=t["surface"], linewidth=2, zorder=4, marker="^")
    ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in ROWS][::-1], color=t["ink"], fontsize=10)
    ax.set_ylim(-0.6, n - 0.4); ax.set_xlim(-0.02, 1.0)
    ax.set_xticks([0, .2, .4, .6, .8, 1.0]); ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], color=t["ink2"], fontsize=9)
    ax.set_xlabel("Share of \u201cforgotten\u201d facts the model still stores", color=t["ink2"], fontsize=9.5)
    ax.grid(axis="x", color=t["grid"], lw=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"]); ax.tick_params(length=0)
    handles = [
        plt.Line2D([], [], color=t["ink"], lw=3, label="Truth"),
        plt.Line2D([], [], color=t["ours"], marker="o", lw=2, markersize=7, label="Our estimate (likely range)"),
        plt.Line2D([], [], color=t["single"], marker="s", lw=0, markersize=7, label="Usual one-question test"),
        plt.Line2D([], [], color=t["anyk"], marker="^", lw=0, markersize=8, label="Any of 10 questions"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=8.5,
                    handlelength=1.6, columnspacing=1.4)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    fig.suptitle("How much “forgotten” knowledge is still stored?", x=0.015, ha="left", y=0.985,
                 color=t["ink"], fontsize=12.5, fontweight="bold")
    fig.text(0.015, 0.905, "Small AI model (Llama-3.2-1B), 400 made-up facts, three forgetting methods. "
             "Pass/fail rules fixed in advance; verdict: NO-GO.", color=t["ink2"], fontsize=8.8)
    fig.subplots_adjust(left=0.17, right=0.975, bottom=0.12, top=0.80)
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    draw("light", HERE / "results.png")
    draw("dark", HERE / "results-dark.png")
    print("wrote results.png and results-dark.png")
