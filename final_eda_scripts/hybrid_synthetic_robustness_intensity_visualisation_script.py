from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm



PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "final_results_eda"

RAW_RESULTS_PATH = OUTPUT_DIR / "synthetic_perturbation_robustness_results.csv"
SUMMARY_RESULTS_PATH = OUTPUT_DIR / "synthetic_perturbation_robustness_summary.csv"



USE_COMPRESSED_AXIS = True

EXPANDED_REGION_START = 98.0
LOWER_AXIS_FRACTION = 0.36

REPRESENTATIONS = [
    "XY",
    "Translated",
    "Scaled",
    "Norm. distance",
    "Angles",
    "Hybrid",
]



TEXT_BLUE = "#6fb7b6"
DARK_NAVY = "#102A43"

SOFT_RED = "#f4b5b0"
GOLD = "#f3c65b"
GREEN = "#7ecb80"
PURPLE = "#9A86D1"
GRID_COLOR = "#d8eeee"
BG_COLOR = "white"

REP_COLORS = {
    "XY": SOFT_RED,
    "Translated": GOLD,
    "Scaled": GREEN,
    "Norm. distance": TEXT_BLUE,
    "Angles": PURPLE,
    "Hybrid": DARK_NAVY,
}

REP_MARKERS = {
    "XY": "o",
    "Translated": "s",
    "Scaled": "^",
    "Norm. distance": "D",
    "Angles": "v",
    "Hybrid": "P",
}

REP_LINESTYLES = {
    "XY": "-",
    "Translated": "-",
    "Scaled": "-",
    "Norm. distance": "-",
    "Angles": "-",
    "Hybrid": "-",
}

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "savefig.facecolor": BG_COLOR,
    "axes.edgecolor": "none",
    "svg.fonttype": "path",
    "mathtext.fontset": "dejavusans",
})



FONT_DIRS = [
    PROJECT_ROOT / "fonts",
    PROJECT_ROOT / "final_eda_scripts" / "fonts",
]


def normalize_font_name(name):
    return str(name).lower().replace(" ", "").replace("-", "").replace("_", "")


def collect_font_files():
    files = []

    for font_dir in FONT_DIRS:
        if font_dir.exists():
            files.extend(list(font_dir.rglob("*.ttf")))
            files.extend(list(font_dir.rglob("*.otf")))

    try:
        files.extend(fm.findSystemFonts(fontpaths=None, fontext="ttf"))
        files.extend(fm.findSystemFonts(fontpaths=None, fontext="otf"))
    except Exception:
        pass

    unique = []
    seen = set()

    for path in files:
        path = Path(path)

        if str(path) not in seen:
            unique.append(path)
            seen.add(str(path))

    return unique


def choose_font(candidate_names, fallback_family="DejaVu Sans", weight="normal"):
    font_files = collect_font_files()
    font_records = []

    for path in font_files:
        try:
            font_name = fm.FontProperties(fname=str(path)).get_name()
            font_records.append((path, font_name, normalize_font_name(font_name)))
        except Exception:
            continue

    candidate_norms = [normalize_font_name(name) for name in candidate_names]

    for candidate in candidate_norms:
        for path, font_name, norm_name in font_records:
            if norm_name == candidate:
                fm.fontManager.addfont(str(path))
                print(f"Using font: {font_name} -> {path}")
                return fm.FontProperties(fname=str(path), weight=weight)

    for candidate in candidate_norms:
        for path, font_name, norm_name in font_records:
            if candidate in norm_name or norm_name in candidate:
                fm.fontManager.addfont(str(path))
                print(f"Using font: {font_name} -> {path}")
                return fm.FontProperties(fname=str(path), weight=weight)

    print(f"Requested fonts not found. Using fallback: {fallback_family}")
    return fm.FontProperties(family=fallback_family, weight=weight)


LABEL_FP = choose_font(
    candidate_names=[
        "Montserrat",
        "Aptos",
        "Bahnschrift",
        "Century Gothic",
        "Segoe UI",
    ],
    fallback_family="DejaVu Sans",
    weight="normal",
)

BOLD_FP = choose_font(
    candidate_names=[
        "Montserrat SemiBold",
        "Montserrat Bold",
        "Montserrat",
        "Aptos Display",
        "Aptos",
        "Bahnschrift",
        "Century Gothic",
        "Segoe UI Semibold",
        "Segoe UI",
    ],
    fallback_family="DejaVu Sans",
    weight="bold",
)



def compressed_accuracy_forward(y):
    """
    Compress everything below EXPANDED_REGION_START and expand
    EXPANDED_REGION_START--100.
    """
    y = np.asarray(y, dtype=float)

    start = EXPANDED_REGION_START
    lower_fraction = LOWER_AXIS_FRACTION
    upper_fraction = 1.0 - lower_fraction

    return np.where(
        y <= start,
        y * (lower_fraction / start),
        lower_fraction + (y - start) * (upper_fraction / (100.0 - start))
    )


def compressed_accuracy_inverse(t):
    """
    Inverse transform for the custom accuracy axis.
    """
    t = np.asarray(t, dtype=float)

    start = EXPANDED_REGION_START
    lower_fraction = LOWER_AXIS_FRACTION
    upper_fraction = 1.0 - lower_fraction

    return np.where(
        t <= lower_fraction,
        t * (start / lower_fraction),
        start + (t - lower_fraction) * ((100.0 - start) / upper_fraction)
    )



def load_summary():
    if SUMMARY_RESULTS_PATH.exists():
        summary = pd.read_csv(SUMMARY_RESULTS_PATH)
        print(f"Loaded summary: {SUMMARY_RESULTS_PATH}")
        return summary

    if not RAW_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {SUMMARY_RESULTS_PATH} or {RAW_RESULTS_PATH}. "
            f"Run 01_run_synthetic_robustness_experiment.py first."
        )

    raw = pd.read_csv(RAW_RESULTS_PATH)

    summary = (
        raw
        .groupby(["level", "representation"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
        )
    )

    summary.to_csv(SUMMARY_RESULTS_PATH, index=False)
    print(f"Created summary: {SUMMARY_RESULTS_PATH}")

    return summary



def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="both", colors=TEXT_BLUE, length=0, pad=7)

    for label in ax.get_xticklabels():
        label.set_fontproperties(LABEL_FP)
        label.set_color(TEXT_BLUE)
        label.set_fontsize(11)

    for label in ax.get_yticklabels():
        label.set_fontproperties(LABEL_FP)
        label.set_color(TEXT_BLUE)
        label.set_fontsize(10)

    ax.grid(True, color=GRID_COLOR, linewidth=0.9, alpha=0.95)
    ax.set_axisbelow(True)


def save_figure(fig, basename):
    png_path = OUTPUT_DIR / f"{basename}.png"
    pdf_path = OUTPUT_DIR / f"{basename}.pdf"
    svg_path = OUTPUT_DIR / f"{basename}.svg"

    fig.savefig(png_path, dpi=400, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Saved PNG to: {png_path}")

    try:
        fig.savefig(pdf_path, bbox_inches="tight", facecolor=BG_COLOR)
        print(f"Saved PDF to: {pdf_path}")
    except Exception as e:
        print("\nPDF save failed because of font embedding.")
        print(f"Error: {e}")
        print("Saving SVG instead.")

        try:
            fig.savefig(svg_path, bbox_inches="tight", facecolor=BG_COLOR)
            print(f"Saved SVG to: {svg_path}")
        except Exception as e2:
            print("\nSVG save also failed.")
            print(f"Error: {e2}")

    plt.close(fig)



def make_plot(summary):
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    handles = []
    labels = []

    plot_order = [rep for rep in REPRESENTATIONS if rep != "Hybrid"] + ["Hybrid"]

    for rep in plot_order:
        sub = summary[summary["representation"] == rep].sort_values("level")

        if sub.empty:
            print(f"Skipping missing representation: {rep}")
            continue

        x = sub["level"].to_numpy()
        y = sub["mean_accuracy"].to_numpy()
        yerr = sub["std_accuracy"].fillna(0).to_numpy()

        line = ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=REP_COLORS[rep],
            marker=REP_MARKERS[rep],
            linestyle=REP_LINESTYLES[rep],
            linewidth=2.5 if rep == "Hybrid" else 1.8,
            markersize=8.0 if rep == "Hybrid" else 6.8,
            capsize=2.8,
            capthick=1.1,
            elinewidth=1.0,
            alpha=1.0 if rep == "Hybrid" else 0.86,
            label=rep,
            zorder=10 if rep == "Hybrid" else 3,
        )

        handles.append(line.lines[0])
        labels.append(rep)

    handle_map = dict(zip(labels, handles))
    legend_handles = [handle_map[rep] for rep in REPRESENTATIONS if rep in handle_map]
    legend_labels = [rep for rep in REPRESENTATIONS if rep in handle_map]

    fig.suptitle(
        "SYNTHETIC PERTURBATION ROBUSTNESS",
        x=0.5,
        y=0.975,
        ha="center",
        va="top",
        fontsize=18,
        fontproperties=BOLD_FP,
        color=TEXT_BLUE,
    )

    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=3,
        frameon=False,
        prop=LABEL_FP,
        handlelength=2.3,
        columnspacing=2.0,
        labelspacing=0.8,
    )

    for text in legend.get_texts():
        text.set_color(TEXT_BLUE)

    ax.set_xlabel(
        "Synthetic perturbation level",
        fontproperties=LABEL_FP,
        fontsize=13,
        color=TEXT_BLUE,
        labelpad=14,
    )

    ax.set_ylabel(
        "Held-out test accuracy (%)",
        fontproperties=LABEL_FP,
        fontsize=13,
        color=TEXT_BLUE,
        labelpad=14,
    )

    levels = sorted(summary["level"].unique())
    ax.set_xticks(levels)
    ax.set_xticklabels([str(int(k)) for k in levels])

    if USE_COMPRESSED_AXIS:
        ax.set_yscale(
            "function",
            functions=(compressed_accuracy_forward, compressed_accuracy_inverse),
        )

        ax.set_ylim(0, 100.15)

        yticks = [0, 40, 80, 90, 98, 99, 100]
        yticklabels = ["0", "40", "80", "90", "98", "99", "100"]

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

        ax.axhline(
            EXPANDED_REGION_START,
            color=GRID_COLOR,
            linewidth=1.0,
            linestyle="--",
            alpha=0.9,
            zorder=1,
        )

        ax.text(
            0.04,
            EXPANDED_REGION_START + 0.06,
            f"expanded {EXPANDED_REGION_START:.0f}--100 region",
            ha="left",
            va="bottom",
            fontsize=8.0,
            fontproperties=LABEL_FP,
            color=TEXT_BLUE,
            alpha=0.85,
        )

    else:
        ymin = max(0, summary["mean_accuracy"].min() - 5)
        ax.set_ylim(ymin, 101)

    style_axis(ax)

    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.16, top=0.78)

    save_figure(fig, "synthetic_perturbation_robustness")



if __name__ == "__main__":
    summary_df = load_summary()
    make_plot(summary_df)

    print("\nDone. Figure saved in:")
    print(OUTPUT_DIR)
