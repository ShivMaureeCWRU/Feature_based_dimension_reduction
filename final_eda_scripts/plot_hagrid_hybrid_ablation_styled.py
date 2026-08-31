from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm



PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "final_results_eda"
OUTPUT_DIR.mkdir(exist_ok=True)

SUMMARY_PATH = OUTPUT_DIR / "hagrid_hybrid_ablation_summary.csv"
OUTPUT_BASENAME = "hagrid_hybrid_ablation_styled"



ABLATION_ORDER = [
    "Hybrid",
    "Hybrid - scale",
    "Hybrid - norm-dist",
    "Hybrid - angle",
]

ABLATION_DISPLAY = {
    "Hybrid": "Hybrid",
    "Hybrid - scale": "-Scale",
    "Hybrid - norm-dist": "-Norm-dist",
    "Hybrid - angle": "-Angle",
}

CLASSIFIER_ORDER = [
    "Multinomial logistic",
    "Random forest",
]

TITLE_TEXT = "HAGRID HYBRID ABLATION"
Y_AXIS_LABEL = "Repeated cross-validation accuracy (%)"

SHOW_DROP_LABELS = True



TEXT_BLUE = "#6fb7b6"
DARK_NAVY = "#102A43"
GRID_COLOR = "#d8eeee"
BG_COLOR = "white"

LOGISTIC_COLOR = "#6fb7b6"
RF_COLOR = "#7ecb80"

BAR_ALPHA = 0.95

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


TITLE_FP = choose_font(
    [
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

LABEL_FP = choose_font(
    [
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
    [
        "Montserrat SemiBold",
        "Montserrat Bold",
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



def load_summary():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {SUMMARY_PATH}\n\n"
            f"Run 01_run_hagrid_hybrid_ablation_experiment.py first."
        )

    df = pd.read_csv(SUMMARY_PATH)

    required = {
        "classifier",
        "ablation_variant",
        "mean_accuracy",
        "std_accuracy",
        "drop_from_full_hybrid",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Summary file is missing columns: {missing}")

    df = df[df["classifier"].isin(CLASSIFIER_ORDER)].copy()
    df = df[df["ablation_variant"].isin(ABLATION_ORDER)].copy()

    df["classifier"] = pd.Categorical(
        df["classifier"],
        categories=CLASSIFIER_ORDER,
        ordered=True,
    )

    df["ablation_variant"] = pd.Categorical(
        df["ablation_variant"],
        categories=ABLATION_ORDER,
        ordered=True,
    )

    return df.sort_values(["ablation_variant", "classifier"]).reset_index(drop=True)



def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="both", length=0, colors=TEXT_BLUE, pad=6)

    for label in ax.get_xticklabels():
        label.set_color(TEXT_BLUE)
        label.set_fontproperties(LABEL_FP)
        label.set_fontsize(10.5)

    for label in ax.get_yticklabels():
        label.set_color(TEXT_BLUE)
        label.set_fontproperties(LABEL_FP)
        label.set_fontsize(10.5)

    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.9, alpha=0.9)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)


def focused_ylim(values, stds, pad_low=1.1, pad_high=1.3, min_span=7.0):
    values = np.asarray(values, dtype=float)
    stds = np.asarray(stds, dtype=float)

    lower = np.nanmin(values - stds) - pad_low
    upper = np.nanmax(values + stds) + pad_high

    if upper - lower < min_span:
        mid = 0.5 * (lower + upper)
        lower = mid - min_span / 2
        upper = mid + min_span / 2

    lower = max(0, lower)
    upper = min(100, upper)

    return lower, upper


def save_figure(fig, basename):
    png_path = OUTPUT_DIR / f"{basename}.png"
    pdf_path = OUTPUT_DIR / f"{basename}.pdf"
    svg_path = OUTPUT_DIR / f"{basename}.svg"

    fig.savefig(png_path, dpi=400, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Saved PNG: {png_path}")

    try:
        fig.savefig(pdf_path, bbox_inches="tight", facecolor=BG_COLOR)
        print(f"Saved PDF: {pdf_path}")
    except Exception as e:
        print("\nPDF save failed because of font embedding.")
        print(f"Error: {e}")
        print("Saving SVG instead.")

        try:
            fig.savefig(svg_path, bbox_inches="tight", facecolor=BG_COLOR)
            print(f"Saved SVG: {svg_path}")
        except Exception as e2:
            print("\nSVG save also failed.")
            print(f"Error: {e2}")

    plt.close(fig)



def make_ablation_plot(df):
    fig, ax = plt.subplots(figsize=(8.9, 4.85))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    x = np.arange(len(ABLATION_ORDER))
    width = 0.30

    color_map = {
        "Multinomial logistic": LOGISTIC_COLOR,
        "Random forest": RF_COLOR,
    }

    offsets = {
        "Multinomial logistic": -width / 1.75,
        "Random forest": width / 1.75,
    }

    legend_handles = []
    legend_labels = []

    all_means = []
    all_stds = []

    for classifier in CLASSIFIER_ORDER:
        sub = df[df["classifier"].astype(str) == classifier].copy()

        means = []
        stds = []
        drops = []

        for variant in ABLATION_ORDER:
            row = sub[sub["ablation_variant"].astype(str) == variant]

            if row.empty:
                means.append(np.nan)
                stds.append(np.nan)
                drops.append(np.nan)
            else:
                means.append(float(row["mean_accuracy"].iloc[0]))
                stds.append(float(row["std_accuracy"].iloc[0]))
                drops.append(float(row["drop_from_full_hybrid"].iloc[0]))

        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)
        drops = np.array(drops, dtype=float)

        all_means.extend(means[~np.isnan(means)])
        all_stds.extend(stds[~np.isnan(stds)])

        xpos = x + offsets[classifier]

        bars = ax.bar(
            xpos,
            means,
            width=width,
            yerr=stds,
            color=color_map[classifier],
            edgecolor="none",
            alpha=BAR_ALPHA,
            capsize=4,
            error_kw={
                "elinewidth": 1.25,
                "capthick": 1.15,
                "ecolor": DARK_NAVY,
                "alpha": 0.72,
            },
            label=classifier,
            zorder=4,
        )

        legend_handles.append(bars[0])
        legend_labels.append(classifier)

        for xi, mean, std, drop, variant in zip(xpos, means, stds, drops, ABLATION_ORDER):
            if np.isnan(mean):
                continue

            ax.text(
                xi,
                mean + std + 0.32,
                f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.7,
                fontproperties=BOLD_FP,
                color=DARK_NAVY,
                zorder=8,
            )

            if SHOW_DROP_LABELS and variant != "Hybrid" and abs(drop) > 1e-8:
                ax.text(
                    xi,
                    mean - 1.75,
                    f"-{abs(drop):.2f}",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    fontproperties=BOLD_FP,
                    fontweight="bold",
                    color="white",
                    alpha=1.0,
                    zorder=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([ABLATION_DISPLAY[v] for v in ABLATION_ORDER])

    lower, upper = focused_ylim(np.array(all_means), np.array(all_stds))
    ax.set_ylim(lower, upper)

    ax.set_ylabel(
        Y_AXIS_LABEL,
        fontproperties=LABEL_FP,
        fontsize=12,
        color=TEXT_BLUE,
        labelpad=12,
    )

    ax.set_xlabel(
        "Ablation variant",
        fontproperties=LABEL_FP,
        fontsize=12,
        color=TEXT_BLUE,
        labelpad=11,
    )

    style_axis(ax)

    fig.suptitle(
        TITLE_TEXT,
        x=0.5,
        y=0.965,
        ha="center",
        va="top",
        fontsize=18,
        fontproperties=TITLE_FP,
        color=TEXT_BLUE,
    )

    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=2,
        frameon=False,
        prop=LABEL_FP,
        handlelength=2.4,
        columnspacing=2.6,
        labelspacing=0.8,
    )

    for text in legend.get_texts():
        text.set_color(TEXT_BLUE)

    fig.subplots_adjust(
        left=0.13,
        right=0.975,
        bottom=0.20,
        top=0.765,
    )

    save_figure(fig, OUTPUT_BASENAME)



if __name__ == "__main__":
    summary_df = load_summary()
    make_ablation_plot(summary_df)

    print("\nDone. Figure saved in:")
    print(OUTPUT_DIR)