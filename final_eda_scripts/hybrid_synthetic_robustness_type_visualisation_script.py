from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.gridspec import GridSpec



PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "final_results_eda"
OUTPUT_DIR.mkdir(exist_ok=True)

SUMMARY_PATH = OUTPUT_DIR / "perturbation_type_summary.csv"
RAW_RESULTS_PATH = OUTPUT_DIR / "perturbation_type_results.csv"



PANEL_ORDER = [
    "Translation only",
    "Scale only",
    "Translation + scale",
]

PANEL_TITLES = {
    "Translation only": "TRANSLATION ONLY",
    "Scale only": "SCALE ONLY",
    "Translation + scale": "TRANSLATION + SCALE",
}

REPRESENTATION_ORDER = [
    "XY",
    "Translated",
    "Scaled",
    "Norm. distance",
    "Angles",
    "Hybrid",
]

STANDARD_LOWER_MAX = 6.0
STANDARD_LOWER_FRACTION = 0.66

PANEL_SCALES = {
    "Translation only": {
        "lower_max": STANDARD_LOWER_MAX,
        "upper_max": 80.0,
        "lower_fraction": STANDARD_LOWER_FRACTION,
        "ticks": [0, 2, 4, 6, 30, 50, 70],
    },
    "Scale only": {
        "lower_max": STANDARD_LOWER_MAX,
        "upper_max": 55.0,
        "lower_fraction": STANDARD_LOWER_FRACTION,
        "ticks": [0, 2, 4, 6, 20, 35, 50],
    },
    "Translation + scale": {
        "lower_max": STANDARD_LOWER_MAX,
        "upper_max": 80.0,
        "lower_fraction": STANDARD_LOWER_FRACTION,
        "ticks": [0, 2, 4, 6, 40, 60, 80],
    },
}

SHOW_COMPRESSION_GUIDE = True



TEXT_BLUE = "#6fb7b6"
DARK_NAVY = "#102A43"
GRID_COLOR = "#d8eeee"
BG_COLOR = "white"

SOFT_RED = "#f4b5b0"
GOLD = "#f3c65b"
GREEN = "#7ecb80"
TEAL = "#75bdbb"
PURPLE = "#9A86D1"
NAVY = DARK_NAVY

REP_COLORS = {
    "XY": SOFT_RED,
    "Translated": GOLD,
    "Scaled": GREEN,
    "Norm. distance": TEAL,
    "Angles": PURPLE,
    "Hybrid": NAVY,
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

REP_LINEWIDTHS = {
    "XY": 1.85,
    "Translated": 1.85,
    "Scaled": 1.85,
    "Norm. distance": 1.85,
    "Angles": 1.85,
    "Hybrid": 2.55,
}

REP_MARKERSIZES = {
    "XY": 6.2,
    "Translated": 6.2,
    "Scaled": 6.2,
    "Norm. distance": 6.2,
    "Angles": 6.2,
    "Hybrid": 8.0,
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

HEADER_FP = choose_font(
    [
        "Montserrat SemiBold",
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



def load_summary():
    if SUMMARY_PATH.exists():
        df = pd.read_csv(SUMMARY_PATH)
        print(f"Loaded summary from: {SUMMARY_PATH}")
        return df

    if not RAW_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find either:\n{SUMMARY_PATH}\n{RAW_RESULTS_PATH}\n\n"
            f"Run the perturbation-type experiment script first."
        )

    raw = pd.read_csv(RAW_RESULTS_PATH)

    summary = (
        raw
        .groupby(
            ["perturbation_type", "level", "representation", "classifier"],
            as_index=False,
        )
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_accuracy_drop=("accuracy_drop", "mean"),
            std_accuracy_drop=("accuracy_drop", "std"),
            clean_accuracy=("clean_accuracy", "first"),
            translation=("translation", "first"),
            scale_low=("scale_low", "first"),
            scale_high=("scale_high", "first"),
            noise=("noise", "first"),
        )
    )

    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"Created summary: {SUMMARY_PATH}")

    return summary



def make_panel_transform(panel_name, panel_df):
    cfg = PANEL_SCALES[panel_name].copy()

    data_max = float(panel_df["mean_accuracy_drop"].max())
    data_max = max(data_max, 0.0)

    upper_max = max(cfg["upper_max"], data_max + 2.0)
    lower_max = cfg["lower_max"]
    lower_fraction = cfg["lower_fraction"]

    if upper_max <= lower_max:
        upper_max = lower_max + 1.0

    def forward(y):
        y = np.asarray(y, dtype=float)
        y_clipped = np.maximum(y, 0.0)

        lower_part = (y_clipped / lower_max) * lower_fraction
        upper_part = lower_fraction + (
            (y_clipped - lower_max) / (upper_max - lower_max)
        ) * (1.0 - lower_fraction)

        return np.where(y_clipped <= lower_max, lower_part, upper_part)

    return forward, lower_max, upper_max


def transformed_errorbar_values(forward, y, yerr):
    y = np.asarray(y, dtype=float)

    if yerr is None:
        return None

    yerr = np.asarray(yerr, dtype=float)

    y_low = np.maximum(y - yerr, 0.0)
    y_high = y + yerr

    y_t = forward(y)
    y_low_t = forward(y_low)
    y_high_t = forward(y_high)

    lower_err = np.maximum(y_t - y_low_t, 0.0)
    upper_err = np.maximum(y_high_t - y_t, 0.0)

    return np.vstack([lower_err, upper_err])



def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(True, color=GRID_COLOR, linewidth=0.9, alpha=0.95)
    ax.set_axisbelow(True)

    ax.tick_params(axis="both", length=0, colors=TEXT_BLUE, pad=6)

    for label in ax.get_xticklabels():
        label.set_color(TEXT_BLUE)
        label.set_fontproperties(LABEL_FP)
        label.set_fontsize(10)

    for label in ax.get_yticklabels():
        label.set_color(TEXT_BLUE)
        label.set_fontproperties(LABEL_FP)
        label.set_fontsize(10)


def format_tick_label(value):
    value = float(value)
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    return f"{value:.1f}"


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



def add_panel(ax, panel_name, panel_df):
    forward, lower_max, upper_max = make_panel_transform(panel_name, panel_df)

    legend_handles = {}
    panel_df = panel_df.copy()

    for rep in REPRESENTATION_ORDER:
        sub = panel_df[panel_df["representation"] == rep].sort_values("level")

        if sub.empty:
            continue

        x = sub["level"].to_numpy()
        y = sub["mean_accuracy_drop"].to_numpy()
        y_t = forward(y)

        if "std_accuracy_drop" in sub.columns:
            yerr = sub["std_accuracy_drop"].fillna(0).to_numpy()
            yerr_t = transformed_errorbar_values(forward, y, yerr)
        else:
            yerr_t = None

        plot_kwargs = dict(
            color=REP_COLORS[rep],
            linestyle=REP_LINESTYLES[rep],
            linewidth=REP_LINEWIDTHS[rep],
            marker=REP_MARKERS[rep],
            markersize=REP_MARKERSIZES[rep],
            alpha=1.0 if rep == "Hybrid" else 0.95,
            zorder=10 if rep == "Hybrid" else 4,
            label=rep,
        )

        if yerr_t is not None and np.nanmax(yerr_t) > 0:
            handle = ax.errorbar(
                x,
                y_t,
                yerr=yerr_t,
                capsize=2.3,
                capthick=0.8,
                elinewidth=0.8,
                **plot_kwargs,
            )
            legend_handles[rep] = handle.lines[0]
        else:
            line, = ax.plot(x, y_t, **plot_kwargs)
            legend_handles[rep] = line

    ax.set_ylim(-0.02, 1.02)

    levels = sorted(panel_df["level"].dropna().unique())
    ax.set_xticks(levels)
    ax.set_xticklabels([str(int(v)) for v in levels])

    ticks_actual = PANEL_SCALES[panel_name]["ticks"]
    ticks_actual = [t for t in ticks_actual if 0 <= t <= upper_max]

    tick_positions = forward(np.array(ticks_actual, dtype=float))
    tick_labels = [format_tick_label(t) for t in ticks_actual]

    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    if SHOW_COMPRESSION_GUIDE:
        ax.axhline(
            forward(lower_max),
            color=GRID_COLOR,
            linewidth=1.0,
            linestyle="--",
            alpha=0.85,
            zorder=1,
        )

        ax.text(
            0.02,
            forward(lower_max) + 0.018,
            f"compressed above {format_tick_label(lower_max)}%",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=7.5,
            fontproperties=LABEL_FP,
            color=TEXT_BLUE,
            alpha=0.75,
        )

    ax.set_title(
        PANEL_TITLES[panel_name],
        fontsize=12.5,
        color=TEXT_BLUE,
        fontproperties=HEADER_FP,
        pad=16,
    )

    style_axis(ax)
    return legend_handles



def make_figure(df):
    required_cols = {
        "perturbation_type",
        "level",
        "representation",
        "mean_accuracy_drop",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Summary file is missing required columns: {missing}")

    df = df.copy()
    df = df[df["representation"].isin(REPRESENTATION_ORDER)]
    df = df[df["perturbation_type"].isin(PANEL_ORDER)]

    fig = plt.figure(figsize=(13.2, 8.8))
    fig.patch.set_facecolor(BG_COLOR)

    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.05])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])   # spans whole bottom row

    axes_with_names = [
        (ax1, "Translation only"),
        (ax2, "Scale only"),
        (ax3, "Translation + scale"),
    ]

    all_handles = {}

    for ax, panel_name in axes_with_names:
        panel_df = df[df["perturbation_type"] == panel_name].copy()

        if panel_df.empty:
            raise ValueError(f"No data found for perturbation type: {panel_name}")

        handles = add_panel(ax, panel_name, panel_df)
        all_handles.update(handles)

    fig.suptitle(
        "PERTURBATION TYPE ROBUSTNESS",
        x=0.5,
        y=0.975,
        ha="center",
        va="top",
        fontsize=22,
        color=TEXT_BLUE,
        fontproperties=TITLE_FP,
    )

    fig.text(
        0.045,
        0.50,
        "Accuracy drop from clean baseline (%)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=15,
        color=TEXT_BLUE,
        fontproperties=LABEL_FP,
    )

    fig.text(
        0.52,
        0.065,
        "Perturbation level",
        ha="center",
        va="center",
        fontsize=15,
        color=TEXT_BLUE,
        fontproperties=LABEL_FP,
    )

    legend_handles = [
        all_handles[rep]
        for rep in REPRESENTATION_ORDER
        if rep in all_handles
    ]
    legend_labels = [
        rep
        for rep in REPRESENTATION_ORDER
        if rep in all_handles
    ]

    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=3,
        frameon=False,
        prop=LABEL_FP,
        handlelength=2.4,
        columnspacing=1.8,
        labelspacing=0.8,
    )

    for text in legend.get_texts():
        text.set_color(TEXT_BLUE)

    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        top=0.80,
        bottom=0.14,
        wspace=0.20,
        hspace=0.48,
    )

    save_figure(fig, "perturbation_type_robustness_styled")



if __name__ == "__main__":
    summary_df = load_summary()
    make_figure(summary_df)

    print("\nDone.")
    print("Saved figure to final_results_eda/")