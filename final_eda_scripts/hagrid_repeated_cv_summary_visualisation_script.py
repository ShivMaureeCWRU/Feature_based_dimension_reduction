from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm



PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "final_results_eda"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_BASENAME = "hagrid_repeated_cv_summary_styled_even_further"



EXTRACT_DIR = OUTPUT_DIR / "_extracted_hagrid_stats"


def find_hagrid_stats_root():
    possible_dirs = [
        PROJECT_ROOT / "hagrid_100_statistical_tests",
        PROJECT_ROOT / "final_results_eda" / "hagrid_100_statistical_tests",
        PROJECT_ROOT / "final_eda" / "hagrid_100_statistical_tests",
        EXTRACT_DIR / "hagrid_100_statistical_tests",
    ]

    for d in possible_dirs:
        if (d / "logistic_regression" / "representation_accuracy_summary.csv").exists():
            return d

    possible_zips = list(PROJECT_ROOT.glob("*hagrid*statistical_tests*.zip"))

    if (PROJECT_ROOT / "final_eda").exists():
        possible_zips += list((PROJECT_ROOT / "final_eda").glob("*hagrid*statistical_tests*.zip"))

    if (PROJECT_ROOT / "final_results_eda").exists():
        possible_zips += list((PROJECT_ROOT / "final_results_eda").glob("*hagrid*statistical_tests*.zip"))

    if possible_zips:
        zip_path = possible_zips[0]
        print(f"Extracting: {zip_path}")
        EXTRACT_DIR.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(EXTRACT_DIR)

        for d in possible_dirs:
            if (d / "logistic_regression" / "representation_accuracy_summary.csv").exists():
                return d

    raise FileNotFoundError(
        "Could not find HaGRID statistical test outputs. "
        "Expected folder hagrid_100_statistical_tests/ or a zip containing it."
    )



REP_MAP = {
    "raw_xy": "XY",
    "raw_xyz": "XYZ",
    "translated_xy": "Translated",
    "scaled_xy": "Scaled",
    "distances_xy": "Distance",
    "normalized_distances_xy": "Norm Dist",
    "angles_xy": "Angles",
    "hybrid": "Hybrid",
}

CLASSIFIER_FOLDERS = {
    "Multinomial logistic": "logistic_regression",
    "Random forest": "random_forest",
}

CLASSIFIER_ORDER = ["Multinomial logistic", "Random forest"]



TEXT_BLUE = "#6fb7b6"
DARK_NAVY = "#102A43"

LOGISTIC_COLOR = "#6fb7b6"
RF_COLOR = "#7ecb80"
GRID_COLOR = "#d8eeee"
BG_COLOR = "white"

ROW_PASTELS = [
    "#EEF3F8",  # Hybrid (navy)
    "#dcefea",  # Norm. distance (sage)
    "#F9EDF4",  # Distance (warm orange)
    "#e4efdc",  # Scaled (green)
    "#F8F2EB",  # Translated (brown)
    "#F3F1FA",  # Angles (purple)
    "#fffbe6",  # XYZ (olive)
    "#FDF1F0",  # XY (salmon)
]

ROW_LABEL_COLORS = [
    "#102A43",  #"#2C6F73"  # Hybrid (teal)
    "#2F6A5A", #"#4D6F2F"  # Norm. distance (green)
    "#75244d",  # Distance (warm brown)
    "#4D6F2F",# "#5A4A82",  # Scaled (purple)
    "#8A5622",# "#355F87",  # Translated (blue)
    "#5A4A82",# "#8B2330",  # Angles (rose)
    "#756300",  # XYZ (olive)
    "#F3AAA5",# "#2F6A5A",  # XY (sage)
]

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
        "Montserrat ExtraBold",
        "Montserrat SemiBold",
        "Montserrat Bold",
        "Montserrat",
        "Aptos Display",
        "Bahnschrift",
    ],
    fallback_family="DejaVu Sans",
    weight="bold",
)



def to_percent(x):
    x = float(x)

    if 0 <= x <= 1:
        return 100 * x

    return x


def load_hagrid_summary():
    stats_root = find_hagrid_stats_root()
    rows = []

    for classifier, folder_name in CLASSIFIER_FOLDERS.items():
        summary_path = stats_root / folder_name / "representation_accuracy_summary.csv"

        if not summary_path.exists():
            print(f"Missing summary file: {summary_path}")
            continue

        df = pd.read_csv(summary_path)

        for _, row in df.iterrows():
            raw_rep = str(row["representation"])
            rep = REP_MAP.get(raw_rep, raw_rep)

            rows.append({
                "classifier": classifier,
                "representation": rep,
                "mean_accuracy": to_percent(row["mean_accuracy"]),
                "std_accuracy": to_percent(row["std_accuracy"]),
                "min_accuracy": to_percent(row["min_accuracy"]),
                "max_accuracy": to_percent(row["max_accuracy"]),
            })

    out = pd.DataFrame(rows)

    if out.empty:
        raise RuntimeError("No statistical summary results were loaded.")

    order = (
        out
        .groupby("representation")["mean_accuracy"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    out["representation"] = pd.Categorical(
        out["representation"],
        categories=order,
        ordered=True,
    )

    out["classifier"] = pd.Categorical(
        out["classifier"],
        categories=CLASSIFIER_ORDER,
        ordered=True,
    )

    out = out.sort_values(["representation", "classifier"]).reset_index(drop=True)

    csv_path = OUTPUT_DIR / "hagrid_repeated_cv_summary.csv"
    out.to_csv(csv_path, index=False)
    print(f"Saved summary CSV: {csv_path}")

    return out, order



def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="both", length=0, pad=6)

    for label in ax.get_xticklabels():
        label.set_fontproperties(LABEL_FP)
        label.set_color(TEXT_BLUE)
        label.set_fontsize(11)


    ax.grid(False)
    ax.set_axisbelow(True)


def save_figure(fig):
    png_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.png"
    pdf_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf"
    svg_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.svg"

    fig.savefig(png_path, dpi=400, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Saved: {png_path}")

    try:
        fig.savefig(pdf_path, bbox_inches="tight", facecolor=BG_COLOR)
        print(f"Saved: {pdf_path}")
    except Exception as e:
        print("\nPDF save failed because of font embedding.")
        print(f"Error: {e}")
        print("Saving SVG instead.")

        try:
            fig.savefig(svg_path, bbox_inches="tight", facecolor=BG_COLOR)
            print(f"Saved: {svg_path}")
        except Exception as e2:
            print("\nSVG save also failed.")
            print(f"Error: {e2}")

    plt.close(fig)



def plot_hagrid_repeated_cv_summary(df, rep_order):
    fig, ax = plt.subplots(figsize=(9.2, 5.65))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    y_base = np.arange(len(rep_order))

    for i, _ in enumerate(rep_order):
        ax.axhspan(
            i - 0.48,
            i + 0.48,
            facecolor=ROW_PASTELS[i % len(ROW_PASTELS)],
            edgecolor="none",
            alpha=0.72,
            zorder=0,
        )

    offsets = {
        "Multinomial logistic": -0.16,
        "Random forest": 0.16,
    }

    colors = {
        "Multinomial logistic": LOGISTIC_COLOR,
        "Random forest": RF_COLOR,
    }

    markers = {
        "Multinomial logistic": "o",
        "Random forest": "s",
    }

    legend_handles = []
    legend_labels = []

    for classifier in CLASSIFIER_ORDER:
        sub = df[df["classifier"].astype(str) == classifier].copy()

        means = []
        stds = []

        for rep in rep_order:
            row = sub[sub["representation"].astype(str) == rep]

            if row.empty:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(float(row["mean_accuracy"].iloc[0]))
                stds.append(float(row["std_accuracy"].iloc[0]))

        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)
        y = y_base + offsets[classifier]

        handle = ax.errorbar(
            means,
            y,
            xerr=stds,
            fmt=markers[classifier],
            markersize=8.2,
            linewidth=0,
            elinewidth=2.0,
            capsize=4,
            capthick=1.8,
            color=colors[classifier],
            ecolor=colors[classifier],
            markeredgewidth=0,
            alpha=0.95,
            label=classifier,
        )

        legend_handles.append(handle)
        legend_labels.append(classifier)

        for xi, yi in zip(means, y):
            if not np.isnan(xi):
                ax.text(
                    xi + 0.35,
                    yi,
                    f"{xi:.1f}",
                    ha="left",
                    va="center",
                    fontsize=8.8,
                    fontproperties=BOLD_FP,
                    color=DARK_NAVY,
                )

    ax.set_yticks(y_base)
    ax.set_yticklabels(rep_order)

    for i, label in enumerate(ax.get_yticklabels()):
        label.set_color(ROW_LABEL_COLORS[i % len(ROW_LABEL_COLORS)])
        label.set_fontproperties(LABEL_FP)
        label.set_fontsize(16)

    ax.invert_yaxis()

    all_x = df["mean_accuracy"].values
    all_std = df["std_accuracy"].values

    xmin = max(50, np.nanmin(all_x - all_std) - 2.0)
    xmax = min(100, np.nanmax(all_x + all_std) + 4.0)

    ax.set_xlim(xmin, xmax)

    ax.set_xlabel(
        "Repeated cross-validation accuracy (%)",
        fontproperties=LABEL_FP,
        fontsize=13,
        color=TEXT_BLUE,
        labelpad=12,
    )

    ax.set_ylabel("")

    style_axis(ax)

    fig.suptitle(
        "REPEATED CROSS-VALIDATION ACCURACY",
        x=0.5,
        y=0.972,
        ha="center",
        va="top",
        fontsize=21,
        fontproperties=TITLE_FP,
        color=TEXT_BLUE,
    )

    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=2,
        frameon=False,
        prop=LABEL_FP,
        handlelength=2.8,
        columnspacing=3.0,
        labelspacing=0.8,
    )

    for text in legend.get_texts():
        text.set_color(TEXT_BLUE)

    ax.text(
        0.995,
        0.015,
        "mean ± 1 SD over repeated folds",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        fontproperties=LABEL_FP,
        color=TEXT_BLUE,
        alpha=0.9,
    )

    fig.subplots_adjust(
        left=0.23,
        right=0.965,
        bottom=0.18,
        top=0.765,
    )

    save_figure(fig)



if __name__ == "__main__":
    summary_df, representation_order = load_hagrid_summary()
    plot_hagrid_repeated_cv_summary(summary_df, representation_order)

    print("\nDone. Figure saved in:")
    print(OUTPUT_DIR)