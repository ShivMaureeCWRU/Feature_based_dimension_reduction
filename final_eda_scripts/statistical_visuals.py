from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm



PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "final_results_eda"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_BASENAME = "hagrid_repeated_cv_summary_styled"



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
    possible_zips += list((PROJECT_ROOT / "final_eda").glob("*hagrid*statistical_tests*.zip")) if (PROJECT_ROOT / "final_eda").exists() else []
    possible_zips += list((PROJECT_ROOT / "final_results_eda").glob("*hagrid*statistical_tests*.zip")) if (PROJECT_ROOT / "final_results_eda").exists() else []

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
    "normalized_distances_xy": "Norm. distance",
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

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "none",
    "svg.fonttype": "path",
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
        out.groupby("representation")["mean_accuracy"]
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

    ax.tick_params(axis="both", colors=TEXT_BLUE, length=0, pad=6)

    for label in ax.get_xticklabels():
        label.set_fontproperties(LABEL_FP)
        label.set_color(TEXT_BLUE)
        label.set_fontsize(11)

    for label in ax.get_yticklabels():
        label.set_fontproperties(LABEL_FP)
        label.set_color(TEXT_BLUE)
        label.set_fontsize(11)

    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.9, alpha=0.9)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)


def save_figure(fig):
    png_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.png"
    pdf_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf"
    svg_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.svg"

    fig.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white")
    print(f"Saved: {png_path}")

    try:
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {pdf_path}")
    except Exception as e:
        print("\nPDF save failed because of font embedding.")
        print(f"Error: {e}")
        print("Saving SVG instead.")

        try:
            fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
            print(f"Saved: {svg_path}")
        except Exception as e2:
            print("\nSVG save also failed.")
            print(f"Error: {e2}")

    plt.close(fig)


def plot_hagrid_repeated_cv_summary(df, rep_order):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_base = np.arange(len(rep_order))
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

        ax.errorbar(
            means,
            y,
            xerr=stds,
            fmt=markers[classifier],
            markersize=8,
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

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=False,
        prop=LABEL_FP,
        handlelength=2.2,
        columnspacing=2.4,
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

    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.22, top=0.96)

    save_figure(fig)



if __name__ == "__main__":
    summary_df, representation_order = load_hagrid_summary()
    plot_hagrid_repeated_cv_summary(summary_df, representation_order)

    print("\nDone. Figure saved in:")
    print(OUTPUT_DIR)