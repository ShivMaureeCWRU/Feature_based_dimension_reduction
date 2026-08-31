from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle



PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "final_results_eda"
OUTPUT_DIR.mkdir(exist_ok=True)



OUTPUT_BASENAME = "results_accuracy_heatmap_styled_further"



RESULT_FOLDERS = [
    {
        "dataset": "Self ASL",
        "classifier": "Logistic",
        "folder": PROJECT_ROOT / "logistic_regression_outputs",
    },
    {
        "dataset": "Self ASL",
        "classifier": "Random Forest",
        "folder": PROJECT_ROOT / "random_forest_outputs",
    },
    {
        "dataset": "SignAlphaSet",
        "classifier": "Logistic",
        "folder": PROJECT_ROOT / "SignAlphaSet_logistic_regression_outputs",
    },
    {
        "dataset": "SignAlphaSet",
        "classifier": "Random Forest",
        "folder": PROJECT_ROOT / "SignAlphaSet_random_forest_outputs",
    },
    {
        "dataset": "HaGRID",
        "classifier": "Logistic",
        "folder": PROJECT_ROOT / "hagrid_100_logistic_regression_outputs",
    },
    {
        "dataset": "HaGRID",
        "classifier": "Random Forest",
        "folder": PROJECT_ROOT / "hagrid_100_random_forest_outputs",
    },
]


ROW_LABEL_COLORS = [
    "#102A43",  # Hybrid
    "#2F6A5A",  # Normalized distance
    "#75244d",  # Distance
    "#4D6F2F",  # Scaled
    "#8A5622",  # Translated
    "#5A4A82",  # Angles
    "#756300",  # XYZ
    "#F3AAA5",  # XY
]
REPRESENTATION_COLORS = {
    "Hybrid": ROW_LABEL_COLORS[0],
    "Normalized distance": ROW_LABEL_COLORS[1],
    "Distance": ROW_LABEL_COLORS[2],
    "Scaled": ROW_LABEL_COLORS[3],
    "Translated": ROW_LABEL_COLORS[4],
    "Angles": ROW_LABEL_COLORS[5],
    "XYZ": ROW_LABEL_COLORS[6],
    "XY": ROW_LABEL_COLORS[7],
}
REPRESENTATION_ORDER = [
    "XY",
    "XYZ",
    "Translated",
    "Scaled",
    "Distance",
    "Normalized distance",
    "Angles",
    "Hybrid",
]

DATASET_ORDER = ["Self ASL", "SignAlphaSet", "HaGRID"]
CLASSIFIER_ORDER = ["Logistic", "Random Forest"]

DATASET_DISPLAY = {
    "Self ASL": "SELF ASL",
    "SignAlphaSet": "SIGNALPHA",
    "HaGRID": "HAGRID",
}

CLASSIFIER_DISPLAY = {
    "Logistic": "MULTINOMIAL LOGISTIC",
    "Random Forest": "RANDOM FOREST",
}



TEXT_BLUE = "#6fb7b6"      # labels, model names, dataset names
CELL_NAVY = "#102A43"      # numbers inside cells

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "none",
    "svg.fonttype": "path",
})


# It searches for rounded/geometric fonts first.

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
        path_str = str(path)

        if path_str not in seen:
            unique.append(path)
            seen.add(path_str)

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


NUMBER_FP = choose_font(
    candidate_names=[
        "Montserrat ExtraBold",
        "Montserrat Black",
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

LABEL_FP = choose_font(
    candidate_names=[
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



def normalize_text(s):
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")


def parse_numeric(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, str):
        value = value.strip().replace("%", "")

    try:
        x = float(value)
    except Exception:
        return np.nan

    if 0 <= x <= 1:
        x *= 100.0

    return x


def infer_representation_from_text(text):
    s = normalize_text(text)

    if any(key in s for key in ["normalized_distances", "normalized_distance", "normalized_dist", "norm_dist", "normdist"]):
        return "Normalized distance"

    if any(key in s for key in ["translated_xy", "translated", "translation", "trans"]):
        return "Translated"

    if any(key in s for key in ["scaled_xy", "scaled", "scale"]):
        return "Scaled"

    if any(key in s for key in ["distances_xy", "pairwise_distance", "pairwise_dist", "distance", "dist"]):
        return "Distance"

    if any(key in s for key in ["angles_xy", "angle", "angles"]):
        return "Angles"

    if "hybrid" in s:
        return "Hybrid"

    if any(key in s for key in ["raw_xyz", "xyz", "3d"]):
        return "XYZ"

    if any(key in s for key in ["raw_xy", "xy", "coordinates"]):
        return "XY"

    return None



def parse_all_representations_summary(csv_path, dataset, classifier):
    rows = []

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return rows

    if "representation" not in df.columns or "test_accuracy" not in df.columns:
        return rows

    for _, row in df.iterrows():
        rep = infer_representation_from_text(row["representation"])

        if rep is None:
            continue

        test_acc = parse_numeric(row.get("test_accuracy", np.nan))

        if np.isnan(test_acc):
            continue

        rows.append({
            "dataset": dataset,
            "classifier": classifier,
            "representation": rep,
            "test_accuracy": test_acc,
            "source_file": str(csv_path),
        })

    return rows


def parse_single_rep_summary(csv_path, dataset, classifier):
    rows = []

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return rows

    if not {"Metric", "Value"}.issubset(set(df.columns)):
        return rows

    rep = None
    test_acc = np.nan

    for _, row in df.iterrows():
        metric = normalize_text(row["Metric"])
        value = row["Value"]

        if metric == "representation":
            rep = infer_representation_from_text(value)

        elif metric == "test_accuracy":
            test_acc = parse_numeric(value)

    if rep is None:
        rep = infer_representation_from_text(csv_path.parent.name)

    if rep is not None and not np.isnan(test_acc):
        rows.append({
            "dataset": dataset,
            "classifier": classifier,
            "representation": rep,
            "test_accuracy": test_acc,
            "source_file": str(csv_path),
        })

    return rows


def load_folder_results(dataset, classifier, folder):
    rows = []

    if not folder.exists():
        print(f"Missing folder, skipped: {folder}")
        return rows

    all_summary_files = list(folder.glob("*all_representations_summary.csv"))

    if all_summary_files:
        for csv_path in all_summary_files:
            rows.extend(parse_all_representations_summary(csv_path, dataset, classifier))

        if rows:
            print(f"Loaded all-representation summary from: {folder}")
            return rows

    summary_files = list(folder.rglob("*summary.csv"))
    summary_files = [
        path for path in summary_files
        if "all_representations_summary" not in path.name
    ]

    for csv_path in summary_files:
        rows.extend(parse_single_rep_summary(csv_path, dataset, classifier))

    if rows:
        print(f"Loaded per-representation summaries from: {folder}")
    else:
        print(f"No usable summaries found in: {folder}")

    return rows


def build_results_dataframe():
    rows = []

    for item in RESULT_FOLDERS:
        rows.extend(
            load_folder_results(
                dataset=item["dataset"],
                classifier=item["classifier"],
                folder=item["folder"],
            )
        )

    if not rows:
        raise RuntimeError("No results loaded. Check folder names or summary CSV files.")

    df = pd.DataFrame(rows)
    df["test_accuracy"] = df["test_accuracy"].astype(float)

    df = (
        df.sort_values("test_accuracy", ascending=False)
        .drop_duplicates(["dataset", "classifier", "representation"])
        .reset_index(drop=True)
    )

    df["representation"] = pd.Categorical(
        df["representation"],
        categories=REPRESENTATION_ORDER,
        ordered=True,
    )

    df["dataset"] = pd.Categorical(
        df["dataset"],
        categories=DATASET_ORDER,
        ordered=True,
    )

    df["classifier"] = pd.Categorical(
        df["classifier"],
        categories=CLASSIFIER_ORDER,
        ordered=True,
    )

    df = df.sort_values(["representation", "classifier", "dataset"]).reset_index(drop=True)

    output_csv = OUTPUT_DIR / "combined_accuracy_results.csv"
    df.to_csv(output_csv, index=False)

    print("\nCombined results:")
    print(df[["dataset", "classifier", "representation", "test_accuracy"]])
    print(f"\nSaved combined results to: {output_csv}")

    return df


# Green mostly concentrated from 90 to 100

def make_pastel_trafficlight_cmap():
    return LinearSegmentedColormap.from_list(
        "pastel_trafficlight",
        [
            (0.00, "#f4b5b0"),  # 50: pastel red
            (0.20, "#f7c8b0"),  # 60: peach
            (0.40, "#f8dda5"),  # 70: warm yellow
            (0.60, "#f5ecaa"),  # 80: pale yellow
            (0.78, "#dff0b3"),  # 89: yellow-green
            (0.80, "#d2ecb2"),  # 90: green begins
            (0.90, "#aee0ad"),  # 95
            (1.00, "#7ecb80"),  # 100
        ]
    )



def plot_structured_heatmap(df):
    column_pairs = []

    for classifier in CLASSIFIER_ORDER:
        for dataset in DATASET_ORDER:
            mask = (df["classifier"] == classifier) & (df["dataset"] == dataset)
            if mask.any():
                column_pairs.append((classifier, dataset))

    reps = [
        r for r in REPRESENTATION_ORDER
        if r in df["representation"].astype(str).unique()
    ]

    values = []

    for rep in reps:
        row = []

        for classifier, dataset in column_pairs:
            sub = df[
                (df["representation"].astype(str) == rep) &
                (df["classifier"].astype(str) == classifier) &
                (df["dataset"].astype(str) == dataset)
            ]

            if len(sub) == 0:
                row.append(np.nan)
            else:
                row.append(float(sub["test_accuracy"].iloc[0]))

        values.append(row)

    values = np.array(values, dtype=float)

    cmap = make_pastel_trafficlight_cmap()
    norm = Normalize(vmin=50, vmax=100)

    model_gap = 1.18

    x_centers = []

    for j in range(len(column_pairs)):
        if j < 3:
            x_centers.append(j)
        else:
            x_centers.append(j + model_gap)

    x_centers = np.array(x_centers, dtype=float)
    y_centers = np.arange(len(reps), dtype=float)

    cell_width = 0.94
    cell_height = 0.94

    fig, ax = plt.subplots(figsize=(10.9, 6.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, y in enumerate(y_centers):
        for j, x in enumerate(x_centers):
            val = values[i, j]

            if np.isnan(val):
                facecolor = "#ffffff"
            else:
                facecolor = cmap(norm(val))

            rect = Rectangle(
                (x - cell_width / 2, y - cell_height / 2),
                cell_width,
                cell_height,
                facecolor=facecolor,
                edgecolor="white",
                linewidth=3.0,
            )

            ax.add_patch(rect)

            if not np.isnan(val):
                ax.text(
                    x,
                    y,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=17,
                    fontproperties=NUMBER_FP,
                    color=CELL_NAVY,
                )

    ax.set_yticks(y_centers)
    ax.set_yticklabels(reps)

    for label in ax.get_yticklabels():
        rep = label.get_text()
        label.set_fontproperties(BOLD_FP)
        label.set_fontsize(15)
        label.set_color(
            REPRESENTATION_COLORS.get(rep, TEXT_BLUE)
        )

    ax.set_xticks([])
    ax.set_xticklabels([])

    model_to_cols = {}

    for idx, (classifier, dataset) in enumerate(column_pairs):
        model_to_cols.setdefault(classifier, []).append(idx)

    for classifier, cols in model_to_cols.items():
        center = np.mean(x_centers[cols])

        ax.text(
            center,
            len(reps) + 0.80,
            CLASSIFIER_DISPLAY[classifier],
            ha="center",
            va="top",
            fontsize=15,
            fontproperties=LABEL_FP,
            color=TEXT_BLUE,
        )

    for j, (_, dataset) in enumerate(column_pairs):
        ax.text(
            x_centers[j],
            len(reps) + 0.34,
            DATASET_DISPLAY[dataset],
            ha="center",
            va="top",
            fontsize=11.5,
            fontproperties=LABEL_FP,
            color=TEXT_BLUE,
        )

    left_label_x = x_centers[0] - 0.95

    ax.text(
        left_label_x,
        len(reps) + 0.34,
        "DATASET",
        ha="right",
        va="top",
        fontsize=10.5,
        fontproperties=LABEL_FP,
        color=TEXT_BLUE,
    )

    ax.text(
        left_label_x,
        len(reps) + 0.80,
        "MODEL USED",
        ha="right",
        va="top",
        fontsize=10.5,
        fontproperties=LABEL_FP,
        color=TEXT_BLUE,
    )

    ax.set_xlim(x_centers[0] - 0.65, x_centers[-1] + 0.65)
    ax.set_ylim(len(reps) + 1.36, -0.5)

    ax.set_xlabel("")
    ax.set_ylabel("")

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="both", length=0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.045)
    cbar.ax.set_title(
        "ACCURACY (%)",
        fontproperties=LABEL_FP,
        fontsize=13,
        color=TEXT_BLUE,
        pad=14,
    )

    cbar.set_ticks([50, 60, 70, 80, 90, 100])

    for tick in cbar.ax.get_yticklabels():
        tick.set_fontproperties(NUMBER_FP)
        tick.set_fontsize(11)
        tick.set_color(TEXT_BLUE)

    cbar.outline.set_visible(False)

    fig.suptitle(
        "ACCURACY HEATMAP ACROSS DATASETS AND MODELS",
        x=0.5,
        y=0.975,
        ha="center",
        va="top",
        fontsize=21,
        fontproperties=TITLE_FP,
        color=TEXT_BLUE,
    )

    plt.subplots_adjust(left=0.22, right=0.88, bottom=0.22, top=0.86)

    png_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.png"
    pdf_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf"
    svg_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.svg"

    fig.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white")
    print(f"Saved: {png_path}")

    try:
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {pdf_path}")
    except Exception as e:
        print("\nPDF save failed because of a font embedding issue.")
        print(f"Error: {e}")
        print("Saving SVG instead.")

        try:
            fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
            print(f"Saved: {svg_path}")
        except Exception as e2:
            print("\nSVG save also failed.")
            print(f"Error: {e2}")

    plt.close(fig)



if __name__ == "__main__":
    df = build_results_dataframe()
    plot_structured_heatmap(df)

    print("\nDone. Heatmap saved in:")
    print(OUTPUT_DIR)