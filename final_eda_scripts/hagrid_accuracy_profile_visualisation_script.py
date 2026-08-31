from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm



PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "final_results_eda"
OUTPUT_DIR.mkdir(exist_ok=True)

COMBINED_RESULTS_CANDIDATES = [
    PROJECT_ROOT / "final_results_eda" / "combined_accuracy_results.csv",
    PROJECT_ROOT / "final_eda" / "combined_accuracy_results.csv",
]



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
        "folder": PROJECT_ROOT / "kaggle_random_forest_outputs",
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

REP_DISPLAY = {
    "XY": "XY",
    "XYZ": "XYZ",
    "Translated": "Translated",
    "Scaled": "Scaled",
    "Distance": "Distance",
    "Normalized distance": "Norm Dist",
    "Angles": "Angles",
    "Hybrid": "Hybrid",
}

REP_SHORT = {
    "XY": "XY",
    "XYZ": "XYZ",
    "Translated": "Trans.",
    "Scaled": "Scaled",
    "Distance": "Distance",
    "Normalized distance": "Norm Dist",
    "Angles": "Angles",
    "Hybrid": "Hybrid",
}

DATASET_ORDER = [
    "Self ASL",
    "SignAlphaSet",
    "HaGRID",
]

CLASSIFIER_ORDER = [
    "Logistic",
    "Random Forest",
]

CLASSIFIER_DISPLAY = {
    "Logistic": "Multinomial logistic",
    "Random Forest": "Random forest",
}



TEXT_BLUE = "#6fb7b6"
DARK_NAVY = "#102A43"

LOGISTIC_COLOR = "#6fb7b6"
RF_COLOR = "#7ecb80"


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


ROW_PASTELS = [
    "#EEF3F8",  # Hybrid
    "#dcefea",  # Norm. distance
    "#F9EDF4",  # Distance
    "#e4efdc",  # Scaled
    "#F8F2EB",  # Translated
    "#F3F1FA",  # Angles
    "#fffbe6",  # XYZ
    "#FDF1F0",  # XY
]

REPRESENTATION_LABEL_COLORS = {
    "Hybrid": ROW_LABEL_COLORS[0],
    "Normalized distance": ROW_LABEL_COLORS[1],
    "Distance": ROW_LABEL_COLORS[2],
    "Scaled": ROW_LABEL_COLORS[3],
    "Translated": ROW_LABEL_COLORS[4],
    "Angles": ROW_LABEL_COLORS[5],
    "XYZ": ROW_LABEL_COLORS[6],
    "XY": ROW_LABEL_COLORS[7],
}

REPRESENTATION_PASTELS = {
    "Hybrid": ROW_PASTELS[0],
    "Normalized distance": ROW_PASTELS[1],
    "Distance": ROW_PASTELS[2],
    "Scaled": ROW_PASTELS[3],
    "Translated": ROW_PASTELS[4],
    "Angles": ROW_PASTELS[5],
    "XYZ": ROW_PASTELS[6],
    "XY": ROW_PASTELS[7],
}


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
    return (
        str(name)
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )


def collect_font_files():
    files = []

    for font_dir in FONT_DIRS:
        if font_dir.exists():
            files.extend(list(font_dir.rglob("*.ttf")))
            files.extend(list(font_dir.rglob("*.otf")))

    try:
        files.extend(
            fm.findSystemFonts(
                fontpaths=None,
                fontext="ttf",
            )
        )

        files.extend(
            fm.findSystemFonts(
                fontpaths=None,
                fontext="otf",
            )
        )

    except Exception:
        pass

    unique = []
    seen = set()

    for path in files:
        path = Path(path)
        path_string = str(path)

        if path_string not in seen:
            unique.append(path)
            seen.add(path_string)

    return unique


def choose_font(
    candidate_names,
    fallback_family="DejaVu Sans",
    weight="normal",
):
    font_files = collect_font_files()
    font_records = []

    for path in font_files:
        try:
            font_name = fm.FontProperties(
                fname=str(path)
            ).get_name()

            font_records.append(
                (
                    path,
                    font_name,
                    normalize_font_name(font_name),
                )
            )

        except Exception:
            continue

    candidate_norms = [
        normalize_font_name(name)
        for name in candidate_names
    ]

    for candidate in candidate_norms:
        for path, font_name, normalized_name in font_records:
            if normalized_name == candidate:
                fm.fontManager.addfont(str(path))

                print(
                    f"Using font: {font_name} -> {path}"
                )

                return fm.FontProperties(
                    fname=str(path),
                    weight=weight,
                )

    for candidate in candidate_norms:
        for path, font_name, normalized_name in font_records:
            if (
                candidate in normalized_name
                or normalized_name in candidate
            ):
                fm.fontManager.addfont(str(path))

                print(
                    f"Using font: {font_name} -> {path}"
                )

                return fm.FontProperties(
                    fname=str(path),
                    weight=weight,
                )

    print(
        f"Requested fonts not found. "
        f"Using fallback: {fallback_family}"
    )

    return fm.FontProperties(
        family=fallback_family,
        weight=weight,
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



def normalize_text(s):
    return (
        str(s)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


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

    if any(
        key in s
        for key in [
            "normalized_distances",
            "normalized_distance",
            "normalized_dist",
            "norm_dist",
            "normdist",
        ]
    ):
        return "Normalized distance"

    if any(
        key in s
        for key in [
            "translated_xy",
            "translated",
            "translation",
            "trans",
        ]
    ):
        return "Translated"

    if any(
        key in s
        for key in [
            "scaled_xy",
            "scaled",
            "scale",
        ]
    ):
        return "Scaled"

    if any(
        key in s
        for key in [
            "distances_xy",
            "pairwise_distance",
            "pairwise_dist",
            "distance",
            "dist",
        ]
    ):
        return "Distance"

    if any(
        key in s
        for key in [
            "angles_xy",
            "angle",
            "angles",
        ]
    ):
        return "Angles"

    if "hybrid" in s:
        return "Hybrid"

    if any(
        key in s
        for key in [
            "raw_xyz",
            "xyz",
            "3d",
        ]
    ):
        return "XYZ"

    if any(
        key in s
        for key in [
            "raw_xy",
            "xy",
            "coordinates",
        ]
    ):
        return "XY"

    return None


def ordered_representations(df):
    present = list(
        df["representation"]
        .astype(str)
        .dropna()
        .unique()
    )

    ordered = [
        representation
        for representation in REPRESENTATION_ORDER
        if representation in present
    ]

    ordered += [
        representation
        for representation in present
        if representation not in ordered
    ]

    return ordered


def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(
        axis="both",
        colors=TEXT_BLUE,
        length=0,
        pad=6,
    )

    for label in ax.get_yticklabels():
        label.set_fontproperties(LABEL_FP)
        label.set_color(TEXT_BLUE)

    ax.grid(False)

    ax.set_axisbelow(True)


def set_axis_text(
    ax,
    xlabel=None,
    ylabel=None,
):
    if xlabel is not None:
        ax.set_xlabel(
            xlabel,
            fontproperties=LABEL_FP,
            fontsize=12,
            color=TEXT_BLUE,
            labelpad=10,
        )

    if ylabel is not None:
        ax.set_ylabel(
            ylabel,
            fontproperties=LABEL_FP,
            fontsize=12,
            color=TEXT_BLUE,
            labelpad=10,
        )


def focused_limits(
    values,
    pad=1.5,
    min_span=8,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        ~np.isnan(values)
    ]

    if len(values) == 0:
        return 0, 100

    lower = values.min() - pad
    upper = values.max() + pad

    if upper - lower < min_span:
        midpoint = 0.5 * (lower + upper)

        lower = midpoint - min_span / 2
        upper = midpoint + min_span / 2

    lower = max(0, lower)
    upper = min(100, upper)

    return lower, upper


def save_figure(
    fig,
    basename,
):
    png_path = OUTPUT_DIR / f"{basename}.png"
    pdf_path = OUTPUT_DIR / f"{basename}.pdf"
    svg_path = OUTPUT_DIR / f"{basename}.svg"

    fig.savefig(
        png_path,
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"Saved: {png_path}")

    try:
        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            facecolor="white",
        )

        print(f"Saved: {pdf_path}")

    except Exception as error:
        print(
            f"\nPDF save failed for {basename}."
        )
        print(f"Error: {error}")
        print("Saving SVG instead.")

        try:
            fig.savefig(
                svg_path,
                bbox_inches="tight",
                facecolor="white",
            )

            print(f"Saved: {svg_path}")

        except Exception as svg_error:
            print("\nSVG save also failed.")
            print(f"Error: {svg_error}")

    plt.close(fig)



def parse_all_representations_summary(
    csv_path,
    dataset,
    classifier,
):
    rows = []

    try:
        df = pd.read_csv(csv_path)

    except Exception:
        return rows

    if (
        "representation" not in df.columns
        or "test_accuracy" not in df.columns
    ):
        return rows

    for _, row in df.iterrows():
        representation = infer_representation_from_text(
            row["representation"]
        )

        test_accuracy = parse_numeric(
            row.get(
                "test_accuracy",
                np.nan,
            )
        )

        cv_accuracy = parse_numeric(
            row.get(
                "cv_best_accuracy",
                np.nan,
            )
        )

        if (
            representation is None
            or np.isnan(test_accuracy)
        ):
            continue

        rows.append({
            "dataset": dataset,
            "classifier": classifier,
            "representation": representation,
            "test_accuracy": test_accuracy,
            "cv_accuracy": cv_accuracy,
            "source_file": str(csv_path),
        })

    return rows


def parse_single_rep_summary(
    csv_path,
    dataset,
    classifier,
):
    rows = []

    try:
        df = pd.read_csv(csv_path)

    except Exception:
        return rows

    if not {"Metric", "Value"}.issubset(
        set(df.columns)
    ):
        return rows

    representation = None
    test_accuracy = np.nan
    cv_accuracy = np.nan

    for _, row in df.iterrows():
        metric = normalize_text(
            row["Metric"]
        )

        value = row["Value"]

        if metric == "representation":
            representation = infer_representation_from_text(
                value
            )

        elif metric == "test_accuracy":
            test_accuracy = parse_numeric(value)

        elif metric in [
            "best_cv_accuracy",
            "cv_best_accuracy",
            "cross_validation_accuracy",
            "cv_accuracy",
        ]:
            cv_accuracy = parse_numeric(value)

    if representation is None:
        representation = infer_representation_from_text(
            csv_path.parent.name
        )

    if (
        representation is not None
        and not np.isnan(test_accuracy)
    ):
        rows.append({
            "dataset": dataset,
            "classifier": classifier,
            "representation": representation,
            "test_accuracy": test_accuracy,
            "cv_accuracy": cv_accuracy,
            "source_file": str(csv_path),
        })

    return rows


def load_folder_results(
    dataset,
    classifier,
    folder,
):
    rows = []

    if not folder.exists():
        print(
            f"Missing folder, skipped: {folder}"
        )
        return rows

    all_summary_files = list(
        folder.glob(
            "*all_representations_summary.csv"
        )
    )

    if all_summary_files:
        for csv_path in all_summary_files:
            rows.extend(
                parse_all_representations_summary(
                    csv_path,
                    dataset,
                    classifier,
                )
            )

        if rows:
            print(
                "Loaded all-representation summary "
                f"from: {folder}"
            )
            return rows

    summary_files = list(
        folder.rglob("*summary.csv")
    )

    summary_files = [
        path
        for path in summary_files
        if "all_representations_summary"
        not in path.name
    ]

    for csv_path in summary_files:
        rows.extend(
            parse_single_rep_summary(
                csv_path,
                dataset,
                classifier,
            )
        )

    if rows:
        print(
            "Loaded per-representation summaries "
            f"from: {folder}"
        )

    else:
        print(
            f"No usable summaries found in: {folder}"
        )

    return rows


def standardize_results_dataframe(df):
    rename_map = {}

    if (
        "accuracy" in df.columns
        and "test_accuracy" not in df.columns
    ):
        rename_map["accuracy"] = "test_accuracy"

    df = df.rename(
        columns=rename_map
    )

    required = {
        "dataset",
        "classifier",
        "representation",
        "test_accuracy",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Combined results file is missing "
            f"columns: {missing}"
        )

    df["test_accuracy"] = (
        df["test_accuracy"]
        .apply(parse_numeric)
        .astype(float)
    )

    if "cv_accuracy" not in df.columns:
        df["cv_accuracy"] = np.nan

    else:
        df["cv_accuracy"] = (
            df["cv_accuracy"]
            .apply(parse_numeric)
        )

    df["dataset"] = (
        df["dataset"]
        .astype(str)
    )

    df["classifier"] = (
        df["classifier"]
        .astype(str)
    )

    df["representation"] = (
        df["representation"]
        .astype(str)
    )

    classifier_fix = {
        "RF": "Random Forest",
        "random_forest": "Random Forest",
        "RandomForest": "Random Forest",
        "Logistic Regression": "Logistic",
        "Multinomial Logistic": "Logistic",
    }

    df["classifier"] = (
        df["classifier"]
        .replace(classifier_fix)
    )

    df = (
        df.sort_values(
            "test_accuracy",
            ascending=False,
        )
        .drop_duplicates([
            "dataset",
            "classifier",
            "representation",
        ])
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

    df = (
        df.sort_values([
            "dataset",
            "classifier",
            "representation",
        ])
        .reset_index(drop=True)
    )

    return df


def build_results_dataframe():
    for candidate in COMBINED_RESULTS_CANDIDATES:
        if candidate.exists():
            print(
                "Loading existing combined results "
                f"from: {candidate}"
            )

            return standardize_results_dataframe(
                pd.read_csv(candidate)
            )

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
        raise RuntimeError(
            "No results loaded. Check folder names "
            "or summary CSV files."
        )

    df = pd.DataFrame(rows)
    df = standardize_results_dataframe(df)

    output_csv = (
        OUTPUT_DIR
        / "combined_accuracy_results.csv"
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    print("\nCombined results:")

    print(
        df[
            [
                "dataset",
                "classifier",
                "representation",
                "test_accuracy",
            ]
        ]
    )

    print(
        f"\nSaved combined results to: "
        f"{output_csv}"
    )

    return df



def plot_hagrid_accuracy_profile(df):
    hagrid_df = df[
        df["dataset"].astype(str) == "HaGRID"
    ].copy()

    if hagrid_df.empty:
        print(
            "No HaGRID results found. "
            "Skipping profile plot."
        )
        return

    representations = ordered_representations(
        hagrid_df
    )

    pivot = (
        hagrid_df
        .pivot_table(
            index="representation",
            columns="classifier",
            values="test_accuracy",
            aggfunc="max",
            observed=False,
        )
        .reindex(index=representations)
    )

    x = (
        np.arange(len(representations))
        * 1.12
    )

    x_labels = [
        REP_SHORT.get(
            representation,
            representation,
        )
        for representation in representations
    ]

    fig, ax = plt.subplots(
        figsize=(9.8, 4.6)
    )

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")


    for x_position, representation in zip(
        x,
        representations,
    ):
        ax.axvspan(
            x_position - 0.54,
            x_position + 0.54,
            facecolor=REPRESENTATION_PASTELS.get(
                representation,
                "#f5f5f5",
            ),
            edgecolor="none",
            alpha=1.0,
            zorder=0,
        )

    specs = {
        "Logistic": {
            "label": "MULTINOMIAL LOGISTIC",
            "color": LOGISTIC_COLOR,
            "marker": "o",
            "linestyle": "-",
        },
        "Random Forest": {
            "label": "RANDOM FOREST",
            "color": RF_COLOR,
            "marker": "s",
            "linestyle": "--",
        },
    }

    all_values = []

    for classifier in CLASSIFIER_ORDER:
        if classifier not in pivot.columns:
            continue

        values = (
            pivot[classifier]
            .values
            .astype(float)
        )

        all_values.extend(
            values[~np.isnan(values)]
        )

        spec = specs[classifier]

        ax.plot(
            x,
            values,
            color=spec["color"],
            marker=spec["marker"],
            linestyle=spec["linestyle"],
            linewidth=2.4,
            markersize=7,
            markeredgewidth=0,
            label=spec["label"],
            zorder=3,
        )

        for x_position, value in zip(
            x,
            values,
        ):
            if not np.isnan(value):
                ax.text(
                    x_position,
                    value + 0.55,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.3,
                    fontproperties=BOLD_FP,
                    color=DARK_NAVY,
                    zorder=4,
                )

    lower, upper = focused_limits(
        all_values,
        pad=3,
        min_span=18,
    )

    ax.set_ylim(
        lower,
        upper,
    )

    ax.set_xlim(
        x[0] - 0.60,
        x[-1] + 0.60,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    style_axis(ax)

    for label, representation in zip(
        ax.get_xticklabels(),
        representations,
    ):
        label.set_fontproperties(BOLD_FP)
        label.set_fontsize(10.5)

        label.set_color(
            REPRESENTATION_LABEL_COLORS.get(
                representation,
                TEXT_BLUE,
            )
        )

    set_axis_text(
        ax,
        xlabel="Feature representation",
        ylabel="Held-out test accuracy (%)",
    )

    ax.set_title(
        "HAGRID ACCURACY PROFILE",
        fontsize=20,
        fontproperties=BOLD_FP,
        color=TEXT_BLUE,
        pad=74,
    )

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=False,
        handlelength=3.0,
        handletextpad=0.8,
        columnspacing=3.2,
        borderaxespad=0,
    )

    for legend_text, classifier in zip(
        legend.get_texts(),
        CLASSIFIER_ORDER,
    ):
        legend_text.set_fontproperties(BOLD_FP)
        legend_text.set_fontsize(11)
        legend_text.set_color(
            specs[classifier]["color"]
        )

    fig.subplots_adjust(
        left=0.11,
        right=0.97,
        bottom=0.21,
        top=0.70,
    )

    save_figure(
        fig,
        "hagrid_accuracy_profile_styled_further",
    )



if __name__ == "__main__":
    results = build_results_dataframe()

    plot_hagrid_accuracy_profile(
        results
    )

    print("\nDone. Figure saved in:")
    print(OUTPUT_DIR)