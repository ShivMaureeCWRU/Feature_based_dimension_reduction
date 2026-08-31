from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


# Paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "final_results_eda"
OUTPUT_DIR.mkdir(exist_ok=True)

USE_CACHED_COMBINED_RESULTS = False

COMBINED_RESULTS_CANDIDATES = [
    PROJECT_ROOT / "final_results_eda" / "combined_accuracy_results.csv",
    PROJECT_ROOT / "final_eda" / "combined_accuracy_results.csv",
]


# Folder configuration

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


# Ordering and labels

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
    "Normalized distance": "Norm. distance",
    "Angles": "Angles",
    "Hybrid": "Hybrid",
}

REP_SHORT = {
    "XY": "XY",
    "XYZ": "XYZ",
    "Translated": "Trans.",
    "Scaled": "Scaled",
    "Distance": "Dist.",
    "Normalized distance": "Norm.\ndist.",
    "Angles": "Angles",
    "Hybrid": "Hybrid",
}

DATASET_ORDER = ["Self ASL", "SignAlphaSet", "HaGRID"]
CLASSIFIER_ORDER = ["Logistic", "Random Forest"]

CLASSIFIER_DISPLAY = {
    "Logistic": "Multinomial logistic",
    "Random Forest": "Random forest",
}


# Aesthetic settings

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


# Font selection

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


# Helpers

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

    if any(key in s for key in [
        "normalized_distances",
        "normalized_distance",
        "normalized_dist",
        "norm_dist",
        "normdist"
    ]):
        return "Normalized distance"

    if any(key in s for key in [
        "translated_xy",
        "translated",
        "translation",
        "trans"
    ]):
        return "Translated"

    if any(key in s for key in [
        "scaled_xy",
        "scaled",
        "scale"
    ]):
        return "Scaled"

    if any(key in s for key in [
        "distances_xy",
        "pairwise_distance",
        "pairwise_dist",
        "distance",
        "dist"
    ]):
        return "Distance"

    if any(key in s for key in [
        "angles_xy",
        "angle",
        "angles"
    ]):
        return "Angles"

    if "hybrid" in s:
        return "Hybrid"

    if any(key in s for key in [
        "raw_xyz",
        "xyz",
        "3d"
    ]):
        return "XYZ"

    if any(key in s for key in [
        "raw_xy",
        "xy",
        "coordinates"
    ]):
        return "XY"

    return None


def ordered_representations(df):
    present = list(df["representation"].astype(str).dropna().unique())
    ordered = [r for r in REPRESENTATION_ORDER if r in present]
    ordered += [r for r in present if r not in ordered]
    return ordered


def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="both", colors=TEXT_BLUE, length=0, pad=7)

    for label in ax.get_xticklabels():
        label.set_fontproperties(LABEL_FP)
        label.set_color(TEXT_BLUE)
        label.set_fontsize(10)

    for label in ax.get_yticklabels():
        label.set_fontproperties(LABEL_FP)
        label.set_color(TEXT_BLUE)
        label.set_fontsize(10)

    ax.grid(True, color=GRID_COLOR, linewidth=0.85, alpha=0.85)
    ax.set_axisbelow(True)


def set_axis_text(ax, xlabel=None, ylabel=None, title=None):
    if xlabel is not None:
        ax.set_xlabel(
            xlabel,
            fontproperties=LABEL_FP,
            fontsize=12,
            color=TEXT_BLUE,
            labelpad=12,
        )

    if ylabel is not None:
        ax.set_ylabel(
            ylabel,
            fontproperties=LABEL_FP,
            fontsize=12,
            color=TEXT_BLUE,
            labelpad=12,
        )

    if title is not None:
        ax.set_title(
            title,
            fontproperties=TITLE_FP,
            fontsize=16,
            color=TEXT_BLUE,
            pad=32,
        )


def add_top_legend(ax, ncol=2, y=1.20):
    legend = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        prop=LABEL_FP,
        labelcolor=TEXT_BLUE,
        handlelength=2.8,
        columnspacing=2.4,
        borderaxespad=0.0,
    )

    return legend


def focused_limits(values, pad=1.5, min_span=8):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return 0, 100

    lower = values.min() - pad
    upper = values.max() + pad

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

    fig.savefig(
        png_path,
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.35,
    )
    print(f"Saved: {png_path}")

    try:
        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            facecolor="white",
            pad_inches=0.35,
        )
        print(f"Saved: {pdf_path}")
    except Exception as e:
        print(f"\nPDF save failed for {basename}.")
        print(f"Error: {e}")
        print("Saving SVG instead.")

        try:
            fig.savefig(
                svg_path,
                bbox_inches="tight",
                facecolor="white",
                pad_inches=0.35,
            )
            print(f"Saved: {svg_path}")
        except Exception as e2:
            print("\nSVG save also failed.")
            print(f"Error: {e2}")

    plt.close(fig)


# CSV parsing

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
        test_acc = parse_numeric(row.get("test_accuracy", np.nan))
        cv_acc = parse_numeric(row.get("cv_best_accuracy", np.nan))

        if rep is None or np.isnan(test_acc):
            continue

        rows.append({
            "dataset": dataset,
            "classifier": classifier,
            "representation": rep,
            "test_accuracy": test_acc,
            "cv_accuracy": cv_acc,
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
    cv_acc = np.nan

    for _, row in df.iterrows():
        metric = normalize_text(row["Metric"])
        value = row["Value"]

        if metric == "representation":
            rep = infer_representation_from_text(value)

        elif metric == "test_accuracy":
            test_acc = parse_numeric(value)

        elif metric in [
            "best_cv_accuracy",
            "cv_best_accuracy",
            "cross_validation_accuracy",
            "cross_validation_best_accuracy",
            "cv_accuracy"
        ]:
            cv_acc = parse_numeric(value)

    if rep is None:
        rep = infer_representation_from_text(csv_path.parent.name)

    if rep is not None and not np.isnan(test_acc):
        rows.append({
            "dataset": dataset,
            "classifier": classifier,
            "representation": rep,
            "test_accuracy": test_acc,
            "cv_accuracy": cv_acc,
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
    if USE_CACHED_COMBINED_RESULTS:
        for candidate in COMBINED_RESULTS_CANDIDATES:
            if candidate.exists():
                print(f"Loading existing combined results from: {candidate}")
                df = pd.read_csv(candidate)
                return standardize_results_dataframe(df)

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
    df = standardize_results_dataframe(df)

    output_csv = OUTPUT_DIR / "combined_accuracy_results.csv"
    df.to_csv(output_csv, index=False)

    print("\nCombined results:")
    print(df[["dataset", "classifier", "representation", "test_accuracy"]])
    print(f"\nSaved combined results to: {output_csv}")

    return df


def standardize_results_dataframe(df):
    rename_map = {}

    if "accuracy" in df.columns and "test_accuracy" not in df.columns:
        rename_map["accuracy"] = "test_accuracy"

    df = df.rename(columns=rename_map)

    required = {"dataset", "classifier", "representation", "test_accuracy"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Combined results file is missing columns: {missing}")

    df["test_accuracy"] = df["test_accuracy"].apply(parse_numeric).astype(float)

    if "cv_accuracy" not in df.columns:
        df["cv_accuracy"] = np.nan
    else:
        df["cv_accuracy"] = df["cv_accuracy"].apply(parse_numeric)

    df["dataset"] = df["dataset"].astype(str)
    df["classifier"] = df["classifier"].astype(str)
    df["representation"] = df["representation"].astype(str)

    classifier_fix = {
        "RF": "Random Forest",
        "random_forest": "Random Forest",
        "RandomForest": "Random Forest",
        "Logistic Regression": "Logistic",
        "Multinomial Logistic": "Logistic",
    }

    df["classifier"] = df["classifier"].replace(classifier_fix)

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

    df = df.sort_values(["dataset", "classifier", "representation"]).reset_index(drop=True)

    return df


# Visualization 1: HaGRID accuracy profile

def plot_hagrid_accuracy_profile(df):
    hagrid_df = df[df["dataset"].astype(str) == "HaGRID"].copy()

    if hagrid_df.empty:
        print("No HaGRID results found. Skipping profile plot.")
        return

    reps = ordered_representations(hagrid_df)

    pivot = (
        hagrid_df
        .pivot_table(
            index="representation",
            columns="classifier",
            values="test_accuracy",
            aggfunc="max",
            observed=False,
        )
        .reindex(index=reps)
    )

    x = np.arange(len(reps))
    x_labels = [REP_SHORT.get(r, r) for r in reps]

    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    line_specs = {
        "Logistic": {
            "label": "MULTINOMIAL LOGISTIC",
            "color": TEXT_BLUE,
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

        vals = pivot[classifier].values.astype(float)
        all_values.extend(vals[~np.isnan(vals)])

        spec = line_specs[classifier]

        ax.plot(
            x,
            vals,
            label=spec["label"],
            color=spec["color"],
            marker=spec["marker"],
            linestyle=spec["linestyle"],
            linewidth=2.5,
            markersize=8.0,
            markeredgewidth=0,
        )

        for xi, val in zip(x, vals):
            if not np.isnan(val):
                ax.text(
                    xi,
                    val + 0.65,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontproperties=NUMBER_FP,
                    color=DARK_NAVY,
                )

    lower, upper = focused_limits(all_values, pad=3.5, min_span=18)
    ax.set_ylim(lower, upper)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    style_axis(ax)
    set_axis_text(
        ax,
        xlabel="FEATURE REPRESENTATION",
        ylabel="HELD-OUT TEST ACCURACY (%)",
        title="HAGRID ACCURACY PROFILE",
    )

    add_top_legend(ax, ncol=2, y=1.23)

    fig.subplots_adjust(top=0.70, bottom=0.18, left=0.10, right=0.98)

    save_figure(fig, "hagrid_accuracy_profile_styled")


# Visualization 2: HaGRID horizontal grouped bars

def plot_hagrid_grouped_bars(df):
    hagrid_df = df[df["dataset"].astype(str) == "HaGRID"].copy()

    if hagrid_df.empty:
        print("No HaGRID results found. Skipping bar chart.")
        return

    reps = ordered_representations(hagrid_df)

    pivot = (
        hagrid_df
        .pivot_table(
            index="representation",
            columns="classifier",
            values="test_accuracy",
            aggfunc="max",
            observed=False,
        )
        .reindex(index=reps)
    )

    y = np.arange(len(reps))
    bar_height = 0.34

    fig, ax = plt.subplots(figsize=(10.4, 5.7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    offsets = [-bar_height / 1.5, bar_height / 1.5]
    color_map = {
        "Logistic": TEXT_BLUE,
        "Random Forest": RF_COLOR,
    }

    all_values = []

    for k, classifier in enumerate(CLASSIFIER_ORDER):
        if classifier not in pivot.columns:
            continue

        vals = pivot[classifier].values.astype(float)
        all_values.extend(vals[~np.isnan(vals)])

        ax.barh(
            y + offsets[k],
            vals,
            height=bar_height,
            color=color_map[classifier],
            edgecolor="none",
            alpha=0.95,
            label=CLASSIFIER_DISPLAY[classifier].upper(),
        )

        for yi, val in zip(y + offsets[k], vals):
            if not np.isnan(val):
                ax.text(
                    val + 0.25,
                    yi,
                    f"{val:.1f}",
                    va="center",
                    ha="left",
                    fontsize=9,
                    fontproperties=NUMBER_FP,
                    color=DARK_NAVY,
                )

    lower, upper = focused_limits(all_values, pad=2.5, min_span=12)
    ax.set_xlim(lower, min(100, upper + 1))

    ax.set_yticks(y)
    ax.set_yticklabels([REP_DISPLAY.get(r, r) for r in reps])
    ax.invert_yaxis()

    style_axis(ax)
    set_axis_text(
        ax,
        xlabel="HELD-OUT TEST ACCURACY (%)",
        ylabel=None,
        title="HAGRID ACCURACY BY FEATURE REPRESENTATION",
    )

    add_top_legend(ax, ncol=2, y=1.20)

    fig.subplots_adjust(top=0.72, bottom=0.16, left=0.18, right=0.98)

    save_figure(fig, "hagrid_accuracy_bars_styled")


# Visualization 3: robustness gap

def plot_robustness_gap(df):
    controlled_datasets = ["Self ASL", "SignAlphaSet"]
    rows = []

    for classifier in CLASSIFIER_ORDER:
        for rep in REPRESENTATION_ORDER:
            controlled = df[
                (df["dataset"].astype(str).isin(controlled_datasets)) &
                (df["classifier"].astype(str) == classifier) &
                (df["representation"].astype(str) == rep)
            ]["test_accuracy"]

            hagrid = df[
                (df["dataset"].astype(str) == "HaGRID") &
                (df["classifier"].astype(str) == classifier) &
                (df["representation"].astype(str) == rep)
            ]["test_accuracy"]

            if len(controlled) > 0 and len(hagrid) > 0:
                rows.append({
                    "classifier": classifier,
                    "representation": rep,
                    "controlled_accuracy": controlled.mean(),
                    "hagrid_accuracy": hagrid.max(),
                    "drop": controlled.mean() - hagrid.max(),
                })

    gap_df = pd.DataFrame(rows)

    if gap_df.empty:
        print("Not enough data for robustness gap plot. Skipping.")
        return

    gap_df.to_csv(OUTPUT_DIR / "robustness_gap_results.csv", index=False)

    reps = ordered_representations(gap_df)

    pivot = (
        gap_df
        .pivot_table(
            index="representation",
            columns="classifier",
            values="drop",
            aggfunc="max",
            observed=False,
        )
        .reindex(index=reps)
    )

    y = np.arange(len(reps))
    bar_height = 0.34

    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    offsets = [-bar_height / 1.5, bar_height / 1.5]
    color_map = {
        "Logistic": TEXT_BLUE,
        "Random Forest": RF_COLOR,
    }

    max_value = 0

    for k, classifier in enumerate(CLASSIFIER_ORDER):
        if classifier not in pivot.columns:
            continue

        vals = pivot[classifier].values.astype(float)
        vals_clean = vals[~np.isnan(vals)]

        if len(vals_clean) > 0:
            max_value = max(max_value, vals_clean.max())

        ax.barh(
            y + offsets[k],
            vals,
            height=bar_height,
            color=color_map[classifier],
            edgecolor="none",
            alpha=0.95,
            label=CLASSIFIER_DISPLAY[classifier].upper(),
        )

        for yi, val in zip(y + offsets[k], vals):
            if not np.isnan(val):
                ax.text(
                    val + 0.25,
                    yi,
                    f"{val:.1f}",
                    va="center",
                    ha="left",
                    fontsize=9,
                    fontproperties=NUMBER_FP,
                    color=DARK_NAVY,
                )

    ax.set_xlim(0, max_value + 4)

    ax.set_yticks(y)
    ax.set_yticklabels([REP_DISPLAY.get(r, r) for r in reps])
    ax.invert_yaxis()

    style_axis(ax)
    set_axis_text(
        ax,
        xlabel="ACCURACY DROP TO HAGRID (PERCENTAGE POINTS)",
        ylabel=None,
        title="ROBUSTNESS GAP FROM CONTROLLED DATASETS TO HAGRID",
    )

    add_top_legend(ax, ncol=2, y=1.20)

    fig.subplots_adjust(top=0.72, bottom=0.15, left=0.19, right=0.98)

    save_figure(fig, "robustness_gap_bars_styled")


# Visualization 4: average rank

def plot_rank_summary(df):
    rank_df = df.copy()
    rank_df["condition"] = (
        rank_df["dataset"].astype(str) +
        " / " +
        rank_df["classifier"].astype(str)
    )

    rank_df["rank"] = (
        rank_df
        .groupby("condition", observed=False)["test_accuracy"]
        .rank(ascending=False, method="min")
    )

    rank_df.to_csv(OUTPUT_DIR / "representation_ranks.csv", index=False)

    mean_rank = (
        rank_df
        .groupby("representation", observed=False)["rank"]
        .mean()
        .reset_index()
    )

    reps = ordered_representations(rank_df)

    mean_rank["representation"] = pd.Categorical(
        mean_rank["representation"],
        categories=reps,
        ordered=True,
    )

    mean_rank = mean_rank.sort_values("representation")

    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(len(mean_rank))
    vals = mean_rank["rank"].values.astype(float)

    ax.barh(
        y,
        vals,
        color=TEXT_BLUE,
        edgecolor="none",
        alpha=0.92,
    )

    for yi, val in zip(y, vals):
        ax.text(
            val + 0.06,
            yi,
            f"{val:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            fontproperties=NUMBER_FP,
            color=DARK_NAVY,
        )

    labels = [
        REP_DISPLAY.get(str(rep), str(rep))
        for rep in mean_rank["representation"]
    ]

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlim(0, max(vals) + 0.8)

    style_axis(ax)
    set_axis_text(
        ax,
        xlabel="AVERAGE RANK (LOWER IS BETTER)",
        ylabel=None,
        title="AVERAGE REPRESENTATION RANK ACROSS CONDITIONS",
    )

    fig.subplots_adjust(top=0.82, bottom=0.15, left=0.18, right=0.98)

    save_figure(fig, "representation_average_rank_styled")


# Visualization 5: controlled vs HaGRID slope plot

def plot_controlled_vs_hagrid_slope(df):
    rows = []

    for classifier in CLASSIFIER_ORDER:
        for rep in REPRESENTATION_ORDER:
            controlled = df[
                (df["dataset"].astype(str).isin(["Self ASL", "SignAlphaSet"])) &
                (df["classifier"].astype(str) == classifier) &
                (df["representation"].astype(str) == rep)
            ]["test_accuracy"]

            hagrid = df[
                (df["dataset"].astype(str) == "HaGRID") &
                (df["classifier"].astype(str) == classifier) &
                (df["representation"].astype(str) == rep)
            ]["test_accuracy"]

            if len(controlled) > 0 and len(hagrid) > 0:
                rows.append({
                    "classifier": classifier,
                    "representation": rep,
                    "controlled": controlled.mean(),
                    "hagrid": hagrid.max(),
                })

    slope_df = pd.DataFrame(rows)

    if slope_df.empty:
        print("Not enough data for controlled-vs-HaGRID slope plot. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    color_map = {
        "Logistic": TEXT_BLUE,
        "Random Forest": RF_COLOR,
    }

    marker_map = {
        "Logistic": "o",
        "Random Forest": "s",
    }

    x_positions = [0, 1]
    x_labels = ["CONTROLLED\nDATASETS", "HAGRID"]

    for _, row in slope_df.iterrows():
        classifier = row["classifier"]
        rep = row["representation"]

        color = color_map.get(classifier, TEXT_BLUE)
        marker = marker_map.get(classifier, "o")

        ax.plot(
            x_positions,
            [row["controlled"], row["hagrid"]],
            color=color,
            linewidth=1.6,
            alpha=0.50,
        )

        ax.scatter(
            x_positions,
            [row["controlled"], row["hagrid"]],
            color=color,
            s=42,
            marker=marker,
            edgecolors="none",
            alpha=0.88,
        )

        if rep == "Hybrid":
            ax.text(
                1.03,
                row["hagrid"],
                "Hybrid",
                va="center",
                ha="left",
                fontsize=9,
                fontproperties=LABEL_FP,
                color=DARK_NAVY,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)

    all_values = list(slope_df["controlled"]) + list(slope_df["hagrid"])
    lower, upper = focused_limits(all_values, pad=3, min_span=22)
    ax.set_ylim(lower, upper)

    style_axis(ax)
    set_axis_text(
        ax,
        xlabel=None,
        ylabel="HELD-OUT TEST ACCURACY (%)",
        title="ACCURACY DROP UNDER VARIABLE IMAGE CONDITIONS",
    )

    handles = []
    labels = []

    for classifier in CLASSIFIER_ORDER:
        if classifier in slope_df["classifier"].values:
            handle = plt.Line2D(
                [0],
                [0],
                color=color_map[classifier],
                marker=marker_map[classifier],
                linestyle="-",
                linewidth=2,
                markersize=7,
                markeredgewidth=0,
            )
            handles.append(handle)
            labels.append(CLASSIFIER_DISPLAY[classifier].upper())

    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=2,
        frameon=False,
        prop=LABEL_FP,
        labelcolor=TEXT_BLUE,
        handlelength=2.2,
        columnspacing=2.4,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(top=0.70, bottom=0.17, left=0.15, right=0.92)

    save_figure(fig, "controlled_vs_hagrid_slope_styled")


# Main

if __name__ == "__main__":
    results = build_results_dataframe()

    plot_hagrid_accuracy_profile(results)
    plot_hagrid_grouped_bars(results)
    plot_robustness_gap(results)
    plot_rank_summary(results)
    plot_controlled_vs_hagrid_slope(results)

    print("\nDone. Figures saved in:")
    print(OUTPUT_DIR)