from pathlib import Path
import random

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm



PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAGRID_ROOT = PROJECT_ROOT / "hagrid_100_data"
GESTURE_NAME = "palm"

OUTPUT_DIR = PROJECT_ROOT / "final_eda"
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_IMAGES = None
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}



TEXT_BLUE = "#6fb7b6"
DARK_NAVY = "#102A43"

SCATTER_COLOR = "#aee0ad"
HIST_COLOR = "#aee0ad"
HIST_EDGE = "#ffffff"
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



def collect_image_paths(gesture_folder):
    image_paths = []

    for path in gesture_folder.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)

    return image_paths


def extract_hand_geometry(image_path, hands_detector):
    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    landmarks = results.multi_hand_landmarks[0].landmark

    xs = np.array([lm.x for lm in landmarks], dtype=float)
    ys = np.array([lm.y for lm in landmarks], dtype=float)

    center_x = xs.mean()
    center_y = ys.mean()

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    bbox_diagonal = np.sqrt(bbox_width**2 + bbox_height**2)
    bbox_area = bbox_width * bbox_height

    return {
        "center_x": center_x,
        "center_y": center_y,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_diagonal": bbox_diagonal,
        "bbox_area": bbox_area,
    }


def build_dataframe():
    random.seed(RANDOM_SEED)

    gesture_folder = HAGRID_ROOT / GESTURE_NAME

    if not gesture_folder.exists():
        raise FileNotFoundError(f"Gesture folder not found: {gesture_folder}")

    image_paths = collect_image_paths(gesture_folder)

    print(f"Gesture: {GESTURE_NAME}")
    print(f"Found {len(image_paths)} images.")

    if MAX_IMAGES is not None and len(image_paths) > MAX_IMAGES:
        image_paths = random.sample(image_paths, MAX_IMAGES)
        print(f"Subsampled to {len(image_paths)} images.")

    rows = []
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3
    ) as hands_detector:

        for k, image_path in enumerate(image_paths, start=1):
            if k % 100 == 0:
                print(f"Processed {k}/{len(image_paths)} images...")

            geometry = extract_hand_geometry(image_path, hands_detector)

            if geometry is None:
                continue

            rows.append({
                "gesture": GESTURE_NAME,
                "image_path": str(image_path),
                **geometry
            })

    df = pd.DataFrame(rows)

    print(f"Detected hands in {len(df)}/{len(image_paths)} images.")

    return df



def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="both", colors=TEXT_BLUE, length=0, pad=6)

    for label in ax.get_xticklabels():
        label.set_fontproperties(LABEL_FP)
        label.set_fontsize(10)
        label.set_color(TEXT_BLUE)

    for label in ax.get_yticklabels():
        label.set_fontproperties(LABEL_FP)
        label.set_fontsize(10)
        label.set_color(TEXT_BLUE)

    ax.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def set_axis_text(ax, xlabel=None, ylabel=None, title=None):
    if xlabel is not None:
        ax.set_xlabel(
            xlabel,
            fontproperties=LABEL_FP,
            fontsize=12,
            color=TEXT_BLUE,
            labelpad=8
        )

    if ylabel is not None:
        ax.set_ylabel(
            ylabel,
            fontproperties=LABEL_FP,
            fontsize=12,
            color=TEXT_BLUE,
            labelpad=8
        )

    if title is not None:
        ax.set_title(
            title,
            fontproperties=TITLE_FP,
            fontsize=15,
            color=TEXT_BLUE,
            pad=14
        )



def make_final_two_part_figure(df):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(8.8, 3.8),
        constrained_layout=True
    )

    ax_scatter, ax_hist = axes

    ax_scatter.scatter(
        df["center_x"],
        df["center_y"],
        s=42,
        alpha=0.60,
        color=SCATTER_COLOR,
        edgecolors="none"
    )

    ax_scatter.set_xlim(0, 1)
    ax_scatter.set_ylim(1, 0)

    set_axis_text(
        ax_scatter,
        xlabel="HAND CENTER X",
        ylabel="HAND CENTER Y",
        title="DETECTED HAND CENTERS"
    )

    style_axis(ax_scatter)

    ax_hist.hist(
        df["bbox_diagonal"],
        bins=24,
        color=HIST_COLOR,
        edgecolor=HIST_EDGE,
        linewidth=1.5,
        alpha=0.95
    )

    median_diag = df["bbox_diagonal"].median()

    ax_hist.axvline(
        median_diag,
        color=DARK_NAVY,
        linewidth=2.0,
        alpha=0.85
    )

    ax_hist.text(
        median_diag,
        ax_hist.get_ylim()[1] * 0.92,
        "median",
        rotation=90,
        ha="right",
        va="top",
        fontsize=9,
        fontproperties=LABEL_FP,
        color=DARK_NAVY
    )

    set_axis_text(
        ax_hist,
        xlabel="BOUNDING-BOX DIAGONAL",
        ylabel="COUNT",
        title="APPARENT HAND-SIZE VARIATION"
    )

    style_axis(ax_hist)

    png_path = OUTPUT_DIR / f"hagrid_{GESTURE_NAME}_position_scale_styled.png"
    pdf_path = OUTPUT_DIR / f"hagrid_{GESTURE_NAME}_position_scale_styled.pdf"
    svg_path = OUTPUT_DIR / f"hagrid_{GESTURE_NAME}_position_scale_styled.svg"

    fig.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white")
    print(f"Saved PNG to: {png_path}")

    try:
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        print(f"Saved PDF to: {pdf_path}")
    except Exception as e:
        print("\nPDF save failed because of font embedding.")
        print(f"Error: {e}")
        print("Saving SVG instead.")

        try:
            fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
            print(f"Saved SVG to: {svg_path}")
        except Exception as e2:
            print("\nSVG save also failed.")
            print(f"Error: {e2}")

    plt.close(fig)



if __name__ == "__main__":
    df = build_dataframe()

    csv_path = OUTPUT_DIR / f"hagrid_{GESTURE_NAME}_position_scale.csv"
    df.to_csv(csv_path, index=False)

    print(f"Saved CSV to: {csv_path}")

    print("\nSummary:")
    print(
        df[["center_x", "center_y", "bbox_width", "bbox_height", "bbox_diagonal", "bbox_area"]]
        .describe()
    )

    make_final_two_part_figure(df)
