from pathlib import Path
import argparse
import math
import textwrap

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Arc, FancyBboxPatch



PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "final_results_eda"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURE_PANEL_DIR = OUTPUT_DIR / "l_sign_feature_maps"
FEATURE_PANEL_DIR.mkdir(exist_ok=True)

INPUT_IMAGE_DIR = PROJECT_ROOT / "data" / "test_L"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")



TEXT_BLUE = "#6fb7b6"
DARK_NAVY = "#102A43"

GRID_COLOR = "#d8eeee"
TAN = "#A77735"
TAN_LIGHT = "#D8BE8A"

NODE_GRAY = "#3E454A"
NODE_BLUE = "#1F67C2"
EDGE_GRAY = "#343A40"
DIST_GREEN = "#4C9A3A"
ANGLE_PURPLE = "#7E2CB0"
BOX_GREEN = "#8cce7e"

SCALE_DIST="#4D6F2F"
NORM_DIST= "#4eb197"
ANGLES="#7f6dab"
HYBRID="#3282cd"

REPRESENTATION_TITLE_COLORS = {
    "Scaled coordinates": "#4D6F2F",
    "Normalized Distances": "#2F6A5A",
    "Angles": "#5A4A82",
    "Hybrid": "#102A43",
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



HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (0, 9), (9, 10), (10, 11), (11, 12),      # middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (5, 9), (9, 13), (13, 17),               # palm bridge
]

DISTANCE_PAIRS = [
    (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),
    (4, 8), (4, 12), (4, 16), (4, 20),
    (8, 12), (8, 16), (8, 20),
    (12, 16), (12, 20),
    (16, 20),
    (1, 5), (5, 9), (9, 13), (13, 17),
]

ANGLE_TRIPLES = [
    (0, 2, 4),       # thumb direction
    (0, 5, 8),       # index direction
    (5, 9, 12),      # middle finger relation
    (9, 13, 16),     # ring finger relation
    (13, 17, 20),    # pinky relation
    (4, 2, 8),       # key L-shape thumb/index geometry
]



def resolve_image_path(image_arg=None):
    if image_arg is not None:
        p = Path(image_arg)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"Image path does not exist:\n{p}")
        return p

    if not INPUT_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Input folder does not exist:\n{INPUT_IMAGE_DIR}\n\n"
            "Place your L-sign image in PROJECT_ROOT/data/test_L."
        )

    images = [
        p for p in sorted(INPUT_IMAGE_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        raise FileNotFoundError(
            f"No image files found in:\n{INPUT_IMAGE_DIR}\n\n"
            f"Accepted extensions: {IMAGE_EXTENSIONS}"
        )

    if len(images) > 1:
        print("Multiple images found in data/test_L. Using the first one:")
        for p in images:
            print(f"  {p.name}")

    return images[0]


def extract_mediapipe_landmarks(image_path):
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit(
            "MediaPipe is not installed. Run:\n"
            "  pip install mediapipe opencv-python matplotlib numpy"
        ) from exc

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.30,
    ) as hands:
        result = hands.process(image_rgb)

    if not result.multi_hand_landmarks:
        raise RuntimeError(
            "MediaPipe did not detect a hand. Try cropping closer around the hand, "
            "improving contrast/lighting, or lowering min_detection_confidence."
        )

    lm = result.multi_hand_landmarks[0].landmark

    points = np.array([[p.x, 1.0 - p.y] for p in lm], dtype=float)

    return points


def normalize_to_unit_box(points, pad=0.08):
    pts = points.copy()
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    span = max_xy - min_xy

    if np.any(span <= 1e-8):
        raise ValueError("Degenerate landmark bounding box.")

    pts = (pts - min_xy) / span
    pts = pad + (1.0 - 2.0 * pad) * pts

    return pts


def transform_points_to_box(pts, left, bottom, width, height):
    return np.column_stack([
        left + width * pts[:, 0],
        bottom + height * pts[:, 1],
    ])



def save_figure(fig, basename, output_dir=FEATURE_PANEL_DIR):
    png_path = output_dir / f"{basename}.png"
    pdf_path = output_dir / f"{basename}.pdf"
    svg_path = output_dir / f"{basename}.svg"

    fig.savefig(png_path, dpi=450, facecolor="white")
    print(f"Saved: {png_path}")

    try:
        fig.savefig(pdf_path, facecolor="white")
        print(f"Saved: {pdf_path}")
    except Exception as e:
        print(f"PDF save failed for {basename}: {e}")

    try:
        fig.savefig(svg_path, facecolor="white")
        print(f"Saved: {svg_path}")
    except Exception as e:
        print(f"SVG save failed for {basename}: {e}")

    plt.close(fig)


def setup_panel(ax, title, caption_lines, edge_color):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    border = FancyBboxPatch(
        (0.035, 0.035),
        0.93,
        0.93,
        boxstyle="round,pad=0.010,rounding_size=0.020",
        linewidth=2.2        ,
        edgecolor=edge_color,
        facecolor="white",
        transform=ax.transAxes,
        zorder=-10,
    )
    ax.add_patch(border)

    ax.text(
        0.5,
        0.915,
        title,
        ha="center",
        va="center",
        fontproperties=BOLD_FP,
        fontsize=18,
        color=REPRESENTATION_TITLE_COLORS.get(title),
        transform=ax.transAxes,
    )

    y_start = 0.105
    for i, line in enumerate(caption_lines):
        ax.text(
            0.5,
            y_start - 0.035 * i,
            line,
            ha="center",
            va="center",
            fontproperties=LABEL_FP,
            fontsize=8.4,
            color=TEXT_BLUE,
            transform=ax.transAxes,
        )


def draw_skeleton(
    ax,
    pts,
    node_color=NODE_GRAY,
    edge_color=EDGE_GRAY,
    node_size=34,
    line_width=1.15,
    label_nodes=False,
    zorder=3,
):
    for a, b in HAND_CONNECTIONS:
        ax.plot(
            [pts[a, 0], pts[b, 0]],
            [pts[a, 1], pts[b, 1]],
            color=edge_color,
            linewidth=line_width,
            solid_capstyle="round",
            zorder=zorder,
        )

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=node_size,
        color=node_color,
        edgecolor="#2C3135",
        linewidth=0.42,
        zorder=zorder + 1,
    )

    if label_nodes:
        for i, (x, y) in enumerate(pts):
            ax.text(
                x + 0.010,
                y + 0.010,
                str(i),
                fontsize=5.5,
                color="#111111",
                ha="left",
                va="bottom",
                fontproperties=LABEL_FP,
                zorder=zorder + 2,
            )


def draw_unit_square(ax, left, bottom, width, height, color):
    ax.plot(
        [left, left + width, left + width, left, left],
        [bottom, bottom, bottom + height, bottom + height, bottom],
        color=EDGE_GRAY,
        linewidth=1.25,
        zorder=0,
    )

    for k in [0.25, 0.5, 0.75]:
        x = left + width * k
        y = bottom + height * k

        ax.plot(
            [x, x],
            [bottom, bottom + height],
            color=color,
            linewidth=0.62,
            linestyle=(0, (4, 4)),
            zorder=0,
        )

        ax.plot(
            [left, left + width],
            [y, y],
            color=color,
            linewidth=0.62,
            linestyle=(0, (4, 4)),
            zorder=0,
        )

    ax.text(left - 0.030, bottom, "0", ha="right", va="center",
            fontsize=11.5, color=EDGE_GRAY, fontproperties=LABEL_FP)
    ax.text(left - 0.030, bottom + height, "1", ha="right", va="center",
            fontsize=11.5, color=EDGE_GRAY, fontproperties=LABEL_FP)
    ax.text(left, bottom - 0.030, "0", ha="center", va="top",
            fontsize=11.5, color=EDGE_GRAY, fontproperties=LABEL_FP)
    ax.text(left + width, bottom - 0.030, "1", ha="center", va="top",
            fontsize=11.5, color=EDGE_GRAY, fontproperties=LABEL_FP)


def draw_distances(ax, pts, pairs=DISTANCE_PAIRS, color=DIST_GREEN, line_width=0.55, alpha=0.82):
    for a, b in pairs:
        ax.plot(
            [pts[a, 0], pts[b, 0]],
            [pts[a, 1], pts[b, 1]],
            color=color,
            linewidth=line_width,
            alpha=alpha,
            zorder=2,
        )


def angle_degrees(vec):
    return math.degrees(math.atan2(vec[1], vec[0]))


def smallest_arc_angles(theta1, theta2):
    delta = (theta2 - theta1 + 360) % 360
    if delta > 180:
        theta1, theta2 = theta2, theta1
        delta = 360 - delta
    return theta1, theta1 + delta


def draw_angle_arc(ax, pts, triple, radius=0.034, color=ANGLE_PURPLE, line_width=1.35):
    a, b, c = triple
    p_a = pts[a]
    p_b = pts[b]
    p_c = pts[c]

    v1 = p_a - p_b
    v2 = p_c - p_b

    if np.linalg.norm(v1) < 1e-8 or np.linalg.norm(v2) < 1e-8:
        return

    theta1 = angle_degrees(v1)
    theta2 = angle_degrees(v2)
    start, end = smallest_arc_angles(theta1, theta2)

    if end - start > 130:
        mid = 0.5 * (start + end)
        start = mid - 55
        end = mid + 55

    arc = Arc(
        p_b,
        width=2 * radius,
        height=2 * radius,
        angle=0,
        theta1=start,
        theta2=end,
        color=color,
        linewidth=line_width,
        zorder=6,
    )
    ax.add_patch(arc)

    ax.scatter(
        [p_b[0]],
        [p_b[1]],
        s=10,
        color=color,
        zorder=7,
        alpha=0.65,
        edgecolor="none",
    )


def draw_angles(ax, pts, triples=ANGLE_TRIPLES, radius=0.034):
    for triple in triples:
        draw_angle_arc(ax, pts, triple, radius=radius)



def plot_scaled_coordinates(pts):
    fig, ax = plt.subplots(figsize=(3.25, 3.80))
    fig.subplots_adjust(0, 0, 1, 1)

    setup_panel(
        ax,
        "Scaled coordinates",
        [
            "preserves normalized global hand layout;",
            "removes translation and scale effects",
        ],
        BOX_GREEN
    )

    left, bottom, width, height = 0.235, 0.245, 0.545, 0.540
    draw_unit_square(ax, left, bottom, width, height, color=SCALE_DIST)

    plot_pts = transform_points_to_box(pts, left, bottom, width, height)

    draw_skeleton(
        ax,
        plot_pts,
        node_color=NODE_GRAY,
        edge_color=EDGE_GRAY,
        node_size=30,
        line_width=1.05,
    )

    save_figure(fig, "A_scaled_coordinates")


def plot_normalized_distances(pts):
    fig, ax = plt.subplots(figsize=(3.25, 3.80))
    fig.subplots_adjust(0, 0, 1, 1)

    setup_panel(
        ax,
        "Normalized Distances",
        [
            "encodes relative landmark spacing;",
            "removes translation and scale effects",
        ],
        NORM_DIST,
    )

    plot_pts = transform_points_to_box(pts, left=0.235, bottom=0.245, width=0.545, height=0.540)

    draw_distances(ax, plot_pts)
    draw_skeleton(
        ax,
        plot_pts,
        node_color=NODE_GRAY,
        edge_color=EDGE_GRAY,
        node_size=31,
        line_width=1.05,
    )

    save_figure(fig, "B_normalized_distances")


def plot_angles(pts):
    fig, ax = plt.subplots(figsize=(3.25, 3.80))
    fig.subplots_adjust(0, 0, 1, 1)

    setup_panel(
        ax,
        "Angles",
        [
            "encodes local bending/orientation;",
            "translation and scale invariant",
        ],
        ANGLES
    )

    plot_pts = transform_points_to_box(pts, left=0.235, bottom=0.245, width=0.545, height=0.540)

    draw_skeleton(
        ax,
        plot_pts,
        node_color=NODE_GRAY,
        edge_color=EDGE_GRAY,
        node_size=31,
        line_width=1.05,
    )
    draw_angles(ax, plot_pts, radius=0.031)

    save_figure(fig, "C_angles")


def plot_hybrid(pts):
    fig, ax = plt.subplots(figsize=(3.25, 3.80))
    fig.subplots_adjust(0, 0, 1, 1)

    setup_panel(
        ax,
        "Hybrid",
        [
            "combines global layout, relative spacing,",
            "and local finger geometry",
        ],
        HYBRID
    )

    plot_pts = transform_points_to_box(pts, left=0.235, bottom=0.245, width=0.545, height=0.540)

    draw_distances(ax, plot_pts, alpha=0.76)
    draw_skeleton(
        ax,
        plot_pts,
        node_color=NODE_BLUE,
        edge_color=EDGE_GRAY,
        node_size=33,
        line_width=1.05,
    )
    draw_angles(ax, plot_pts, radius=0.031)

    save_figure(fig, "D_hybrid")


def plot_combined_grid():
    import matplotlib.image as mpimg

    panel_files = [
        FEATURE_PANEL_DIR / "A_scaled_coordinates.png",
        FEATURE_PANEL_DIR / "B_normalized_distances.png",
        FEATURE_PANEL_DIR / "C_angles.png",
        FEATURE_PANEL_DIR / "D_hybrid.png",
    ]

    missing = [p for p in panel_files if not p.exists()]
    if missing:
        print("Skipping combined grid. Missing:")
        for p in missing:
            print(f"  {p}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 7.9))
    fig.patch.set_facecolor("white")

    for ax, path in zip(axes.ravel(), panel_files):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.02, hspace=0.02)

    save_figure(fig, "feature_map_grid")

def plot_combined_grid_horizontal():
    import matplotlib.image as mpimg

    panel_files = [
        FEATURE_PANEL_DIR / "A_scaled_coordinates.png",
        FEATURE_PANEL_DIR / "B_normalized_distances.png",
        FEATURE_PANEL_DIR / "C_angles.png",
        FEATURE_PANEL_DIR / "D_hybrid.png",
    ]

    missing = [p for p in panel_files if not p.exists()]
    if missing:
        print("Skipping combined grid. Missing:")
        for p in missing:
            print(f"  {p}")
        return

    fig, axes = plt.subplots(1, 4, figsize=(6.8, 7.9))
    fig.patch.set_facecolor("white")

    for ax, path in zip(axes.ravel(), panel_files):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.02, hspace=0.02)

    save_figure(fig, "feature_map_horizontal")

def plot_landmarks_only(pts):
    fig, ax = plt.subplots(figsize=(3.25, 3.80))
    fig.subplots_adjust(0, 0, 1, 1)

    setup_panel(
        ax,
        "Landmarks",
        [
            "represents the hand using extracted keypoints;",
            "retains landmark position and connectivity",
        ],
    )

    plot_pts = transform_points_to_box(
        pts,
        left=0.235,
        bottom=0.245,
        width=0.545,
        height=0.540,
    )

    draw_skeleton(
        ax,
        plot_pts,
        node_color=NODE_GRAY,
        edge_color=EDGE_GRAY,
        node_size=30,
        line_width=1.05,
    )

    save_figure(fig, "landmarks_only")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional image path. If omitted, the first image in data/test_L is used.",
    )
    args = parser.parse_args()

    image_path = resolve_image_path(args.image)
    print(f"Using input image: {image_path}")

    raw_landmarks = extract_mediapipe_landmarks(image_path)
    scaled_landmarks = normalize_to_unit_box(raw_landmarks)

    plot_scaled_coordinates(scaled_landmarks)
    plot_normalized_distances(scaled_landmarks)
    plot_angles(scaled_landmarks)
    plot_hybrid(scaled_landmarks)
    plot_combined_grid()
    plot_combined_grid_horizontal()

    print("\nDone. Figures saved in:")
    print(FEATURE_PANEL_DIR)


if __name__ == "__main__":
    main()