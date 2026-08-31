from pathlib import Path
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Arc, FancyBboxPatch


# Paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "final_results_eda"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURE_PANEL_DIR = OUTPUT_DIR / "l_sign_feature_maps"
FEATURE_PANEL_DIR.mkdir(exist_ok=True)

INPUT_IMAGE_DIR = PROJECT_ROOT / "data" / "test_L"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


# Aesthetic settings

TEXT_BLUE = "#6fb7b6"
DARK_NAVY = "#102A43"

GRID_COLOR = "#d8eeee"
TAN = "#A77735"
TAN_LIGHT = "#D8BE8A"
NODE_GRAY = "#3E454A"
NODE_BLUE = "#1F67C2"
DIST_GREEN = "#4C9A3A"
ANGLE_PURPLE = "#7E2CB0"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "none",
    "svg.fonttype": "path",
})


# Font handling

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


# Hand geometry configuration

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (0, 9), (9, 10), (10, 11), (11, 12),      # middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (5, 9), (9, 13), (13, 17),               # palm bridge
]

# The actual pairwise-distance representation may contain all C(21,2) distances;
DISTANCE_PAIRS = [
    (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),
    (4, 8), (4, 12), (4, 16), (4, 20),
    (8, 12), (8, 16), (8, 20),
    (12, 16), (12, 20),
    (16, 20),
    (1, 5), (5, 9), (9, 13), (13, 17),
    (4, 5), (8, 9), (12, 13), (16, 17),
]

ANGLE_TRIPLES = [
    (0, 1, 2), (1, 2, 3),       # thumb
    (0, 5, 6), (5, 6, 7),       # index
    (0, 9, 10), (9, 10, 11),    # middle
    (0, 13, 14), (13, 14, 15),  # ring
    (0, 17, 18), (17, 18, 19),  # pinky
    (4, 2, 8),                  # schematic thumb/index relation for L shape
]


# Helpers

def find_input_image():
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


def save_figure(fig, basename, output_dir=FEATURE_PANEL_DIR):
    png_path = output_dir / f"{basename}.png"
    pdf_path = output_dir / f"{basename}.pdf"
    svg_path = output_dir / f"{basename}.svg"

    fig.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white")
    print(f"Saved: {png_path}")

    try:
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {pdf_path}")
    except Exception as e:
        print(f"\nPDF save failed for {basename}.")
        print(f"Error: {e}")
        print("Saving SVG instead.")
        try:
            fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
            print(f"Saved: {svg_path}")
        except Exception as e2:
            print("\nSVG save also failed.")
            print(f"Error: {e2}")

    plt.close(fig)


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
        min_detection_confidence=0.35,
    ) as hands:
        result = hands.process(image_rgb)

    if not result.multi_hand_landmarks:
        raise RuntimeError(
            "MediaPipe did not detect a hand. Try cropping closer around the hand, "
            "improving contrast/lighting, or lowering min_detection_confidence."
        )

    lm = result.multi_hand_landmarks[0].landmark

    # Matplotlib uses y increasing upward, so we invert y.
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


def setup_panel(ax, title, caption_lines):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    border = FancyBboxPatch(
        (0.015, 0.015),
        0.97,
        0.97,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.5,
        edgecolor=TAN_LIGHT,
        facecolor="white",
        transform=ax.transAxes,
        zorder=-10,
    )
    ax.add_patch(border)

    ax.text(
        0.5,
        0.94,
        title,
        ha="center",
        va="center",
        fontproperties=BOLD_FP,
        fontsize=22,
        color=TAN,
        transform=ax.transAxes,
    )

    for i, line in enumerate(caption_lines):
        ax.text(
            0.5,
            0.088 - 0.04 * i,
            line,
            ha="center",
            va="center",
            fontproperties=LABEL_FP,
            fontsize=12.5,
            color=TAN,
            transform=ax.transAxes,
        )


def draw_skeleton(
    ax,
    pts,
    node_color=NODE_GRAY,
    edge_color=NODE_GRAY,
    node_size=42,
    line_width=1.35,
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
        linewidth=0.5,
        zorder=zorder + 1,
    )

    if label_nodes:
        for i, (x, y) in enumerate(pts):
            ax.text(
                x + 0.014,
                y + 0.014,
                str(i),
                fontsize=7,
                color="#111111",
                ha="left",
                va="bottom",
                fontproperties=LABEL_FP,
                zorder=zorder + 2,
            )


def draw_unit_square(ax, left, bottom, size):
    ax.plot(
        [left, left + size, left + size, left, left],
        [bottom, bottom, bottom + size, bottom + size, bottom],
        color="#2F3439",
        linewidth=1.5,
        zorder=0,
    )

    for k in [0.25, 0.5, 0.75]:
        x = left + size * k
        y = bottom + size * k

        ax.plot(
            [x, x],
            [bottom, bottom + size],
            color="#D9D9D9",
            linewidth=0.8,
            linestyle=(0, (4, 4)),
            zorder=0,
        )

        ax.plot(
            [left, left + size],
            [y, y],
            color="#D9D9D9",
            linewidth=0.8,
            linestyle=(0, (4, 4)),
            zorder=0,
        )

    ax.text(left - 0.035, bottom, "0", ha="right", va="center", fontsize=18, color="#343A40", fontproperties=LABEL_FP)
    ax.text(left - 0.035, bottom + size, "1", ha="right", va="center", fontsize=18, color="#343A40", fontproperties=LABEL_FP)
    ax.text(left, bottom - 0.04, "0", ha="center", va="top", fontsize=18, color="#343A40", fontproperties=LABEL_FP)
    ax.text(left + size, bottom - 0.04, "1", ha="center", va="top", fontsize=18, color="#343A40", fontproperties=LABEL_FP)


def transform_points_to_box(pts, left, bottom, size):
    return np.column_stack([
        left + size * pts[:, 0],
        bottom + size * pts[:, 1],
    ])


def transform_points_to_panel(pts):
    left, bottom, size = 0.18, 0.21, 0.64
    return transform_points_to_box(pts, left, bottom, size)


def draw_distances(ax, pts, pairs=DISTANCE_PAIRS, color=DIST_GREEN, line_width=0.75, alpha=0.90):
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


def draw_angle_arc(ax, pts, triple, radius=0.038, color=ANGLE_PURPLE, line_width=2.0):
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

    delta = (theta2 - theta1 + 360) % 360

    if delta > 180:
        theta1, theta2 = theta2, theta1
        delta = 360 - delta

    arc = Arc(
        p_b,
        width=2 * radius,
        height=2 * radius,
        angle=0,
        theta1=theta1,
        theta2=theta1 + delta,
        color=color,
        linewidth=line_width,
        zorder=6,
    )
    ax.add_patch(arc)

    end_angle = math.radians(theta1 + delta)
    prev_angle = math.radians(theta1 + delta - 8)

    end = p_b + radius * np.array([math.cos(end_angle), math.sin(end_angle)])
    prev = p_b + radius * np.array([math.cos(prev_angle), math.sin(prev_angle)])

    ax.annotate(
        "",
        xy=end,
        xytext=prev,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=line_width,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=7,
    )


def draw_angles(ax, pts, triples=ANGLE_TRIPLES, radius=0.038):
    for triple in triples:
        draw_angle_arc(ax, pts, triple, radius=radius)


# Plotting

def plot_scaled_coordinates(pts):
    fig, ax = plt.subplots(figsize=(4.0, 5.0))
    setup_panel(
        ax,
        "A. Scaled coordinates",
        [
            "preserves normalized",
            "global hand layout; removes translation and scale effects",
        ],
    )

    left, bottom, size = 0.18, 0.23, 0.64
    draw_unit_square(ax, left, bottom, size)

    plot_pts = transform_points_to_box(pts, left, bottom, size)

    draw_skeleton(
        ax,
        plot_pts,
        node_color=NODE_GRAY,
        edge_color=NODE_GRAY,
        node_size=38,
        line_width=1.35,
    )

    save_figure(fig, "A_scaled_coordinates")


def plot_normalized_distances(pts):
    fig, ax = plt.subplots(figsize=(4.0, 5.0))
    setup_panel(
        ax,
        "B. Normalized distances",
        [
            "encodes relative landmark",
            "spacing; removes translation and scale effects",
        ],
    )

    plot_pts = transform_points_to_panel(pts)

    draw_skeleton(
        ax,
        plot_pts,
        node_color=NODE_GRAY,
        edge_color=NODE_GRAY,
        node_size=42,
        line_width=1.35,
    )
    draw_distances(ax, plot_pts)

    save_figure(fig, "B_normalized_distances")


def plot_angles(pts):
    fig, ax = plt.subplots(figsize=(4.0, 5.0))
    setup_panel(
        ax,
        "C. Angles",
        [
            "encodes local",
            "bending/orientation; translation and scale invariant",
        ],
    )

    plot_pts = transform_points_to_panel(pts)

    draw_skeleton(
        ax,
        plot_pts,
        node_color=NODE_GRAY,
        edge_color=NODE_GRAY,
        node_size=42,
        line_width=1.35,
    )
    draw_angles(ax, plot_pts)

    save_figure(fig, "C_angles")


def plot_hybrid(pts):
    fig, ax = plt.subplots(figsize=(4.0, 5.0))
    setup_panel(
        ax,
        "D. Hybrid",
        [
            "combines global layout,",
            "relative spacing, and local finger geometry",
        ],
    )

    plot_pts = transform_points_to_panel(pts)

    draw_skeleton(
        ax,
        plot_pts,
        node_color=NODE_BLUE,
        edge_color=NODE_GRAY,
        node_size=45,
        line_width=1.35,
    )
    draw_distances(ax, plot_pts, alpha=0.88)
    draw_angles(ax, plot_pts)

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

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 9.4))
    fig.patch.set_facecolor("white")

    for ax, path in zip(axes.ravel(), panel_files):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")

    fig.subplots_adjust(wspace=0.02, hspace=0.02)

    save_figure(fig, "feature_map_grid")


# Main

if __name__ == "__main__":
    image_path = find_input_image()
    print(f"Using input image: {image_path}")

    raw_landmarks = extract_mediapipe_landmarks(image_path)
    scaled_landmarks = normalize_to_unit_box(raw_landmarks)

    plot_scaled_coordinates(scaled_landmarks)
    plot_normalized_distances(scaled_landmarks)
    plot_angles(scaled_landmarks)
    plot_hybrid(scaled_landmarks)
    plot_combined_grid()

    print("\nDone. Figures saved in:")
    print(FEATURE_PANEL_DIR)