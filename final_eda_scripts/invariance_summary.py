from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm



PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "final_results_eda"
OUTPUT_DIR.mkdir(exist_ok=True)



TEXT_BLUE = "#6fb7b6"
DARK_NAVY = "#102A43"
GREEN = "#63B45D"
BG_COLOR = "white"

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

HEADER_FP = choose_font(
    candidate_names=[
        "Montserrat SemiBold",
        "Montserrat Bold",
        "Montserrat",
        "Aptos",
        "Bahnschrift",
        "Century Gothic",
        "Segoe UI Semibold",
        "Segoe UI",
    ],
    fallback_family="DejaVu Sans",
    weight="bold",
)

BODY_FP = choose_font(
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



TITLE = "Feature-Based Dimension Reduction for Gesture Recognition"

HEADERS = [
    "Representation",
    r"$\phi_r$",
    "Translation Invariant",
    "Scale Invariant",
    "Primary Geometry",
]

ROWS = [
    [
        "Image-plane coordinates",
        r"$\phi_{xy}$",
        "No",
        "No",
        "Absolute landmark position",
    ],
    [
        "Translated coordinates",
        r"$\phi_{\mathrm{trans}}$",
        "Yes",
        "No",
        "Relative landmark position",
    ],
    [
        "Scaled coordinates",
        r"$\phi_{\mathrm{scale}}$",
        "Yes",
        "Yes",
        "Normalized hand shape",
    ],
    [
        "3D coordinates",
        r"$\phi_{xyz}$",
        r"With $z$-shift",
        r"With $z$-scale",
        "Image-plane position + relative depth",
    ],
    [
        "Pairwise distances",
        r"$\phi_{\mathrm{dist}}$",
        "Yes",
        "No",
        "Relative landmark spacing",
    ],
    [
        "Normalized distances",
        r"$\phi_{\mathrm{norm}\!-\!\mathrm{dist}}$",
        "Yes",
        "Yes",
        "Scale-free spacing",
    ],
    [
        "Angles",
        r"$\phi_{\mathrm{angle}}$",
        "Yes",
        "Yes",
        "Finger bending/orientation",
    ],
    [
        "Hybrid features",
        r"$\phi_{\mathrm{hybrid}}$",
        "Depends",
        "Depends",
        "Combined geometry",
    ],
]



def make_invariance_summary_table():
    n_rows = len(ROWS)

    fig, ax = plt.subplots(figsize=(15.6, 4.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left = 0.045
    right = 0.975
    top = 0.845
    bottom = 0.09

    x_rep = 0.070
    x_phi = 0.330
    x_trans = 0.485
    x_scale = 0.645
    x_geom = 0.845

    ax.text(
        0.5,
        0.935,
        TITLE,
        ha="center",
        va="center",
        fontsize=22,
        fontproperties=TITLE_FP,
        color=DARK_NAVY,
    )

    top_rule_y = top
    header_rule_y = top - 0.105
    bottom_rule_y = bottom

    ax.plot([left, right], [top_rule_y, top_rule_y], color=GREEN, lw=2.2)
    ax.plot([left, right], [header_rule_y, header_rule_y], color=GREEN, lw=2.2)
    ax.plot([left, right], [bottom_rule_y, bottom_rule_y], color=GREEN, lw=2.2)

    header_y = (top_rule_y + header_rule_y) / 2 + 0.002

    header_positions = [x_rep, x_phi, x_trans, x_scale, x_geom]
    header_alignments = ["left", "center", "center", "center", "center"]

    for header, x, ha in zip(HEADERS, header_positions, header_alignments):
        ax.text(
            x,
            header_y,
            header,
            ha=ha,
            va="center",
            fontsize=16.2,
            fontproperties=HEADER_FP,
            color=TEXT_BLUE,
        )

    row_region_top = header_rule_y - 0.045
    row_region_bottom = bottom_rule_y + 0.045
    row_step = (row_region_top - row_region_bottom) / (n_rows - 1)

    for i, row in enumerate(ROWS):
        y = row_region_top - i * row_step

        ax.text(
            x_rep,
            y,
            row[0],
            ha="left",
            va="center",
            fontsize=15.2,
            fontproperties=BODY_FP,
            color=DARK_NAVY,
        )

        ax.text(
            x_phi,
            y,
            row[1],
            ha="center",
            va="center",
            fontsize=15.6,
            fontproperties=BODY_FP,
            color=DARK_NAVY,
        )

        ax.text(
            x_trans,
            y,
            row[2],
            ha="center",
            va="center",
            fontsize=15.2,
            fontproperties=BODY_FP,
            color=DARK_NAVY,
        )

        ax.text(
            x_scale,
            y,
            row[3],
            ha="center",
            va="center",
            fontsize=15.2,
            fontproperties=BODY_FP,
            color=DARK_NAVY,
        )

        ax.text(
            x_geom,
            y,
            row[4],
            ha="center",
            va="center",
            fontsize=15.2,
            fontproperties=BODY_FP,
            color=DARK_NAVY,
        )

    png_path = OUTPUT_DIR / "invariance_summary_table_with_phi.png"
    pdf_path = OUTPUT_DIR / "invariance_summary_table_with_phi.pdf"
    svg_path = OUTPUT_DIR / "invariance_summary_table_with_phi.svg"

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



if __name__ == "__main__":
    make_invariance_summary_table()
