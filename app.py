"""
╔══════════════════════════════════════════════════════════════╗
║                        ImgKit v1.0                           ║
║         Lightweight Image Manipulation Tool                  ║
║         Built with: Python · Streamlit · scikit-image        ║
╚══════════════════════════════════════════════════════════════╝

SDLC PHASE: Development  |  Version: 1.0  |  Status: Active

PROJECT STRUCTURE:
    app.py              ← You are here (main entry point)
    requirements.txt    ← All dependencies

PIPELINE ORDER (fixed):
    Upload → Metadata → Crop → Greyscale → Blur → Brightness → Contrast → Export
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import io
import os

import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter

# scikit-image for image processing operations
from skimage import color
from skimage.filters import gaussian


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ImgKit",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — colorful dark theme, compact sidebar, fit image to screen
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0d0f18;
    --surface:   #13161f;
    --border:    #1f2333;
    --accent:    #38bdf8;
    --accent2:   #a78bfa;
    --accent3:   #34d399;
    --accent4:   #fb923c;
    --text:      #e2e8f0;
    --muted:     #64748b;
}

/* ── GLOBAL ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

/* Remove default top padding so header sits tight */
[data-testid="stAppViewContainer"] > .main > div:first-child {
    padding-top: 0.5rem !important;
}

/* ── TOP HEADER BAR (full width) ── */
.top-header {
    background: linear-gradient(90deg, #0d0f18 0%, #13161f 40%, #0d0f18 100%);
    border-bottom: 1px solid #1f2333;
    padding: 10px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1rem;
}
.top-header .brand {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.top-header .sep { color: #1f2333; font-size: 1.2rem; }
.top-header .page { color: #64748b; font-size: 0.85rem; }
.top-header .pill {
    margin-left: auto;
    background: #1f2333;
    border: 1px solid #38bdf8;
    color: #38bdf8;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 2px;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Sidebar brand — compact at top */
.sb-brand {
    padding: 10px 0 6px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}
.sb-brand .name {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sb-brand .sub {
    font-size: 0.6rem;
    color: #475569;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Section labels */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 4px;
    margin: 10px 0 6px 0;
}

/* Color-coded section labels */
.sl-upload  { border-color: #38bdf8; color: #38bdf8; }
.sl-meta    { border-color: #34d399; color: #34d399; }
.sl-edit    { border-color: #a78bfa; color: #a78bfa; }
.sl-export  { border-color: #fb923c; color: #fb923c; }

/* Pipeline badge */
.pipeline-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: linear-gradient(135deg, #1e3a5f, #2d1b69);
    border: 1px solid #38bdf8;
    color: #38bdf8;
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    padding: 3px 8px;
    border-radius: 4px;
    letter-spacing: 1px;
    margin-bottom: 4px;
}

/* Info box */
.info-box {
    background: #0f1628;
    border-left: 3px solid #38bdf8;
    padding: 8px 12px;
    border-radius: 0 6px 6px 0;
    font-size: 0.78rem;
    color: #94a3b8;
    margin: 4px 0;
    line-height: 1.5;
}

/* Metadata cards */
.meta-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    margin-top: 6px;
}
.meta-card {
    background: #13161f;
    border: 1px solid #1f2333;
    border-radius: 8px;
    padding: 8px 12px;
    transition: border-color 0.2s;
}
.meta-card:hover { border-color: #38bdf8; }
.meta-card .label {
    font-size: 0.6rem;
    letter-spacing: 2px;
    color: #475569;
    text-transform: uppercase;
}
.meta-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: #38bdf8;
    margin-top: 2px;
    word-break: break-all;
}

/* Apply button */
div.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
    padding: 0.45rem 1rem !important;
    width: 100%;
    transition: all 0.2s;
    box-shadow: 0 0 12px rgba(56,189,248,0.2);
}
div.stButton > button:hover {
    box-shadow: 0 0 20px rgba(56,189,248,0.4) !important;
    transform: translateY(-1px);
}

/* Download button */
div.stDownloadButton > button {
    background: linear-gradient(135deg, #065f46, #064e3b) !important;
    color: #34d399 !important;
    border: 1px solid #34d399 !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    width: 100%;
}

/* Expander */
[data-testid="stExpander"] {
    background: #0f1120 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    margin-bottom: 4px !important;
}
[data-testid="stExpander"]:hover {
    border-color: #38bdf8 !important;
}

/* Image container — constrained height so it fits screen */
[data-testid="stImage"] {
    max-height: 65vh;
    overflow: hidden;
    display: flex;
    justify-content: center;
}
[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid #1f2333;
    max-height: 65vh;
    width: auto !important;
    max-width: 100%;
    object-fit: contain;
    box-shadow: 0 0 30px rgba(56,189,248,0.08);
}

/* Image wrapper card */
.img-card {
    background: #0f1120;
    border: 1px solid #1f2333;
    border-radius: 12px;
    padding: 12px;
}


/* ── FONT SIZES — increased for readability ── */
p, span, div, label { font-size: 16px !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div  { font-size: 15px !important; }
.sb-brand .name                { font-size: 1.5rem !important; }
.sb-brand .sub                 { font-size: 0.82rem !important; }
.section-label                 { font-size: 0.82rem !important; letter-spacing: 2px !important; }
.pipeline-badge                { font-size: 0.78rem !important; }
.info-box                      { font-size: 0.95rem !important; }
.meta-card .label              { font-size: 0.75rem !important; }
.meta-card .value              { font-size: 1rem !important; }
.top-header .brand             { font-size: 1.5rem !important; }
.top-header .page              { font-size: 1rem !important; }
.top-header .pill              { font-size: 0.72rem !important; }
[data-testid="stCheckbox"] label          { font-size: 1rem !important; }
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label            { font-size: 0.95rem !important; color: #94a3b8 !important; }
[data-testid="stExpander"] summary p      { font-size: 1rem !important; }
div.stButton > button                     { font-size: 0.95rem !important; }
div.stDownloadButton > button             { font-size: 0.95rem !important; }
[data-testid="stCaptionContainer"] p      { font-size: 0.88rem !important; }
[data-testid="stMarkdownContainer"] p     { font-size: 1rem !important; }


/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #1f2333; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #38bdf8; }

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_image(uploaded_file) -> Image.Image:
    """
    Load an uploaded file as a PIL Image.
    Handles JPG, JPEG, PNG, TIFF.
    Returns RGB mode image for consistent processing.
    """
    img = Image.open(uploaded_file)
    # Convert to RGB so all operations work consistently
    # (TIFF/PNG can be RGBA or other modes)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def get_metadata(uploaded_file, img: Image.Image) -> dict:
    """
    Extract metadata from the uploaded file and PIL image.
    Returns a dictionary of metadata fields.

    Fields:
        filename    - original file name
        file_size   - size on disk in KB/MB
        format      - image format (JPEG, PNG, TIFF...)
        dimensions  - width x height in pixels
        channels    - number of color channels (1=Grey, 3=RGB, 4=RGBA)
        resolution  - DPI if available
        mode        - PIL color mode
        crs         - Coordinate Reference System (for GeoTIFF)
        lat_lon     - Geographic coordinates (for GeoTIFF)
    """
    meta = {}

    # File name & size
    meta["filename"] = uploaded_file.name
    size_bytes = uploaded_file.size
    if size_bytes > 1_048_576:
        meta["file_size"] = f"{size_bytes / 1_048_576:.2f} MB"
    else:
        meta["file_size"] = f"{size_bytes / 1024:.1f} KB"

    # Image properties
    meta["format"]     = img.format if img.format else uploaded_file.name.split(".")[-1].upper()
    meta["dimensions"] = f"{img.width} × {img.height} px"
    meta["total_pixels"] = f"{img.width * img.height:,} px"

    # Channels from mode
    mode_channels = {"1": 1, "L": 1, "RGB": 3, "RGBA": 4, "CMYK": 4, "P": 1}
    meta["channels"] = mode_channels.get(img.mode, "—")
    meta["mode"]     = img.mode

    # DPI / Resolution
    try:
        dpi = img.info.get("dpi", None)
        meta["resolution"] = f"{int(dpi[0])} × {int(dpi[1])} DPI" if dpi else "Not embedded"
    except Exception:
        meta["resolution"] = "Not embedded"

    # GeoTIFF — CRS and Lat/Lon
    # These exist only in geo-referenced TIFF files
    try:
        geo_keys = img.tag_v2 if hasattr(img, "tag_v2") else {}
        # Tag 34737 = GeoASCIIParamsTag (CRS name)
        crs_tag = geo_keys.get(34737, None)
        meta["crs"] = crs_tag.strip("|").strip() if crs_tag else "Not a GeoTIFF"

        # Tag 33922 = ModelTiepointTag → contains lon/lat
        tiepoint = geo_keys.get(33922, None)
        if tiepoint and len(tiepoint) >= 6:
            lon = tiepoint[3]
            lat = tiepoint[4]
            meta["lat_lon"] = f"{lat:.6f}°, {lon:.6f}°"
        else:
            meta["lat_lon"] = "Not available"
    except Exception:
        meta["crs"]     = "Not a GeoTIFF"
        meta["lat_lon"] = "Not available"

    return meta


def render_metadata(meta: dict):
    """Render the metadata as a clean card grid in Streamlit."""
    fields = [
        ("📄 Filename",     meta["filename"]),
        ("💾 File Size",    meta["file_size"]),
        ("🖼️ Format",       meta["format"]),
        ("📐 Dimensions",   meta["dimensions"]),
        ("🔢 Total Pixels", meta["total_pixels"]),
        ("🎨 Channels",     str(meta["channels"])),
        ("🔡 Color Mode",   meta["mode"]),
        ("📡 Resolution",   meta["resolution"]),
        ("🌍 CRS",          meta["crs"]),
        ("📍 Lat / Lon",    meta["lat_lon"]),
    ]
    # Render 2-column grid using HTML
    cards_html = '<div class="meta-grid">'
    for label, value in fields:
        cards_html += f"""
        <div class="meta-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>"""
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PROCESSING FUNCTIONS
# Each function takes a PIL Image and returns a PIL Image.
# This makes them chainable in the pipeline.
# ─────────────────────────────────────────────────────────────────────────────

def apply_crop(img: Image.Image, left: int, top: int, right: int, bottom: int) -> Image.Image:
    """
    PIPELINE STEP 1: Crop
    Crops the image to the box defined by (left, top, right, bottom).
    These are pixel coordinates from the top-left corner.
    Validates that the crop box is within image bounds.
    """
    w, h = img.size
    # Clamp values to image bounds — safety check
    left   = max(0, min(left,   w - 1))
    top    = max(0, min(top,    h - 1))
    right  = max(left + 1, min(right,  w))
    bottom = max(top  + 1, min(bottom, h))
    return img.crop((left, top, right, bottom))


def apply_greyscale(img: Image.Image) -> Image.Image:
    """
    PIPELINE STEP 2: Greyscale
    Converts image to greyscale using scikit-image's rgb2gray.
    rgb2gray uses luminosity weights: 0.2126R + 0.7152G + 0.0722B
    (matches human eye sensitivity — better than simple averaging).
    Result is converted back to PIL RGB for pipeline consistency.
    """
    np_img = np.array(img)
    if np_img.ndim == 3:
        grey = color.rgb2gray(np_img)               # float64, range [0,1]
        grey_uint8 = (grey * 255).astype(np.uint8)  # convert back to 0-255
        return Image.fromarray(grey_uint8, mode="L").convert("RGB")
    return img  # already greyscale


def apply_blur(img: Image.Image, sigma: float) -> Image.Image:
    """
    PIPELINE STEP 3: Gaussian Blur
    Uses scikit-image's gaussian filter.
    sigma = blur intensity (higher = more blurry).
    multichannel=True processes each RGB channel separately.
    """
    np_img = np.array(img).astype(np.float64) / 255.0
    if np_img.ndim == 3:
        blurred = gaussian(np_img, sigma=sigma, channel_axis=-1)
    else:
        blurred = gaussian(np_img, sigma=sigma)
    blurred_uint8 = (blurred * 255).astype(np.uint8)
    return Image.fromarray(blurred_uint8)


def apply_brightness(img: Image.Image, factor: float) -> Image.Image:
    """
    PIPELINE STEP 4: Brightness
    Uses PIL's ImageEnhance.Brightness.
    factor = 1.0 → original | < 1.0 → darker | > 1.0 → brighter
    """
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def apply_contrast(img: Image.Image, factor: float) -> Image.Image:
    """
    PIPELINE STEP 5: Contrast
    Uses PIL's ImageEnhance.Contrast.
    factor = 1.0 → original | < 1.0 → flat | > 1.0 → more contrast
    """
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def run_pipeline(img: Image.Image, settings: dict) -> Image.Image:
    """
    MAIN PIPELINE RUNNER
    Applies enabled edits in fixed order:
        Crop → Greyscale → Blur → Brightness → Contrast

    Why fixed order?
        - Crop first = you work on less data (faster)
        - Greyscale before blur = blur on 1 channel (faster)
        - Brightness & Contrast last = final look adjustment

    Args:
        img      : original PIL Image
        settings : dict of all edit parameters from the UI

    Returns:
        processed PIL Image
    """
    result = img.copy()

    # STEP 1 — CROP
    if settings.get("crop_enabled"):
        result = apply_crop(
            result,
            settings["crop_left"],
            settings["crop_top"],
            settings["crop_right"],
            settings["crop_bottom"],
        )

    # STEP 2 — GREYSCALE
    if settings.get("grey_enabled"):
        result = apply_greyscale(result)

    # STEP 3 — BLUR
    if settings.get("blur_enabled"):
        result = apply_blur(result, settings["blur_sigma"])

    # STEP 4 — BRIGHTNESS
    if settings.get("brightness_enabled"):
        result = apply_brightness(result, settings["brightness_factor"])

    # STEP 5 — CONTRAST
    if settings.get("contrast_enabled"):
        result = apply_contrast(result, settings["contrast_factor"])

    return result


def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert PIL Image to bytes for download button."""
    buf = io.BytesIO()
    save_fmt = "JPEG" if fmt == "JPG" else fmt
    if save_fmt == "JPEG" and img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format=save_fmt)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Brand + Upload + Edit Controls
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:

    # Brand — compact at top
    st.markdown('<div class="sb-brand"><div class="name">ImgKit</div><div class="sub">v1.0 · Image Manipulation Tool</div></div>', unsafe_allow_html=True)

    # ── UPLOAD ────────────────────────────────────────────────
    st.markdown('<div class="section-label sl-upload">01 · Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="Choose an image",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="Supported: JPG, JPEG, PNG, TIFF"
    )

    # Only show controls if an image is uploaded
    if uploaded_file:

        img_original = load_image(uploaded_file)
        w, h = img_original.size

        # ── METADATA TOGGLE ───────────────────────────────────
        st.markdown('<div class="section-label sl-meta">02 · Metadata</div>', unsafe_allow_html=True)
        show_meta = st.toggle("Show image metadata", value=False)

        # ── PIPELINE CONTROLS ─────────────────────────────────
        st.markdown('<div class="section-label sl-edit">03 · Edit Pipeline</div>', unsafe_allow_html=True)
        st.markdown('<span class="pipeline-badge">PIPELINE · FIXED ORDER</span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">Enable any combination of edits below. '
            'They apply in this order:<br>'
            '<b>Crop → Greyscale → Blur → Brightness → Contrast</b></div>',
            unsafe_allow_html=True
        )

        settings = {}

        # ── CROP ──────────────────────────────────────────────
        with st.expander("✂️ Crop", expanded=False):
            settings["crop_enabled"] = st.checkbox("Enable Crop", key="crop_en")
            st.markdown(
                '<div class="info-box">'
                '<b>How cropping works:</b><br>'
                'Enter pixel coordinates from the <b>top-left corner (0,0)</b> of the image.<br>'
                '• <b>Left</b> = start column &nbsp;• <b>Top</b> = start row<br>'
                '• <b>Right</b> = end column &nbsp;• <b>Bottom</b> = end row<br>'
                f'Your image is <b>{w} × {h} px</b>. '
                f'So Right max = {w}, Bottom max = {h}.'
                '</div>',
                unsafe_allow_html=True
            )
            col1, col2 = st.columns(2)
            with col1:
                settings["crop_left"]   = st.number_input("Left",   0, w-1, 0,   step=1)
                settings["crop_right"]  = st.number_input("Right",  1, w,   w,   step=1)
            with col2:
                settings["crop_top"]    = st.number_input("Top",    0, h-1, 0,   step=1)
                settings["crop_bottom"] = st.number_input("Bottom", 1, h,   h,   step=1)

        # ── GREYSCALE ─────────────────────────────────────────
        with st.expander("⬜ Greyscale", expanded=False):
            settings["grey_enabled"] = st.checkbox("Enable Greyscale", key="grey_en")
            st.markdown(
                '<div class="info-box">Converts to greyscale using luminosity weights '
                '(scikit-image rgb2gray). Preserves perceived brightness better than '
                'simple averaging.</div>',
                unsafe_allow_html=True
            )

        # ── BLUR ──────────────────────────────────────────────
        with st.expander("💧 Blur", expanded=False):
            settings["blur_enabled"] = st.checkbox("Enable Blur", key="blur_en")
            settings["blur_sigma"]   = st.slider(
                "Blur intensity (sigma)", 0.5, 10.0, 1.5, step=0.5,
                help="Higher = more blurry. Uses Gaussian blur (scikit-image)."
            )
            st.markdown(
                '<div class="info-box">Gaussian blur smooths the image by averaging '
                'each pixel with its neighbors. Sigma controls the spread of the blur.</div>',
                unsafe_allow_html=True
            )

        # ── BRIGHTNESS ────────────────────────────────────────
        with st.expander("☀️ Brightness", expanded=False):
            settings["brightness_enabled"] = st.checkbox("Enable Brightness", key="bright_en")
            settings["brightness_factor"]  = st.slider(
                "Brightness", 0.1, 3.0, 1.0, step=0.05,
                help="1.0 = original. < 1.0 = darker. > 1.0 = brighter."
            )

        # ── CONTRAST ──────────────────────────────────────────
        with st.expander("🔘 Contrast", expanded=False):
            settings["contrast_enabled"] = st.checkbox("Enable Contrast", key="cont_en")
            settings["contrast_factor"]  = st.slider(
                "Contrast", 0.1, 3.0, 1.0, step=0.05,
                help="1.0 = original. < 1.0 = flat/grey. > 1.0 = more vivid."
            )

        # ── APPLY + EXPORT ────────────────────────────────────
        st.markdown('<div class="section-label sl-export">04 · Export</div>', unsafe_allow_html=True)
        apply_btn   = st.button("▶  Apply Pipeline")
        export_fmt  = st.selectbox("Save as", ["PNG", "JPG", "TIFF"], index=0)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA — Image display & results
# ─────────────────────────────────────────────────────────────────────────────

# Header bar
st.markdown("""
<div class="top-header">
    <span class="brand">ImgKit</span>
    <span class="sep">/</span>
    <span class="page">Workspace</span>
    <span class="pill">v1.0</span>
</div>
""", unsafe_allow_html=True)

if not uploaded_file:
    # Empty state
    st.markdown("""
    <div style='
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; height: 60vh;
        border: 1px dashed #2a2d3a; border-radius: 12px;
        color: #334155; text-align: center; padding: 2rem;
    '>
        <div style='font-size: 3rem; margin-bottom: 1rem;'>🖼️</div>
        <div style='font-family: Space Mono, monospace; font-size: 1rem; color: #4f8ef7;'>
            No image loaded
        </div>
        <div style='font-size: 0.85rem; color: #475569; margin-top: 0.5rem;'>
            Upload a JPG, PNG, or TIFF from the sidebar to get started
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── METADATA ────────────────────────────────────────────
    if show_meta:
        st.markdown('<div class="section-label">Image Metadata</div>', unsafe_allow_html=True)
        meta = get_metadata(uploaded_file, img_original)
        render_metadata(meta)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── IMAGE DISPLAY ───────────────────────────────────────
    # Check if any edit is enabled
    any_edit_enabled = any([
        settings.get("crop_enabled"),
        settings.get("grey_enabled"),
        settings.get("blur_enabled"),
        settings.get("brightness_enabled"),
        settings.get("contrast_enabled"),
    ])

    if not apply_btn:
        # Show original before Apply is clicked
        st.markdown('<div class="section-label">Original Image</div>', unsafe_allow_html=True)
        st.markdown('<div class="img-card">', unsafe_allow_html=True)
        # clamp_to_height: display at max 65vh — use_container_width=False + CSS handles it
        st.image(img_original, use_container_width=False, width=min(img_original.width, 900))
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption(f"📐 {img_original.width} × {img_original.height} px  ·  {uploaded_file.name}")

        if any_edit_enabled:
            st.info("✓ Pipeline configured — hit **▶ Apply Pipeline** in the sidebar to process.")

    else:
        # Run the pipeline and show result
        with st.spinner("Running pipeline..."):
            result_img = run_pipeline(img_original, settings)

        st.markdown('<div class="section-label">Processed Image</div>', unsafe_allow_html=True)

        # Show which steps were applied
        steps_applied = []
        if settings.get("crop_enabled"):       steps_applied.append("Crop")
        if settings.get("grey_enabled"):       steps_applied.append("Greyscale")
        if settings.get("blur_enabled"):       steps_applied.append(f"Blur (σ={settings['blur_sigma']})")
        if settings.get("brightness_enabled"): steps_applied.append(f"Brightness (×{settings['brightness_factor']})")
        if settings.get("contrast_enabled"):   steps_applied.append(f"Contrast (×{settings['contrast_factor']})")

        if steps_applied:
            pipeline_str = " → ".join(steps_applied)
            st.markdown(
                f'<div class="info-box"><b>Pipeline applied:</b> {pipeline_str}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="info-box">No edits were enabled. Showing original image.</div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="img-card">', unsafe_allow_html=True)
        st.image(result_img, use_container_width=False, width=min(result_img.width, 900))
        st.markdown('</div>', unsafe_allow_html=True)

        # ── DOWNLOAD ──────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        fname_base = os.path.splitext(uploaded_file.name)[0]
        out_filename = f"{fname_base}_imgkit.{export_fmt.lower()}"
        img_bytes = image_to_bytes(result_img, export_fmt)

        st.download_button(
            label=f"⬇  Download as {export_fmt}",
            data=img_bytes,
            file_name=out_filename,
            mime=f"image/{export_fmt.lower()}",
        )
        st.caption(f"Output: {result_img.size[0]} × {result_img.size[1]} px · {export_fmt}")