"""
╔══════════════════════════════════════════════════════════════╗
║                        ImgKit v2.1                           ║
║         Lightweight Image Manipulation Tool                  ║
║         Built with: Python · Streamlit · scikit-image        ║
╚══════════════════════════════════════════════════════════════╝
SDLC PHASE: Development  |  Version: 2.1  |  Status: Active

CHANGELOG v2.1:
    + Multi-output mode (1-4 outputs, each independent)
    + Each output slot has own edit settings
    + All outputs shown side by side, individual download
    + Upload once, get multiple processed versions

PIPELINE ORDER (per output):
    Crop → Greyscale → Blur → Brightness → Contrast → Resample
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import io
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance
from skimage import color
from skimage.filters import gaussian
from skimage.transform import resize as skimage_resize


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ImgKit",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:      #0d0f18;
    --surface: #13161f;
    --border:  #1f2333;
    --text:    #e2e8f0;
    --muted:   #64748b;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text); font-size: 16px;
}
[data-testid="stAppViewContainer"] > .main > div:first-child { padding-top: 0 !important; }
[data-testid="block-container"] { padding-top: 0.5rem !important; }

[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] > div:first-child { padding-top: 0.6rem !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; font-size: 15px !important; }

.sb-brand { padding: 6px 0 8px 0; border-bottom: 1px solid var(--border); margin-bottom: 6px; }
.sb-brand .name {
    font-family: 'Space Mono', monospace; font-size: 1.4rem !important; font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.sb-brand .sub { font-size: 0.72rem !important; color: #475569; letter-spacing: 2px; text-transform: uppercase; }

.section-label {
    font-family: 'Space Mono', monospace; font-size: 0.75rem !important;
    letter-spacing: 2px; text-transform: uppercase;
    padding-bottom: 3px; margin: 10px 0 5px 0;
    border-bottom: 1px solid var(--border); color: var(--muted);
}
.sl-upload { color: #38bdf8 !important; border-color: #38bdf8 !important; }
.sl-meta   { color: #34d399 !important; border-color: #34d399 !important; }
.sl-edit   { color: #a78bfa !important; border-color: #a78bfa !important; }
.sl-export { color: #fb923c !important; border-color: #fb923c !important; }
.sl-bit    { color: #f472b6 !important; border-color: #f472b6 !important; }
.sl-multi  { color: #fbbf24 !important; border-color: #fbbf24 !important; }

.pipeline-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1e3a5f, #2d1b69);
    border: 1px solid #38bdf8; color: #38bdf8 !important;
    font-family: 'Space Mono', monospace; font-size: 0.72rem !important;
    padding: 3px 8px; border-radius: 4px; letter-spacing: 1px; margin-bottom: 4px;
}

.info-box {
    background: #0f1628; border-left: 3px solid #38bdf8;
    padding: 8px 12px; border-radius: 0 6px 6px 0;
    font-size: 0.9rem !important; color: #94a3b8; margin: 4px 0; line-height: 1.5;
}

/* Output slot card */
.output-card {
    background: #0f1120;
    border: 1px solid #1f2333;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 8px;
}
.output-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #fbbf24;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 1px solid #fbbf24;
    padding-bottom: 4px;
    margin-bottom: 8px;
}

[data-testid="stExpander"] {
    background: #0f1120 !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; margin-bottom: 3px !important;
}
[data-testid="stExpander"]:hover { border-color: #38bdf8 !important; }
[data-testid="stExpander"] summary p { font-size: 1rem !important; }

div.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #7c3aed) !important;
    color: white !important; border: none !important; border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.9rem !important;
    letter-spacing: 1px !important; width: 100%;
    box-shadow: 0 0 12px rgba(56,189,248,0.25); transition: all 0.2s;
}
div.stButton > button:hover { box-shadow: 0 0 22px rgba(56,189,248,0.5) !important; }

div.stDownloadButton > button {
    background: linear-gradient(135deg, #065f46, #064e3b) !important;
    color: #34d399 !important; border: 1px solid #34d399 !important;
    border-radius: 6px !important; font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important; width: 100%;
}

.top-header {
    background: linear-gradient(90deg, #0d0f18, #13161f 50%, #0d0f18);
    border-bottom: 1px solid #1f2333; padding: 10px 20px;
    display: flex; align-items: center; gap: 10px; margin-bottom: 14px;
}
.top-header .brand {
    font-family: 'Space Mono', monospace; font-size: 1.4rem !important; font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.top-header .sep  { color: #2a2d3a; }
.top-header .page { color: #64748b; font-size: 0.95rem !important; }
.top-header .pill {
    margin-left: auto; background: #1a1d2e; border: 1px solid #38bdf8; color: #38bdf8;
    font-family: 'Space Mono', monospace; font-size: 0.68rem !important;
    padding: 2px 9px; border-radius: 20px; letter-spacing: 2px;
}

.meta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 5px; margin-top: 5px; }
.meta-card {
    background: #0f1120; border: 1px solid #1f2333;
    border-radius: 7px; padding: 7px 11px; transition: border-color 0.2s;
}
.meta-card:hover { border-color: #38bdf8; }
.meta-card .label { font-size: 0.68rem !important; letter-spacing: 2px; color: #475569; text-transform: uppercase; }
.meta-card .value { font-family: 'Space Mono', monospace; font-size: 0.9rem !important; color: #38bdf8; margin-top: 2px; word-break: break-all; }

.img-card {
    background: #0f1120; border: 1px solid #1f2333;
    border-radius: 12px; padding: 10px;
    display: flex; justify-content: center; align-items: center;
}
[data-testid="stImage"] img {
    border-radius: 8px; border: 1px solid #1f2333;
    max-height: 55vh !important; max-width: 100% !important;
    width: auto !important; object-fit: contain;
    box-shadow: 0 0 30px rgba(56,189,248,0.07);
}

.feature-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 20px 0; }
.feature-card {
    background: #FDFBD4; border: 1px solid #1f2333; border-radius: 10px;
    padding: 16px; text-align: center; transition: border-color 0.2s, transform 0.2s;
}
.feature-card:hover { border-color: #38bdf8; transform: translateY(-2px); }
.feature-card .icon  { font-size: 1.8rem; margin-bottom: 8px; }
.feature-card .title { font-family: 'Space Mono', monospace; font-size: 0.82rem; color: #38bdf8; margin-bottom: 4px; }
.feature-card .desc  { font-size: 0.78rem; color: #64748b; line-height: 1.4; }

[data-testid="stCheckbox"] label  { font-size: 1rem !important; }
[data-testid="stSlider"] label    { font-size: 0.92rem !important; color: #94a3b8 !important; }
[data-testid="stSelectbox"] label { font-size: 0.92rem !important; color: #94a3b8 !important; }
[data-testid="stCaptionContainer"] p  { font-size: 0.85rem !important; }
[data-testid="stMarkdownContainer"] p { font-size: 1rem !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #1f2333; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #38bdf8; }

#MainMenu, footer, header { visibility: hidden; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RESAMPLE METHODS
# ─────────────────────────────────────────────────────────────────────────────
RESAMPLE_METHODS = {
    "Nearest Neighbour": 0,
    "Bilinear"         : 1,
    "Biquadratic"      : 2,
    "Bicubic"          : 3,
    "Biquartic"        : 4,
    "Biquintic"        : 5,
}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD IMAGE
# ─────────────────────────────────────────────────────────────────────────────
def load_image(uploaded_file) -> Image.Image:
    filename = uploaded_file.name.lower()
    if filename.endswith((".tif", ".tiff")):
        try:
            uploaded_file.seek(0)
            img = Image.open(uploaded_file)
            img.load()
            return img.convert("RGB")
        except Exception:
            pass
        try:
            import rasterio
            uploaded_file.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            with rasterio.open(tmp_path) as src:
                def norm(b):
                    b = b.astype(np.float32)
                    mn, mx = b.min(), b.max()
                    return np.zeros_like(b, dtype=np.uint8) if mx == mn else ((b-mn)/(mx-mn)*255).astype(np.uint8)
                if src.count >= 3:
                    img = Image.fromarray(np.stack([norm(src.read(i)) for i in [1,2,3]], axis=-1), "RGB")
                else:
                    img = Image.fromarray(norm(src.read(1)), "L").convert("RGB")
            os.unlink(tmp_path)
            return img
        except Exception as e:
            st.error(f"Could not open TIFF: {e}")
            st.stop()
    else:
        uploaded_file.seek(0)
        return Image.open(uploaded_file).convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# METADATA
# ─────────────────────────────────────────────────────────────────────────────
def get_metadata(uploaded_file, img):
    meta = {}
    meta["filename"]     = uploaded_file.name
    sz = uploaded_file.size
    meta["file_size"]    = f"{sz/1_048_576:.2f} MB" if sz > 1_048_576 else f"{sz/1024:.1f} KB"
    meta["format"]       = uploaded_file.name.split(".")[-1].upper()
    meta["dimensions"]   = f"{img.width} × {img.height} px"
    meta["total_pixels"] = f"{img.width * img.height:,} px"
    meta["channels"]     = str({"1":1,"L":1,"RGB":3,"RGBA":4,"CMYK":4,"P":1}.get(img.mode,"—"))
    meta["mode"]         = img.mode
    try:
        dpi = img.info.get("dpi")
        meta["resolution"] = f"{int(dpi[0])} × {int(dpi[1])} DPI" if dpi else "Not embedded"
    except Exception:
        meta["resolution"] = "Not embedded"
    if uploaded_file.name.lower().endswith((".tif", ".tiff")):
        try:
            import rasterio
            from rasterio.warp import transform as wt
            uploaded_file.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
                tmp.write(uploaded_file.read()); tmp_path = tmp.name
            with rasterio.open(tmp_path) as src:
                meta["crs"] = str(src.crs) if src.crs else "Not available"
                res = src.res
                meta["resolution"] = f"{res[0]:.4f} × {res[1]:.4f} (CRS units/px)"
                if src.crs:
                    cx = (src.bounds.left + src.bounds.right) / 2
                    cy = (src.bounds.top  + src.bounds.bottom) / 2
                    lon, lat = wt(src.crs, "EPSG:4326", [cx], [cy])
                    meta["lat_lon"] = f"{lat[0]:.6f}°, {lon[0]:.6f}°"
                else:
                    meta["lat_lon"] = "Not available"
            os.unlink(tmp_path)
        except Exception:
            meta["crs"] = meta["lat_lon"] = "Not available"
    else:
        meta["crs"] = "Not a GeoTIFF"
        meta["lat_lon"] = "Not available"
    return meta


def render_metadata(meta):
    fields = [
        ("📄 Filename", meta["filename"]), ("💾 File Size", meta["file_size"]),
        ("🖼️ Format", meta["format"]),    ("📐 Dimensions", meta["dimensions"]),
        ("🔢 Total Pixels", meta["total_pixels"]), ("🎨 Channels", meta["channels"]),
        ("🔡 Color Mode", meta["mode"]),  ("📡 Resolution", meta["resolution"]),
        ("🌍 CRS", meta["crs"]),          ("📍 Lat / Lon", meta["lat_lon"]),
    ]
    html = '<div class="meta-grid">'
    for label, value in fields:
        html += f'<div class="meta-card"><div class="label">{label}</div><div class="value">{value}</div></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def apply_crop(img, left, top, right, bottom):
    w, h = img.size
    l = int(left/100*w); t = int(top/100*h)
    r = int(right/100*w); b = int(bottom/100*h)
    l, r = max(0,l), min(w, max(r, l+1))
    t, b = max(0,t), min(h, max(b, t+1))
    return img.crop((l, t, r, b))

def apply_greyscale(img):
    arr = np.array(img)
    if arr.ndim == 3:
        grey = (color.rgb2gray(arr)*255).astype(np.uint8)
        return Image.fromarray(grey,"L").convert("RGB")
    return img

def apply_blur(img, sigma):
    arr = np.array(img).astype(np.float64)/255.0
    blurred = gaussian(arr, sigma=sigma, channel_axis=-1) if arr.ndim==3 else gaussian(arr, sigma=sigma)
    return Image.fromarray((blurred*255).astype(np.uint8))

def apply_brightness(img, factor):
    return ImageEnhance.Brightness(img).enhance(factor)

def apply_contrast(img, factor):
    return ImageEnhance.Contrast(img).enhance(factor)

def apply_resample(img, scale_pct, method_name):
    order = RESAMPLE_METHODS[method_name]
    arr   = np.array(img).astype(np.float64)/255.0
    w, h  = img.size
    nw    = max(1, int(w*scale_pct/100))
    nh    = max(1, int(h*scale_pct/100))
    resized = skimage_resize(
        arr, (nh, nw, arr.shape[2]) if arr.ndim==3 else (nh, nw),
        order=order, anti_aliasing=(scale_pct<100), mode='reflect'
    )
    return Image.fromarray((resized*255).astype(np.uint8))

def run_pipeline(img, settings):
    result = img.copy()
    if settings.get("crop_enabled"):
        result = apply_crop(result, settings["crop_left"], settings["crop_top"],
                            settings["crop_right"], settings["crop_bottom"])
    if settings.get("grey_enabled"):
        result = apply_greyscale(result)
    if settings.get("blur_enabled"):
        result = apply_blur(result, settings["blur_sigma"])
    if settings.get("brightness_enabled"):
        result = apply_brightness(result, settings["brightness_factor"])
    if settings.get("contrast_enabled"):
        result = apply_contrast(result, settings["contrast_factor"])
    if settings.get("resample_enabled"):
        result = apply_resample(result, settings["resample_scale"], settings["resample_method"])
    return result

def image_to_bytes(img, fmt="PNG"):
    buf = io.BytesIO()
    save_fmt = "JPEG" if fmt == "JPG" else fmt
    if save_fmt == "JPEG" and img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format=save_fmt)
    return buf.getvalue()

def apply_bit_depth(img, target_bits):
    arr = np.array(img)
    if target_bits == 8:
        return arr.astype(np.uint8)
    elif target_bits == 16:
        return (arr.astype(np.uint16)*257)
    elif target_bits == 32:
        return arr.astype(np.float32)/255.0


# ─────────────────────────────────────────────────────────────────────────────
# HISTOGRAM
# ─────────────────────────────────────────────────────────────────────────────
def render_histogram(img, key_suffix=""):
    arr = np.array(img)
    c1, c2, c3 = st.columns(3)
    with c1:
        bin_size = st.selectbox("Bin Range", [5,10,20,50], index=1, key=f"bin_{key_suffix}")
    with c2:
        stretch_factor = st.selectbox("Stretch Factor", [1,1.5,2,2.5,3,3.5], index=0, key=f"str_{key_suffix}")
    with c3:
        st.markdown("**Channels**")
        ca,cb,cc = st.columns(3)
        with ca: show_r = st.checkbox("R", value=True, key=f"r_{key_suffix}")
        with cb: show_g = st.checkbox("G", value=True, key=f"g_{key_suffix}")
        with cc: show_b = st.checkbox("B", value=True, key=f"b_{key_suffix}")

    if stretch_factor != 1:
        arr_f = arr.astype(np.float32)
        if arr_f.ndim == 3:
            stretched = np.zeros_like(arr_f)
            for ch in range(arr_f.shape[2]):
                mn, mx = arr_f[:,:,ch].min(), arr_f[:,:,ch].max()
                if mx > mn:
                    stretched[:,:,ch] = (arr_f[:,:,ch]-mn)/(mx-mn)*255*stretch_factor
            arr = np.clip(stretched,0,255).astype(np.uint8)
        else:
            mn, mx = arr_f.min(), arr_f.max()
            arr = np.clip((arr_f-mn)/(mx-mn)*255*stretch_factor,0,255).astype(np.uint8)

    bins = np.arange(0, 256+bin_size, bin_size)
    fig, ax = plt.subplots(figsize=(9,3))
    fig.patch.set_facecolor("#0f1120")
    ax.set_facecolor("#0f1120")
    if arr.ndim == 3:
        if show_r: ax.hist(arr[:,:,0].ravel(), bins=bins, color="#ef4444", alpha=0.6, label="Red")
        if show_g: ax.hist(arr[:,:,1].ravel(), bins=bins, color="#22c55e", alpha=0.6, label="Green")
        if show_b: ax.hist(arr[:,:,2].ravel(), bins=bins, color="#3b82f6", alpha=0.6, label="Blue")
        if any([show_r,show_g,show_b]):
            ax.legend(fontsize=8, facecolor="#13161f", labelcolor="white", framealpha=0.8)
    else:
        ax.hist(arr.ravel(), bins=bins, color="#94a3b8", alpha=0.8)
    bin_labels = [f"{i}-{i+bin_size-1}" for i in range(0,256,bin_size)]
    ax.set_xticks(bins[:-1]+bin_size/2)
    ax.set_xticklabels(bin_labels[:len(bins)-1], rotation=90, fontsize=6, color="#64748b")
    ax.set_xlabel("Pixel Value Range", color="#64748b", fontsize=9)
    ax.set_ylabel("Pixel Count",       color="#64748b", fontsize=9)
    ax.set_xlim(0,255)
    ax.tick_params(axis='y', colors="#475569", labelsize=7)
    for spine in ax.spines.values(): spine.set_edgecolor("#1f2333")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT SLOT SETTINGS
# Renders all edit controls for one output slot
# Returns settings dict for that slot
# ─────────────────────────────────────────────────────────────────────────────
def render_output_slot_settings(slot_num, w, h):
    """
    Renders edit controls for one output slot inside the sidebar.
    slot_num = 1,2,3,4 (used as unique key suffix)
    w, h     = original image dimensions
    Returns  : settings dict for this slot
    """
    s = {}   # settings dictionary for this slot
    k = f"s{slot_num}"  # unique key prefix

    # Color per slot for visual separation
    colors = {1:"#38bdf8", 2:"#a78bfa", 3:"#34d399", 4:"#fb923c"}
    c = colors.get(slot_num, "#38bdf8")

    st.markdown(
        f'<div style="font-family:Space Mono,monospace; font-size:0.72rem; '
        f'color:{c}; border-bottom:1px solid {c}; padding-bottom:3px; '
        f'margin:8px 0 5px 0; letter-spacing:2px;">OUTPUT {slot_num}</div>',
        unsafe_allow_html=True
    )

    with st.expander(f"✂️ Crop", expanded=False):
        s["crop_enabled"] = st.checkbox("Enable", key=f"{k}_crop")
        s["crop_left"]    = st.slider("Left %",   0, 99,  0,  key=f"{k}_cl")
        s["crop_right"]   = st.slider("Right %",  1, 100, 100,key=f"{k}_cr")
        s["crop_top"]     = st.slider("Top %",    0, 99,  0,  key=f"{k}_ct")
        s["crop_bottom"]  = st.slider("Bottom %", 1, 100, 100,key=f"{k}_cb")

    with st.expander(f"⬜ Greyscale", expanded=False):
        s["grey_enabled"] = st.checkbox("Enable", key=f"{k}_grey")

    with st.expander(f"💧 Blur", expanded=False):
        s["blur_enabled"] = st.checkbox("Enable", key=f"{k}_blur")
        s["blur_sigma"]   = st.slider("Sigma", 0.5, 10.0, 1.5, step=0.5, key=f"{k}_bs")

    with st.expander(f"☀️ Brightness", expanded=False):
        s["brightness_enabled"] = st.checkbox("Enable", key=f"{k}_bright")
        s["brightness_factor"]  = st.slider("Factor", 0.1, 3.0, 1.0, step=0.05, key=f"{k}_bf")

    with st.expander(f"🔘 Contrast", expanded=False):
        s["contrast_enabled"] = st.checkbox("Enable", key=f"{k}_cont")
        s["contrast_factor"]  = st.slider("Factor", 0.1, 3.0, 1.0, step=0.05, key=f"{k}_cf")

    with st.expander(f"🔄 Resample", expanded=False):
        s["resample_enabled"] = st.checkbox("Enable", key=f"{k}_res")
        s["resample_scale"]   = st.slider("Scale %", 10, 200, 100, step=5, key=f"{k}_rs")
        s["resample_method"]  = st.selectbox("Method", list(RESAMPLE_METHODS.keys()), key=f"{k}_rm")

    return s


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown(
        '<div class="sb-brand">'
        '<div class="name">ImgKit</div>'
        '<div class="sub">v2.1 · Image Manipulation Tool</div>'
        '</div>', unsafe_allow_html=True
    )

    # 01 UPLOAD
    st.markdown('<div class="section-label sl-upload">01 · Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg","jpeg","png","tif","tiff"],
        help="JPG, JPEG, PNG, TIFF (including multi-band GeoTIFF)"
    )

    if uploaded_file:
        # Clear old results if a new file is uploaded
        if "last_file" not in st.session_state or st.session_state["last_file"] != uploaded_file.name:
            st.session_state["results"]     = {}
            st.session_state["last_file"]   = uploaded_file.name
        img_original = load_image(uploaded_file)
        w, h = img_original.size

        # 02 METADATA + HISTOGRAM
        st.markdown('<div class="section-label sl-meta">02 · Info</div>', unsafe_allow_html=True)
        show_meta = st.toggle("Show metadata", value=False)
        show_hist = st.toggle("Show histogram", value=False)

        # 03 MULTI OUTPUT
        st.markdown('<div class="section-label sl-multi">03 · Multi Output</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">Upload once → get multiple processed versions.<br>'
            'Each output has <b>independent edit settings.</b></div>',
            unsafe_allow_html=True
        )
        num_outputs = st.slider("Number of outputs", 1, 4, 2, key="num_out")

        # Render settings for each output slot
        all_settings = {}
        for i in range(1, num_outputs+1):
            all_settings[i] = render_output_slot_settings(i, w, h)

        # 04 APPLY + EXPORT
        st.markdown('<div class="section-label sl-export">04 · Export</div>', unsafe_allow_html=True)
        apply_btn  = st.button("▶  Apply All Outputs")
        export_fmt = st.selectbox("Save as", ["PNG","JPG","TIFF"], index=0)

        # 05 BIT DEPTH
        st.markdown('<div class="section-label sl-bit">05 · Bit Depth</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">'
            '<b>8-bit</b> = 0–255 standard<br>'
            '<b>16-bit</b> = 0–65535 GeoTIFF<br>'
            '<b>32-bit</b> = 0.0–1.0 scientific'
            '</div>', unsafe_allow_html=True
        )
        bit_target  = st.selectbox("Convert to", ["8-bit","16-bit","32-bit float"], key="bit_sel")
        convert_btn = st.button("⚙  Convert Bit Depth", key="bit_btn")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-header">
    <span class="brand">ImgKit</span>
    <span class="sep">/</span>
    <span class="page">Workspace</span>
    <span class="pill">v2.1</span>
</div>
""", unsafe_allow_html=True)

if not uploaded_file:
    st.markdown("""
    <div style='text-align:center; padding:2rem 0 1rem 0;'>
        <div style='font-family:Space Mono,monospace; font-size:2rem;
            background:linear-gradient(90deg,#38bdf8,#a78bfa,#34d399);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:6px;'>
            ImgKit
        </div>
        <div style='color:#FDFBD4; font-size:0.95rem; letter-spacing:2px; text-transform:uppercase;'>
            - An Easy Img Manipulation Tool!
        </div>
    </div>
    <div class="feature-grid">
        <div class="feature-card"><div class="icon">✂️</div><div class="title">CROP</div><div class="desc">Slider-based crop with live pixel preview</div></div>
        <div class="feature-card"><div class="icon">⬜</div><div class="title">GREYSCALE</div><div class="desc">Luminosity-weighted via scikit-image rgb2gray</div></div>
        <div class="feature-card"><div class="icon">💧</div><div class="title">BLUR</div><div class="desc">Gaussian blur with adjustable sigma</div></div>
        <div class="feature-card"><div class="icon">☀️</div><div class="title">BRIGHTNESS</div><div class="desc">PIL brightness from 0.1× to 3.0×</div></div>
        <div class="feature-card"><div class="icon">🔘</div><div class="title">CONTRAST</div><div class="desc">Adjust light/dark separation</div></div>
        <div class="feature-card"><div class="icon">🔄</div><div class="title">RESAMPLE</div><div class="desc">6 methods — Nearest to Biquintic</div></div>
        <div class="feature-card"><div class="icon">📊</div><div class="title">HISTOGRAM</div><div class="desc">Interactive bins, channel toggles, stretch</div></div>
        <div class="feature-card"><div class="icon">🎚️</div><div class="title">BIT DEPTH</div><div class="desc">8 / 16 / 32-bit conversion</div></div>
        <div class="feature-card"><div class="icon">🖼️</div><div class="title">MULTI OUTPUT</div><div class="desc">1-4 outputs from one upload, each independent</div></div>
    </div>
    <div style='text-align:center; margin-top:1.5rem; border:1px dashed #1f2333; border-radius:12px; padding:1.5rem;'>
        <div style='font-size:2rem; margin-bottom:8px;'>⬆️</div>
        <div style='font-family:Space Mono,monospace; color:#38bdf8; font-size:0.95rem;'>
            Upload an image from the sidebar to get started
        </div>
        <div style='color:#FDFBD4; font-size:0.82rem; margin-top:4px;'>ImgKIT is an image manipulation tool with a fixed pipeline architecture, GeoTIFF support via rasterio, multi-output processing, and interactive histogram analysis.</div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── ORIGINAL IMAGE ────────────────────────────────────────
    st.markdown('<div class="section-label sl-upload">Original Image</div>', unsafe_allow_html=True)
    st.markdown('<div class="img-card">', unsafe_allow_html=True)
    st.image(img_original, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption(f"📐 {w} × {h} px  ·  {uploaded_file.name}")

    # ── METADATA ──────────────────────────────────────────────
    if show_meta:
        st.markdown('<div class="section-label sl-meta">Image Metadata</div>', unsafe_allow_html=True)
        render_metadata(get_metadata(uploaded_file, img_original))
        st.markdown("<br>", unsafe_allow_html=True)

    # ── HISTOGRAM (original) ──────────────────────────────────
    if show_hist:
        st.markdown('<div class="section-label sl-meta">Histogram — Original</div>', unsafe_allow_html=True)
        render_histogram(img_original, key_suffix="orig")

    # ── MULTI OUTPUT RESULTS ──────────────────────────────────
    # Store results in session state so downloads don't clear them
    if apply_btn:
        st.session_state["results"]      = {}
        st.session_state["num_outputs"]  = num_outputs
        st.session_state["all_settings"] = {i: dict(all_settings[i]) for i in all_settings}
        for i in range(1, num_outputs+1):
            with st.spinner(f"Processing Output {i}..."):
                st.session_state["results"][i] = run_pipeline(img_original, all_settings[i])

    if "results" in st.session_state and st.session_state["results"]:
        results      = st.session_state["results"]
        num_outputs  = st.session_state["num_outputs"]
        all_settings = st.session_state["all_settings"]
        st.markdown('<div class="section-label sl-multi">Processed Outputs</div>', unsafe_allow_html=True)

        # Display outputs in columns (max 2 per row for readability)
        if num_outputs == 1:
            cols = [st.columns(1)[0]]
        elif num_outputs == 2:
            cols = st.columns(2)
        elif num_outputs == 3:
            cols = st.columns(3)
        else:
            cols = st.columns(2)   # 4 outputs = 2 rows of 2

        slot_colors = {1:"#38bdf8", 2:"#a78bfa", 3:"#34d399", 4:"#fb923c"}

        for idx, (i, result_img) in enumerate(results.items()):
            col = cols[idx % len(cols)]
            with col:
                # Output title
                c = slot_colors.get(i, "#38bdf8")
                st.markdown(
                    f'<div style="font-family:Space Mono,monospace; font-size:0.75rem; '
                    f'color:{c}; border-bottom:1px solid {c}; padding-bottom:3px; '
                    f'margin-bottom:8px; letter-spacing:2px;">OUTPUT {i}</div>',
                    unsafe_allow_html=True
                )

                # Pipeline summary for this output
                s = all_settings[i]
                steps = []
                if s.get("crop_enabled"):       steps.append("Crop")
                if s.get("grey_enabled"):       steps.append("Grey")
                if s.get("blur_enabled"):       steps.append(f"Blur σ{s['blur_sigma']}")
                if s.get("brightness_enabled"): steps.append(f"Bright ×{s['brightness_factor']}")
                if s.get("contrast_enabled"):   steps.append(f"Cont ×{s['contrast_factor']}")
                if s.get("resample_enabled"):   steps.append(f"Resamp {s['resample_scale']}%")
                label = " → ".join(steps) if steps else "No edits"

                st.markdown(
                    f'<div class="info-box" style="font-size:0.78rem !important;">'
                    f'{label}</div>',
                    unsafe_allow_html=True
                )

                # Image
                st.markdown('<div class="img-card">', unsafe_allow_html=True)
                st.image(result_img, width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)
                st.caption(f"📐 {result_img.width} × {result_img.height} px")

                # Individual download
                fname = os.path.splitext(uploaded_file.name)[0]
                st.download_button(
                    label=f"⬇ Download Output {i}",
                    data=image_to_bytes(result_img, export_fmt),
                    file_name=f"{fname}_output{i}.{export_fmt.lower()}",
                    mime=f"image/{export_fmt.lower()}",
                    key=f"dl_{i}"
                )

        # If 4 outputs — render row 2
        if num_outputs == 4:
            st.markdown("<br>", unsafe_allow_html=True)
            cols2 = st.columns(2)
            for idx, i in enumerate([3, 4]):
                if i in results:
                    result_img = results[i]
                    col = cols2[idx]
                    with col:
                        c = slot_colors.get(i, "#38bdf8")
                        st.markdown(
                            f'<div style="font-family:Space Mono,monospace; font-size:0.75rem; '
                            f'color:{c}; border-bottom:1px solid {c}; padding-bottom:3px; '
                            f'margin-bottom:8px; letter-spacing:2px;">OUTPUT {i}</div>',
                            unsafe_allow_html=True
                        )
                        s = all_settings[i]
                        steps = []
                        if s.get("crop_enabled"):       steps.append("Crop")
                        if s.get("grey_enabled"):       steps.append("Grey")
                        if s.get("blur_enabled"):       steps.append(f"Blur σ{s['blur_sigma']}")
                        if s.get("brightness_enabled"): steps.append(f"Bright ×{s['brightness_factor']}")
                        if s.get("contrast_enabled"):   steps.append(f"Cont ×{s['contrast_factor']}")
                        if s.get("resample_enabled"):   steps.append(f"Resamp {s['resample_scale']}%")
                        label = " → ".join(steps) if steps else "No edits"
                        st.markdown(
                            f'<div class="info-box" style="font-size:0.78rem !important;">'
                            f'{label}</div>', unsafe_allow_html=True
                        )
                        st.markdown('<div class="img-card">', unsafe_allow_html=True)
                        st.image(result_img, width='stretch')
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.caption(f"📐 {result_img.width} × {result_img.height} px")
                        fname = os.path.splitext(uploaded_file.name)[0]
                        st.download_button(               
                            label=f"⬇ Download Output {i}",
                            data=image_to_bytes(result_img, export_fmt),
                            file_name=f"{fname}_output{i}.{export_fmt.lower()}",
                            mime=f"image/{export_fmt.lower()}",
                            key=f"dl_{i}"
                        )

        # Histogram of outputs
        if show_hist:
            st.markdown('<div class="section-label sl-meta">Histogram — Outputs</div>', unsafe_allow_html=True)
            for i, result_img in results.items():
                st.caption(f"Output {i}")
                render_histogram(result_img, key_suffix=f"out{i}")

    # ── BIT DEPTH ─────────────────────────────────────────────
    if convert_btn:
        bit_map = {"8-bit":8, "16-bit":16, "32-bit float":32}
        target  = bit_map[bit_target]
        with st.spinner(f"Converting to {bit_target}..."):
            converted = apply_bit_depth(img_original, target)

        st.markdown('<div class="section-label sl-bit">Bit Depth Conversion</div>', unsafe_allow_html=True)
        buf = io.BytesIO() 
        if target == 32:
            pil_img = Image.fromarray((converted*255).astype(np.uint8))
        elif target == 16:
            pil_img = Image.fromarray(converted.astype(np.uint16))
        else:
            pil_img = Image.fromarray(converted.astype(np.uint8))
        pil_img.save(buf, format="TIFF")
        buf.seek(0)

        c1, c2 = st.columns(2)
        with c1: st.metric("Original", "8-bit uint8")
        with c2: st.metric("Converted to", bit_target)

        fname = os.path.splitext(uploaded_file.name)[0]
        st.download_button(
            label=f"⬇ Download {bit_target} .tif",
            data=buf,
            file_name=f"{fname}_{bit_target.replace(' ','_').replace('-','')}.tif",
            mime="image/tiff",
            key="bit_dl"
        )
        st.caption("💡 Open in QGIS, ImageJ, or Python: rasterio.open('file.tif')")