"""
╔══════════════════════════════════════════════════════════════╗
║  GlaucomaNet — Streamlit App                                 ║
║  TFLite classification  +  OpenCV ROI / Disc-Cup heatmap    ║
║                                                              ║
║  HOW TO RUN                                                  ║
║    pip install streamlit opencv-python-headless pillow       ║
║               matplotlib numpy tensorflow                    ║
║    streamlit run glaucoma_app.py                             ║
║                                                              ║
║  UPLOAD:                                                     ║
║    1. Your TFLite model  (.tflite)                           ║
║    2. A fundus image     (.jpg / .png)                       ║
╚══════════════════════════════════════════════════════════════╝
"""

import io
import tempfile

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# ── Try TFLite runtime (lightweight) then fall back to full TF ──────────────
try:
    import tflite_runtime.interpreter as tflite
    def _make_interpreter(path):
        return tflite.Interpreter(model_path=path)
except ImportError:
    import tensorflow as tf
    def _make_interpreter(path):
        return tf.lite.Interpreter(model_path=path)


# ════════════════════════════════════════════════════════════════
#  SECTION A — OPTIC DISC / CUP PIPELINE  (from gradcamUpdated.py)
# ════════════════════════════════════════════════════════════════

def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """CLAHE on L-channel only (keeps colour balance)."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def detect_disc_center(img_rgb: np.ndarray):
    """Return (cx, cy, radius) of the optic disc."""
    h, w = img_rgb.shape[:2]
    green = img_rgb[:, :, 1].astype(np.float32)
    blur = cv2.GaussianBlur(green, (0, 0), sigmaX=w * 0.02)

    thresh_val = np.percentile(blur, 95)
    _, binary = cv2.threshold(blur.astype(np.uint8), int(thresh_val), 255,
                               cv2.THRESH_BINARY)

    k = max(5, w // 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    best_idx, best_score = -1, -1
    min_area = (w * h) * 0.002
    max_area = (w * h) * 0.25

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        bw   = stats[i, cv2.CC_STAT_WIDTH]
        bh_  = stats[i, cv2.CC_STAT_HEIGHT]
        if area < min_area or area > max_area:
            continue
        compactness = area / (bw * bh_ + 1e-6)
        blob_mask   = (labels == i).astype(np.uint8)
        mean_bright = cv2.mean(green, mask=blob_mask)[0]
        score       = mean_bright * compactness
        if score > best_score:
            best_score = score
            best_idx   = i

    if best_idx == -1:
        _, _, _, max_loc = cv2.minMaxLoc(blur)
        cx, cy = max_loc
        r = min(h, w) // 10
        return cx, cy, r

    cx, cy = centroids[best_idx]
    bw     = stats[best_idx, cv2.CC_STAT_WIDTH]
    bh_    = stats[best_idx, cv2.CC_STAT_HEIGHT]
    r      = int((bw + bh_) / 4)

    # Refine with Hough in local window
    margin = r * 3
    x1c = max(0, int(cx) - margin); y1c = max(0, int(cy) - margin)
    x2c = min(w, int(cx) + margin); y2c = min(h, int(cy) + margin)
    local      = img_rgb[y1c:y2c, x1c:x2c, 1]
    local_blur = cv2.GaussianBlur(local, (9, 9), 0)

    circles = cv2.HoughCircles(
        local_blur, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=local.shape[0],
        param1=40, param2=20,
        minRadius=max(5, r // 2),
        maxRadius=int(r * 2.5),
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        lx, ly, lr = circles[0][0]
        cx = x1c + int(lx); cy = y1c + int(ly); r = int(lr)

    return int(cx), int(cy), r


def extract_roi(img_rgb, x, y, r, expand=5):
    r_exp = r + expand
    h, w  = img_rgb.shape[:2]
    x1 = max(0, x - r_exp); y1 = max(0, y - r_exp)
    x2 = min(w, x + r_exp); y2 = min(h, y + r_exp)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid ROI: ({x1},{y1})->({x2},{y2})")
    roi = img_rgb[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError("Empty ROI — disc outside image bounds")
    return roi, (x1, y1, x2, y2)


def preprocess_roi(roi: np.ndarray) -> np.ndarray:
    """Per-channel histogram stretch + unsharp-mask sharpen."""
    out = np.zeros_like(roi)
    for c in range(3):
        ch   = roi[:, :, c].astype(np.float32)
        p_lo = np.percentile(ch, 2); p_hi = np.percentile(ch, 98)
        if p_hi > p_lo:
            ch = (ch - p_lo) / (p_hi - p_lo) * 255.0
        out[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
    blurred   = cv2.GaussianBlur(out, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(out, 1.5, blurred, -0.5, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def make_circle_mask(roi, cx_local, cy_local, r):
    h, w = roi.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx_local, cy_local), r, 255, -1)
    return mask


def keep_largest_blob(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean   = np.zeros_like(mask)
    clean[labels == largest] = 255
    return clean


def segment_disc(roi, circle_mask):
    green = roi[:, :, 1].copy()
    green[circle_mask == 0] = 0
    blur  = cv2.GaussianBlur(green, (7, 7), 0)

    masked_pixels = blur[circle_mask > 0]
    otsu_thresh, _ = cv2.threshold(
        masked_pixels.reshape(-1, 1), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    _, mask = cv2.threshold(blur, float(otsu_thresh) * 0.85, 255, cv2.THRESH_BINARY)
    mask    = cv2.bitwise_and(mask.astype(np.uint8), circle_mask)

    k      = max(3, roi.shape[0] // 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    return keep_largest_blob(mask)


def segment_cup(roi, disc_mask):
    green = roi[:, :, 1].copy()
    green[disc_mask == 0] = 0
    blur  = cv2.GaussianBlur(green, (5, 5), 0)

    vals   = blur[disc_mask > 0].astype(np.float32)
    thresh = np.mean(vals) + 0.8 * np.std(vals)

    _, mask = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
    mask    = cv2.bitwise_and(mask.astype(np.uint8), disc_mask)

    k      = max(3, roi.shape[0] // 30)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return keep_largest_blob(mask)


def fit_ellipse(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    return cv2.fitEllipse(c) if len(c) >= 5 else None


def compute_cdr(cup_mask, disc_mask) -> float:
    disc_area = np.sum(disc_mask > 0)
    return float(np.sum(cup_mask > 0) / disc_area) if disc_area > 0 else 0.0


def generate_heatmap_overlay(roi, disc_mask, cup_ellipse, disc_ellipse):
    """Brightness-activation heatmap — works with any model incl. TFLite."""
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0),
                             sigmaX=max(3, roi.shape[0] // 15))

    activation = np.zeros_like(blur)
    m = disc_mask > 0
    if m.sum() > 0:
        vals = blur[m]
        activation[m] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-6)

    heatmap_u8    = np.uint8(255 * activation)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    result  = roi.copy()
    blended = cv2.addWeighted(roi, 0.45, heatmap_color, 0.55, 0)
    result[m] = blended[m]

    if disc_ellipse is not None:
        cv2.ellipse(result, disc_ellipse, (0, 255, 0), 2)
    if cup_ellipse is not None:
        cv2.ellipse(result, cup_ellipse, (255, 255, 255), 2)

    return result


# ════════════════════════════════════════════════════════════════
#  SECTION B — FULL ANALYSIS PIPELINE
# ════════════════════════════════════════════════════════════════

def run_optic_pipeline(img_rgb: np.ndarray):
    """
    Returns dict with all intermediate results and heatmap.
    img_rgb : H×W×3 uint8 RGB image (any size).
    """
    clahe_img = apply_clahe(img_rgb)
    x, y, r   = detect_disc_center(clahe_img)

    roi_raw, bbox = extract_roi(clahe_img, x, y, r, expand=5)
    x1, y1, x2, y2 = bbox
    roi           = preprocess_roi(roi_raw)

    cx_local = x - x1; cy_local = y - y1
    circle_mask  = make_circle_mask(roi, cx_local, cy_local, r + 5)
    disc_mask    = segment_disc(roi, circle_mask)
    cup_mask     = segment_cup(roi, disc_mask)
    disc_ellipse = fit_ellipse(disc_mask)
    cup_ellipse  = fit_ellipse(cup_mask)
    cdr          = compute_cdr(cup_mask, disc_mask)
    heatmap      = generate_heatmap_overlay(roi, disc_mask, cup_ellipse, disc_ellipse)

    # Annotate full image with disc ring
    full_vis = clahe_img.copy()
    cv2.circle(full_vis, (x, y), r,     (0, 255, 0),   2)
    cv2.circle(full_vis, (x, y), r + 5, (255, 165, 0), 1)
    cv2.circle(full_vis, (x, y), 4,     (255, 0, 0),  -1)

    return dict(
        original     = img_rgb,
        clahe_detect = full_vis,
        roi          = roi,
        disc_mask    = disc_mask,
        cup_mask     = cup_mask,
        heatmap      = heatmap,
        cdr          = cdr,
        disc_center  = (x, y, r),
    )


def run_tflite_inference(interpreter, img_rgb: np.ndarray) -> float:
    """
    Returns glaucoma probability (0-1).
    Automatically detects input shape + dtype.
    """
    in_det  = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    _, h, w, _ = in_det["shape"]

    # Resize
    resized = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # dtype
    if in_det["dtype"] == np.uint8:
        inp = resized.astype(np.uint8)[np.newaxis]
    else:
        inp = (resized.astype(np.float32) / 255.0)[np.newaxis]

    interpreter.set_tensor(in_det["index"], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(out_det["index"])

    # handle uint8 output (int8-quantised models)
    if out_det["dtype"] == np.uint8:
        scale, zero = out_det["quantization"]
        out = (out.astype(np.float32) - zero) * scale

    prob = float(np.squeeze(out))
    # sigmoid if logit range detected
    if prob < 0 or prob > 1:
        prob = float(1 / (1 + np.exp(-prob)))
    return prob


# ════════════════════════════════════════════════════════════════
#  SECTION C — MATPLOTLIB FIGURE
# ════════════════════════════════════════════════════════════════

def make_figure(results: dict, prob: float, threshold: float = 0.5) -> io.BytesIO:
    cdr     = results["cdr"]
    is_gla  = prob >= threshold
    cdr_bad = cdr > 0.65

    titles = [
        "① Original Fundus",
        "② CLAHE  +  Disc Ring",
        "③ ROI  (preprocessed)",
        "④ Disc Mask",
        "⑤ Cup Mask",
        f"⑥ Heatmap + Ellipses   |   CDR = {cdr:.3f}",
    ]
    images = [
        results["original"],
        results["clahe_detect"],
        results["roi"],
        results["disc_mask"],
        results["cup_mask"],
        results["heatmap"],
    ]
    cmaps = [None, None, None, "gray", "gray", None]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor("#111827")

    for ax, title, im, cmap in zip(axes.flat, titles, images, cmaps):
        ax.set_facecolor("#111827")
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, color="#e5e7eb", fontsize=9.5, pad=6,
                     fontfamily="monospace")
        ax.axis("off")

    # ── Bottom verdict bar ─────────────────────────────────────
    model_txt   = f"Model: {'⚠ GLAUCOMA' if is_gla else '✓ NORMAL'}  ({prob:.1%})"
    cdr_txt     = f"CDR: {cdr:.3f}  {'— HIGH RISK' if cdr_bad else '— OK'}"
    verdict_clr = "#f87171" if is_gla or cdr_bad else "#4ade80"

    fig.text(0.5, 0.015,
             f"{model_txt}    |    {cdr_txt}",
             ha="center", fontsize=12, fontweight="bold",
             color=verdict_clr, fontfamily="monospace")

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# ════════════════════════════════════════════════════════════════
#  SECTION D — STREAMLIT UI
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="GlaucomaNet",
    page_icon="👁",
    layout="wide",
)

# ── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Sora:wght@300;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: #0f172a;
    color: #e2e8f0;
}
.stApp { background: #0f172a; }

/* Header */
.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 60%, #1a1035 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "👁";
    position: absolute;
    right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.08;
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0 0 0.3rem;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p { color: #94a3b8; margin: 0; font-size: 0.9rem; font-weight: 300; }

/* Cards */
.card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.card h3 {
    margin: 0 0 0.5rem;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
}

/* Metric chips */
.metric-row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-top: 0.5rem; }
.metric-chip {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    flex: 1; min-width: 120px;
    text-align: center;
}
.metric-chip .val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1;
}
.metric-chip .lbl {
    font-size: 0.7rem;
    color: #64748b;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.25rem;
}
.glaucoma { color: #f87171; border-color: #7f1d1d; background: #1c0a0a; }
.normal   { color: #4ade80; border-color: #14532d; background: #0a1c0e; }
.warn     { color: #fbbf24; border-color: #78350f; background: #1c1005; }

/* Upload box */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #334155 !important;
    border-radius: 10px;
    background: #1e293b;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.02em;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* Divider */
hr { border-color: #334155; margin: 1.5rem 0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}
</style>
""", unsafe_allow_html=True)

# ── Hero header ─────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>GlaucomaNet</h1>
  <p>TFLite Classification  ·  Disc &amp; Cup Segmentation  ·  Brightness-Activation Heatmap</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar — settings ───────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ Settings")
    threshold = st.slider(
        "Classification threshold",
        min_value=0.10, max_value=0.90,
        value=0.50, step=0.01,
        help="Probability above this → Glaucoma",
    )
    cdr_thresh = st.slider(
        "CDR risk threshold",
        min_value=0.40, max_value=0.85,
        value=0.65, step=0.01,
        help="Cup-to-disc ratio above this → high risk",
    )
    st.markdown("---")
    st.markdown("""
**How it works**
1. TFLite model → glaucoma probability
2. CLAHE enhancement → optic disc detection
3. Otsu segmentation → disc & cup masks
4. Brightness-activation heatmap (no backprop — TFLite-safe)
5. CDR computed from mask areas
""")

# ── Upload columns ───────────────────────────────────────────────
col_m, col_i = st.columns([1, 1], gap="large")

with col_m:
    st.markdown('<div class="card"><h3>① TFLite Model</h3>', unsafe_allow_html=True)
    model_file = st.file_uploader(
        "Upload .tflite file",
        type=["tflite"],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_i:
    st.markdown('<div class="card"><h3>② Fundus Image</h3>', unsafe_allow_html=True)
    image_file = st.file_uploader(
        "Upload fundus image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ── Run button ───────────────────────────────────────────────────
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    run_btn = st.button("🔬  Analyse Image", use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Analysis ─────────────────────────────────────────────────────
if run_btn:
    if model_file is None:
        st.error("Please upload a TFLite model file.")
        st.stop()
    if image_file is None:
        st.error("Please upload a fundus image.")
        st.stop()

    # Save TFLite to temp file (TFLite needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as tf_tmp:
        tf_tmp.write(model_file.read())
        tflite_path = tf_tmp.name

    # Load image
    pil_img  = Image.open(image_file).convert("RGB")
    img_rgb  = np.array(pil_img)

    with st.spinner("Running TFLite inference…"):
        try:
            interp = _make_interpreter(tflite_path)
            interp.allocate_tensors()
            prob   = run_tflite_inference(interp, img_rgb)
            tflite_ok = True
        except Exception as e:
            st.warning(f"TFLite inference failed: {e}\n\nShowing visual analysis only.")
            prob      = None
            tflite_ok = False

    with st.spinner("Segmenting disc & cup — building heatmap…"):
        try:
            results   = run_optic_pipeline(img_rgb)
            optic_ok  = True
        except Exception as e:
            st.error(f"Optic pipeline error: {e}")
            st.stop()

    cdr = results["cdr"]

    # ── Metric cards ────────────────────────────────────────────
    st.markdown("### Results")
    m1, m2, m3, m4 = st.columns(4)

    if tflite_ok and prob is not None:
        is_gla    = prob >= threshold
        cls_label = "GLAUCOMA" if is_gla else "NORMAL"
        cls_class = "glaucoma" if is_gla else "normal"
        with m1:
            st.markdown(f"""
<div class="metric-chip {cls_class}">
  <div class="val">{'⚠' if is_gla else '✓'}</div>
  <div class="lbl">{cls_label}</div>
</div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
<div class="metric-chip">
  <div class="val" style="font-family:'JetBrains Mono',monospace;color:#60a5fa">
    {prob:.1%}</div>
  <div class="lbl">Probability</div>
</div>""", unsafe_allow_html=True)
    else:
        with m1:
            st.markdown('<div class="metric-chip"><div class="val">—</div><div class="lbl">No model</div></div>',
                        unsafe_allow_html=True)
        prob = 0.0   # sentinel for figure

    cdr_class = "glaucoma" if cdr > cdr_thresh else "normal"
    with m3:
        st.markdown(f"""
<div class="metric-chip {cdr_class}">
  <div class="val" style="font-family:'JetBrains Mono',monospace">{cdr:.3f}</div>
  <div class="lbl">Cup-to-Disc Ratio</div>
</div>""", unsafe_allow_html=True)

    cdr_risk = "HIGH RISK" if cdr > cdr_thresh else "NORMAL"
    with m4:
        st.markdown(f"""
<div class="metric-chip {cdr_class}">
  <div class="val">{'⚠' if cdr > cdr_thresh else '✓'}</div>
  <div class="lbl">CDR — {cdr_risk}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 6-panel figure ──────────────────────────────────────────
    fig_buf = make_figure(results, prob, threshold)
    st.image(fig_buf, use_column_width=True, caption="Full diagnostic panel")

    # ── Download button ─────────────────────────────────────────
    st.download_button(
        label="⬇  Download diagnostic image",
        data=fig_buf,
        file_name="glaucoma_analysis.png",
        mime="image/png",
    )

    # ── Interpretation guide ─────────────────────────────────────
    with st.expander("📖  How to interpret these results"):
        st.markdown("""
**Model probability** — output of your TFLite ResNet50 (sigmoid).  
Values above the threshold (default 0.5, adjustable in sidebar) → Glaucoma.

**Cup-to-Disc Ratio (CDR)**  
- < 0.5 → generally normal  
- 0.5 – 0.65 → borderline / monitor  
- > 0.65 → high risk indicator (consult ophthalmologist)

**Heatmap colour scale**  
🔴 Red / Yellow → highest brightness (cup zone, most activated)  
🔵 Blue / Cyan  → mid-brightness (disc rim)  
Disc boundary shown in **green**, cup ellipse in **white**.

**Important**: This tool is for research / screening assistance only.  
Always consult a qualified ophthalmologist for clinical decisions.
""")

elif not run_btn:
    # ── Placeholder ─────────────────────────────────────────────
    st.markdown("""
<div style="text-align:center; padding:3rem; color:#334155;">
  <div style="font-size:4rem; margin-bottom:1rem;">👁</div>
  <div style="font-size:1rem; font-family:'JetBrains Mono',monospace;">
    Upload your TFLite model &amp; a fundus image, then click <strong style="color:#60a5fa">Analyse</strong>
  </div>
</div>
""", unsafe_allow_html=True)