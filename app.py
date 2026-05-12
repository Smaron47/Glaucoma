import streamlit as st
import cv2
import numpy as np
import json, os, math, warnings, logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input as _resnet_pre

# ── Suppress harmless Streamlit / TF warnings ─────────────────────
logging.getLogger("streamlit.runtime.scriptrunner.script_run_context").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════
CFG = {
    "raw_size": 512,          # image resized for processing
    "img_size": 224,          # input size for the TFLite model
    "cdr_threshold": 0.55,    # CDR glaucoma threshold
}

# ══════════════════════════════════════════════════════════════════
# IMAGE PROCESSING (from gradcamUpdated.py)
# ══════════════════════════════════════════════════════════════════

def apply_clahe(img):
    """CLAHE on L channel only – returns RGB image."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def detect_disc_center(img):
    """
    Robust optic disc detection using brightness + compactness scoring.
    Returns (cx, cy, radius).
    """
    h, w = img.shape[:2]
    green = img[:, :, 1].astype(np.float32)

    blur = cv2.GaussianBlur(green, (0, 0), sigmaX=w * 0.02)
    thresh_val = np.percentile(blur, 95)
    _, binary = cv2.threshold(blur.astype(np.uint8), int(thresh_val), 255, cv2.THRESH_BINARY)

    k = max(5, w // 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    best_idx, best_score = -1, -1
    min_area = (w * h) * 0.002
    max_area = (w * h) * 0.25

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if area < min_area or area > max_area:
            continue
        compactness = area / (bw * bh + 1e-6)
        blob_mask = (labels == i).astype(np.uint8)
        mean_bright = cv2.mean(green, mask=blob_mask)[0]
        score = mean_bright * compactness
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx == -1:
        _, _, _, max_loc = cv2.minMaxLoc(blur)
        cx, cy = max_loc
        r = min(h, w) // 10
        return int(cx), int(cy), r

    cx, cy = centroids[best_idx]
    bw = stats[best_idx, cv2.CC_STAT_WIDTH]
    bh = stats[best_idx, cv2.CC_STAT_HEIGHT]
    r = int((bw + bh) / 4)

    # refine with HoughCircles in a local window
    margin = r * 3
    x1c = max(0, int(cx) - margin)
    y1c = max(0, int(cy) - margin)
    x2c = min(w, int(cx) + margin)
    y2c = min(h, int(cy) + margin)
    local = img[y1c:y2c, x1c:x2c, 1]
    local_blur = cv2.GaussianBlur(local, (9, 9), 0)
    circles = cv2.HoughCircles(
        local_blur, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=local.shape[0],
        param1=40, param2=20,
        minRadius=max(5, r // 2),
        maxRadius=int(r * 2.5)
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        lx, ly, lr = circles[0][0]
        cx = x1c + int(lx)
        cy = y1c + int(ly)
        r = int(lr)

    return int(cx), int(cy), r

def extract_roi(img, x, y, r, expand=5):
    """Crop a square region around disc center."""
    r_exp = r + expand
    h, w = img.shape[:2]
    x1 = max(0, x - r_exp)
    y1 = max(0, y - r_exp)
    x2 = min(w, x + r_exp)
    y2 = min(h, y + r_exp)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid ROI bounds: ({x1},{y1})->({x2},{y2})")
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError("Empty ROI")
    return roi, (x1, y1, x2, y2)

def preprocess_roi(roi):
    """Adaptive per-channel histogram stretch + sharpen."""
    out = np.zeros_like(roi)
    for c in range(3):
        ch = roi[:, :, c].astype(np.float32)
        p_lo = np.percentile(ch, 2)
        p_hi = np.percentile(ch, 98)
        if p_hi > p_lo:
            ch = (ch - p_lo) / (p_hi - p_lo) * 255.0
        out[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(out, 1.5, blurred, -0.5, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def make_circle_mask(roi, cx_local, cy_local, r):
    h, w = roi.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx_local, cy_local), r, 255, -1)
    return mask

def keep_largest_blob(mask):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean = np.zeros_like(mask)
    clean[labels == largest] = 255
    return clean

def segment_disc(roi, circle_mask):
    green = roi[:, :, 1].copy()
    green[circle_mask == 0] = 0
    blur = cv2.GaussianBlur(green, (7, 7), 0)

    masked_pixels = blur[circle_mask > 0]
    otsu_thresh, _ = cv2.threshold(
        masked_pixels.reshape(-1, 1), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    _, mask = cv2.threshold(blur, float(otsu_thresh) * 0.85, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask.astype(np.uint8), circle_mask)

    k = max(3, roi.shape[0] // 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return keep_largest_blob(mask)

def segment_cup(roi, disc_mask):
    green = roi[:, :, 1].copy()
    green[disc_mask == 0] = 0
    blur = cv2.GaussianBlur(green, (5, 5), 0)

    vals = blur[disc_mask > 0].astype(np.float32)
    thresh = np.mean(vals) + 0.8 * np.std(vals)

    _, mask = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask.astype(np.uint8), disc_mask)

    k = max(3, roi.shape[0] // 30)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return keep_largest_blob(mask)

def fit_ellipse(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    return cv2.fitEllipse(c) if len(c) >= 5 else None

def compute_cdr(cup_mask, disc_mask):
    disc_area = np.sum(disc_mask > 0)
    return np.sum(cup_mask > 0) / disc_area if disc_area > 0 else 0.0

def generate_heatmap_overlay(roi, disc_mask, cup_ellipse, disc_ellipse):
    """Brightness activation heatmap inside the disc region only."""
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(3, roi.shape[0] // 15))

    activation = np.zeros_like(blur)
    m = disc_mask > 0
    if m.sum() > 0:
        vals = blur[m]
        activation[m] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-6)

    heatmap_u8 = np.uint8(255 * activation)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    result = roi.copy()
    blended = cv2.addWeighted(roi, 0.45, heatmap_color, 0.55, 0)
    result[m] = blended[m]

    if disc_ellipse is not None:
        cv2.ellipse(result, disc_ellipse, (0, 255, 0), 2)
    if cup_ellipse is not None:
        cv2.ellipse(result, cup_ellipse, (255, 255, 255), 2)

    return result

# ══════════════════════════════════════════════════════════════════
# CBAM layer (needed only if you ever load a Keras model – kept for compatibility)
# ══════════════════════════════════════════════════════════════════
@tf.keras.utils.register_keras_serializable(package="GlaucomaNet")
class CBAM(layers.Layer):
    def __init__(self, ratio=8, ks=7, **kw):
        super().__init__(**kw)
        self.ratio, self.ks = ratio, ks
    def build(self, input_shape):
        c = int(input_shape[-1])
        self.fc1 = layers.Dense(max(1, c // self.ratio), activation="relu")
        self.fc2 = layers.Dense(c)
        self.sa = layers.Conv2D(1, self.ks, padding="same", activation="sigmoid", dtype="float32")
        super().build(input_shape)
    def call(self, x, training=False):
        x32 = tf.cast(x, tf.float32)
        avg_c = tf.reduce_mean(x32, axis=[1, 2])
        max_c = tf.reduce_max(x32, axis=[1, 2])
        ca = tf.nn.sigmoid(self.fc2(self.fc1(avg_c)) + self.fc2(self.fc1(max_c)))
        ca = tf.reshape(ca, (-1, 1, 1, tf.shape(ca)[-1]))
        x32 = x32 * ca
        avg_s = tf.reduce_mean(x32, axis=-1, keepdims=True)
        max_s = tf.reduce_max(x32, axis=-1, keepdims=True)
        sa = tf.cast(self.sa(tf.concat([avg_s, max_s], axis=-1)), tf.float32)
        x32 = x32 * sa
        return tf.cast(x32, x.dtype)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"ratio": self.ratio, "ks": self.ks})
        return cfg

# ══════════════════════════════════════════════════════════════════
# TFLite inference helper
# ══════════════════════════════════════════════════════════════════
def tflite_predict(interpreter, img_preprocessed):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_details['index'], img_preprocessed[np.newaxis, ...].astype(np.float32))
    interpreter.invoke()
    prob = float(interpreter.get_tensor(output_details['index'])[0, 0])
    return prob

# ══════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Glaucoma Detection – CDR & TFLite", layout="wide")
st.title("🧠 Glaucoma Detection (CDR + TFLite)")

st.markdown("""
Upload a fundus image and the TFLite model with its threshold file.  
The app shows a **brightness‑based heatmap** of the optic disc,  
the **Cup‑to‑Disc Ratio (CDR)**, and the deep‑learning probability.
""")

col1, col2 = st.columns(2)
with col1:
    tflite_file = "model.tflite"
with col2:
    threshold_file = "config.json"

img_file = st.file_uploader("Fundus image", type=["jpg", "jpeg", "png"])

if tflite_file and threshold_file and img_file:
    # Save uploaded files
    with open("temp_model.tflite", "wb") as f: f.write(tflite_file.read())
    with open("temp_threshold.json", "wb") as f: f.write(threshold_file.read())
    with open("temp_img.jpg", "wb") as f: f.write(img_file.read())

    # Load threshold
    with open("temp_threshold.json") as f:
        th_data = json.load(f)
    tflite_threshold = th_data.get("threshold", 0.5)

    # Load TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path="temp_model.tflite")
    interpreter.allocate_tensors()

    # ── Load and preprocess the image ────────────────────────────
    bgr = cv2.imread("temp_img.jpg")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_512 = cv2.resize(rgb, (CFG["raw_size"], CFG["raw_size"]))

    # 1. Apply CLAHE for better disc detection
    enhanced = apply_clahe(rgb_512)

    # 2. Detect disc center and radius (new robust method)
    dx, dy, dr = detect_disc_center(enhanced)

    # 3. Extract ROI around disc (with expansion)
    roi_raw, (x1, y1, x2, y2) = extract_roi(rgb_512, dx, dy, dr, expand=5)

    # 4. Preprocess ROI (sharpen, stretch)
    roi = preprocess_roi(roi_raw)

    # 5. Local disc center inside ROI
    cx_local = dx - x1
    cy_local = dy - y1

    # 6. Hard circular boundary
    circle_mask = make_circle_mask(roi, cx_local, cy_local, dr + 5)

    # 7. Segment disc & cup
    disc_mask = segment_disc(roi, circle_mask)
    cup_mask = segment_cup(roi, disc_mask)
    disc_ellipse = fit_ellipse(disc_mask)
    cup_ellipse = fit_ellipse(cup_mask)
    cdr = compute_cdr(cup_mask, disc_mask)

    # 8. Diagnose via CDR (threshold = 0.55)
    cdr_label = "⚠️ Possible Glaucoma" if cdr > CFG["cdr_threshold"] else "✓ Normal"
    cdr_color = "#ff4444" if cdr > CFG["cdr_threshold"] else "#44ff88"

    # 9. Generate heatmap overlay
    heatmap_vis = generate_heatmap_overlay(roi, disc_mask, cup_ellipse, disc_ellipse)

    # ── TFLite prediction ────────────────────────────────────────
    # Prepare model input: square crop around disc (like before)
    side = min(roi_raw.shape[0], roi_raw.shape[1])
    crop_square = roi_raw[(roi_raw.shape[0]-side)//2:(roi_raw.shape[0]+side)//2,
                          (roi_raw.shape[1]-side)//2:(roi_raw.shape[1]+side)//2]
    model_input = cv2.resize(crop_square, (CFG["img_size"], CFG["img_size"]))
    model_input = _resnet_pre(model_input.astype(np.float32))
    tflite_prob = tflite_predict(interpreter, model_input)
    tflite_label = "Glaucoma" if tflite_prob >= tflite_threshold else "Normal"
    confidence = f"{tflite_prob:.1%}"

    # ── Visualizations ───────────────────────────────────────────
    # Original fundus with disc circle
    fundus_disp = rgb_512.copy()
    cv2.circle(fundus_disp, (dx, dy), dr, (0, 255, 0), 2)          # disc boundary
    cv2.circle(fundus_disp, (dx, dy), dr + 5, (255, 165, 0), 1)    # expanded
    cv2.circle(fundus_disp, (dx, dy), 4, (255, 0, 0), -1)          # center

    # ── Display in three columns ──────────────────────────────────
    colA, colB, colC = st.columns(3)
    with colA:
        st.subheader("Fundus (Disc Detection)")
        st.image(fundus_disp, use_container_width=True, caption="Green: disc boundary")
    with colB:
        st.subheader("ROI Crop + Heatmap")
        st.image(heatmap_vis, use_container_width=True,
                 caption=f"CDR = {cdr:.3f}  ({cdr_label})")
    with colC:
        st.subheader("Disc / Cup Masks")
        # Combine disc (green) and cup (red) into a single image
        mask_vis = np.zeros((*roi.shape[:2], 3), dtype=np.uint8)
        mask_vis[disc_mask > 0] = (0, 255, 0)      # green disc
        mask_vis[cup_mask > 0]  = (255, 0, 0)      # red cup
        mask_vis = cv2.addWeighted(roi, 0.5, mask_vis, 0.5, 0)
        st.image(mask_vis, use_container_width=True, caption="Green: disc, Red: cup")

    # ── Results ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Diagnostic Summary")
    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.metric("CDR (threshold=0.55)", f"{cdr:.3f}", delta=cdr_label)
    with col_res2:
        st.metric("TFLite Prediction", tflite_label)
    with col_res3:
        st.metric("TFLite Probability", confidence,
                  delta=f"Threshold = {tflite_threshold:.4f}")

    # Cleanup temp files
    for f in ["temp_model.tflite", "temp_threshold.json", "temp_img.jpg"]:
        if os.path.exists(f): os.remove(f)

else:
    st.info("Upload the TFLite model, threshold.json, and a fundus image to start.")