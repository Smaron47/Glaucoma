import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "images (1).jpg"


# ================================================
# 1. LOAD IMAGE
# ================================================
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ================================================
# 2. CLAHE — applied only on L channel
# ================================================
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ================================================
# 3. DYNAMIC OPTIC DISC DETECTION
#    The optic disc = brightest, most compact blob
#    in the green channel. Uses morphology + blob
#    scoring rather than relying on HoughCircles.
# ================================================
def detect_disc_center(img):
    h, w = img.shape[:2]
    green = img[:, :, 1].astype(np.float32)

    # smooth heavily to suppress vessels/noise
    blur = cv2.GaussianBlur(green, (0, 0), sigmaX=w * 0.02)

    # dynamic threshold: top 5% brightest pixels
    thresh_val = np.percentile(blur, 95)
    _, binary = cv2.threshold(
        blur.astype(np.uint8), int(thresh_val), 255, cv2.THRESH_BINARY
    )

    # morphological clean-up
    k = max(5, w // 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    # score each blob by brightness × compactness
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
        compactness  = area / (bw * bh_ + 1e-6)
        blob_mask    = (labels == i).astype(np.uint8)
        mean_bright  = cv2.mean(green, mask=blob_mask)[0]
        score        = mean_bright * compactness
        if score > best_score:
            best_score = score
            best_idx   = i

    if best_idx == -1:
        _, _, _, max_loc = cv2.minMaxLoc(blur)
        cx, cy = max_loc
        r = min(h, w) // 10
        print("⚠  Fallback: brightest pixel used as disc center")
        return cx, cy, r

    cx, cy = centroids[best_idx]
    bw     = stats[best_idx, cv2.CC_STAT_WIDTH]
    bh_    = stats[best_idx, cv2.CC_STAT_HEIGHT]
    r      = int((bw + bh_) / 4)

    # refine with HoughCircles in a local window
    margin = r * 3
    x1c = max(0, int(cx) - margin)
    y1c = max(0, int(cy) - margin)
    x2c = min(w, int(cx) + margin)
    y2c = min(h, int(cy) + margin)
    local      = img[y1c:y2c, x1c:x2c, 1]
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
        r  = int(lr)

    return int(cx), int(cy), r


# ================================================
# 4. EXTRACT ROI — disc region + expand by px
# ================================================
def extract_roi(img, x, y, r, expand=5):
    r_exp = r + expand
    h, w  = img.shape[:2]
    x1 = max(0, x - r_exp)
    y1 = max(0, y - r_exp)
    x2 = min(w, x + r_exp)
    y2 = min(h, y + r_exp)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid ROI bounds: ({x1},{y1})->({x2},{y2})")
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError("Empty ROI — disc center outside image bounds")
    return roi, (x1, y1, x2, y2)


# ================================================
# 5. PREPROCESS ROI
#    Adaptive per-channel histogram stretch + sharpen
# ================================================
def preprocess_roi(roi):
    out = np.zeros_like(roi)
    for c in range(3):
        ch    = roi[:, :, c].astype(np.float32)
        p_lo  = np.percentile(ch, 2)
        p_hi  = np.percentile(ch, 98)
        if p_hi > p_lo:
            ch = (ch - p_lo) / (p_hi - p_lo) * 255.0
        out[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

    blurred   = cv2.GaussianBlur(out, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(out, 1.5, blurred, -0.5, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ================================================
# 6. CIRCULAR MASK — hard boundary for the ROI
# ================================================
def make_circle_mask(roi, cx_local, cy_local, r):
    h, w = roi.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx_local, cy_local), r, 255, -1)
    return mask


# ================================================
# 7. SEGMENT DISC — Otsu inside circle mask only
# ================================================
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
    mask    = cv2.bitwise_and(mask.astype(np.uint8), circle_mask)

    k      = max(3, roi.shape[0] // 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    return keep_largest_blob(mask)


# ================================================
# 8. SEGMENT CUP — strictly inside disc mask
# ================================================
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


# ================================================
# 9. KEEP LARGEST BLOB
# ================================================
def keep_largest_blob(mask):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean   = np.zeros_like(mask)
    clean[labels == largest] = 255
    return clean


# ================================================
# 10. FIT ELLIPSE
# ================================================
def fit_ellipse(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    return cv2.fitEllipse(c) if len(c) >= 5 else None


# ================================================
# 11. CDR
# ================================================
def compute_cdr(cup_mask, disc_mask):
    disc_area = np.sum(disc_mask > 0)
    return np.sum(cup_mask > 0) / disc_area if disc_area > 0 else 0.0


# ================================================
# 12. HEATMAP — brightness activation inside disc
#     Hot (red/yellow) = brightest  →  cup zone
#     Cool (blue/cyan) = mid-bright →  disc rim
#     Outside disc     = original image pixels
# ================================================
def generate_heatmap_overlay(roi, disc_mask, cup_ellipse, disc_ellipse):
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(3, roi.shape[0] // 15))

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


# ================================================
# MAIN PIPELINE
# ================================================

img   = load_image(IMAGE_PATH)
clahe = apply_clahe(img)

x, y, r = detect_disc_center(clahe)
print(f"Disc → center=({x},{y})  radius={r}px")

roi_raw, (x1, y1, x2, y2) = extract_roi(clahe, x, y, r, expand=5)
roi = preprocess_roi(roi_raw)

# Local disc center within the ROI crop
cx_local = x - x1
cy_local = y - y1

# Hard circular boundary — nothing leaks outside
circle_mask = make_circle_mask(roi, cx_local, cy_local, r + 5)

disc_mask    = segment_disc(roi, circle_mask)
cup_mask     = segment_cup(roi, disc_mask)
disc_ellipse = fit_ellipse(disc_mask)
cup_ellipse  = fit_ellipse(cup_mask)
cdr          = compute_cdr(cup_mask, disc_mask)

print(f"Cup-to-Disc Ratio : {cdr:.3f}")
print("⚠  Possible Glaucoma" if cdr > 0.6 else "✓  Normal")

heatmap_vis = generate_heatmap_overlay(roi, disc_mask, cup_ellipse, disc_ellipse)

# Full-image annotation
full_vis = clahe.copy()
cv2.circle(full_vis, (x, y), r,     (0, 255, 0),   2)
cv2.circle(full_vis, (x, y), r + 5, (255, 165, 0), 1)
cv2.circle(full_vis, (x, y), 4,     (255, 0, 0),  -1)

# ------------------------------------------------
# VISUALIZATION
# ------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.patch.set_facecolor("#1a1a1a")

titles = [
    "Original",
    "CLAHE + Disc Detection",
    "ROI  (preprocessed, +5 px)",
    "Disc Mask  (inside ROI only)",
    "Cup Mask  (inside Disc only)",
    f"Heatmap + Ellipses  |  CDR = {cdr:.3f}",
]
images = [img, full_vis, roi, disc_mask, cup_mask, heatmap_vis]
cmaps  = [None, None, None, "gray", "gray", None]

for ax, title, im, cmap in zip(axes.flat, titles, images, cmaps):
    ax.set_facecolor("#1a1a1a")
    ax.imshow(im, cmap=cmap)
    ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.axis("off")

verdict       = "⚠  POSSIBLE GLAUCOMA" if cdr > 0.6 else "✓  NORMAL"
verdict_color = "#ff4444"              if cdr > 0.6 else "#44ff88"
fig.text(0.5, 0.01, verdict, ha="center", fontsize=13,
         fontweight="bold", color=verdict_color)

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("result.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("✓  Saved → result.png")