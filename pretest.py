import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "images (1).jpg"


# -------------------------------------------------
# Load Image
# -------------------------------------------------
def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# -------------------------------------------------
# CLAHE Enhancement
# -------------------------------------------------
def apply_clahe(img):

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    l,a,b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    cl = clahe.apply(l)

    merged = cv2.merge((cl,a,b))

    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    return enhanced


# -------------------------------------------------
# Optic Disc Localization (ROI)
# -------------------------------------------------
def detect_roi(img):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray,(9,9),0)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=30,
        maxRadius=120
    )

    if circles is not None:

        circles = np.uint16(np.around(circles))

        x,y,r = circles[0][0]

        roi = img[y-r:y+r, x-r:x+r]

        return roi,(x,y,r)

    return img,(0,0,0)


# -------------------------------------------------
# Disc Segmentation
# -------------------------------------------------
def segment_disc(roi):

    green = roi[:,:,1]

    blur = cv2.GaussianBlur(green,(7,7),0)

    thresh_val = np.mean(blur)

    _,th = cv2.threshold(
        blur,
        thresh_val,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((7,7),np.uint8)

    th = cv2.morphologyEx(th,cv2.MORPH_CLOSE,kernel)

    return th


# -------------------------------------------------
# Cup Segmentation
# -------------------------------------------------
def segment_cup(roi):

    green = roi[:,:,1]

    blur = cv2.GaussianBlur(green,(7,7),0)

    thresh_val = np.mean(blur) + np.std(blur)

    _,th = cv2.threshold(
        blur,
        thresh_val,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((5,5),np.uint8)

    th = cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel)

    return th


# -------------------------------------------------
# Ellipse Fitting
# -------------------------------------------------
def fit_ellipse(mask):

    contours,_ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours)==0:
        return None

    c = max(contours,key=cv2.contourArea)

    if len(c) < 5:
        return None

    ellipse = cv2.fitEllipse(c)

    return ellipse


# -------------------------------------------------
# Cup-to-Disc Ratio
# -------------------------------------------------
def compute_cdr(cup_mask,disc_mask):

    cup_area = np.sum(cup_mask>0)
    disc_area = np.sum(disc_mask>0)

    if disc_area == 0:
        return 0

    return cup_area/disc_area


# -------------------------------------------------
# GradCAM-style heatmap (for visualization only)
# -------------------------------------------------
def generate_heatmap(roi):

    gray = cv2.cvtColor(roi,cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray,40,120)

    heat = cv2.GaussianBlur(edges,(31,31),0)

    heat = heat/np.max(heat)

    return heat


def overlay_heatmap(img,heatmap):

    heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))

    heatmap = np.uint8(255*heatmap)

    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img,0.6,heatmap,0.4,0)

    return overlay


# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------

img = load_image(IMAGE_PATH)

clahe = apply_clahe(img)

roi,(x,y,r) = detect_roi(clahe)

disc_mask = segment_disc(roi)

cup_mask = segment_cup(roi)

disc_ellipse = fit_ellipse(disc_mask)

cup_ellipse = fit_ellipse(cup_mask)

cdr = compute_cdr(cup_mask,disc_mask)

print("Cup to Disc Ratio:", round(cdr,3))

if cdr > 0.6:
    print("⚠ Possible Glaucoma")
else:
    print("✓ Normal")


# Draw ellipses
roi_vis = roi.copy()

if disc_ellipse is not None:
    cv2.ellipse(roi_vis,disc_ellipse,(0,255,0),2)

if cup_ellipse is not None:
    cv2.ellipse(roi_vis,cup_ellipse,(255,0,0),2)


heat = generate_heatmap(roi)

overlay = overlay_heatmap(roi_vis,heat)


# -------------------------------------------------
# Visualization
# -------------------------------------------------

plt.figure(figsize=(12,8))

plt.subplot(231)
plt.title("Original")
plt.imshow(img)
plt.axis("off")

plt.subplot(232)
plt.title("CLAHE")
plt.imshow(clahe)
plt.axis("off")

plt.subplot(233)
plt.title("ROI")
plt.imshow(roi)
plt.axis("off")

plt.subplot(234)
plt.title("Cup Mask")
plt.imshow(cup_mask,cmap="gray")
plt.axis("off")

plt.subplot(235)
plt.title("Disc Mask")
plt.imshow(disc_mask,cmap="gray")
plt.axis("off")

plt.subplot(236)
plt.title("Ellipse + Heatmap")
plt.imshow(overlay)
plt.axis("off")

plt.show()