# 👁️ GlaucomaAI — Explainable Glaucoma Screening from Fundus Images

<p align="center">
  <strong>AI-assisted optic disc & optic cup analysis for glaucoma screening</strong>
</p>

<p align="center">
  <em>Fundus image → optic disc/cup segmentation → morphometric features → TFLite inference → glaucoma probability + visual explanation</em>
</p>

---

## 📌 Overview

**GlaucomaAI** is an experimental computer-vision system for glaucoma screening from retinal fundus photographs.

Instead of feeding the complete fundus image directly into a classifier, the application first performs an image-processing pipeline designed to identify the **optic disc** and **optic cup**. It then derives morphometric measurements such as the vertical and horizontal cup-to-disc ratios and uses these features together with a cropped optic-disc image for model inference.

The repository contains the Flask inference application, TensorFlow Lite model artifacts, feature scaler, model-development/training code, Grad-CAM/visualization utilities, evaluation figures, and example images.

The current Flask application exposes a browser interface and a REST-style `/analyze` endpoint. It can operate in two modes:

1. **Full model mode** — TFLite model + feature scaler + configurable decision threshold.
2. **Segmentation-only fallback mode** — optic disc/cup segmentation plus a CDR-based heuristic when the model artifacts are unavailable.

The repository currently contains the following main model artifacts:

```text
model/
├── glaucoma_fp16 (1).tflite
└── feat_scaler (1).pkl
```

The Flask application loads these artifacts by default from the `model/` directory. 

---

## ✨ Key Features

- 👁️ Fundus-image glaucoma screening
- 🔬 Optic disc detection
- 🟡 Optic cup segmentation
- 📐 Cup-to-disc ratio analysis
- 🧮 Eight morphometric features
- 🤖 TensorFlow Lite inference
- ⚡ Lightweight inference-oriented model format
- 🎯 Configurable glaucoma decision threshold
- 🔥 Heatmap/visual overlay
- 🩺 Cup & disc mask visualization
- ✂️ Model-input optic-disc crop visualization
- 📊 Morphometric feature display
- 🌐 Flask web application
- 🔌 JSON API through `/analyze`
- 🧠 Feature scaling with a saved `joblib` scaler
- 📦 Fallback CDR-based heuristic when model files are unavailable

---

# 🏗️ System Architecture

```text
                 ┌────────────────────────┐
                 │     Fundus Image       │
                 │       JPG / PNG        │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Image Preprocessing  │
                 │      OpenCV / NumPy    │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Field-of-View (FOV)  │
                 │        Detection       │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Optic Disc Detection │
                 │  threshold + morphology│
                 │   + contour analysis   │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │    Optic Cup Detection │
                 │  within detected disc  │
                 └────────────┬───────────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
      ┌────────────────────┐    ┌────────────────────┐
      │ Morphometric       │    │ Optic Disc ROI     │
      │ Features           │    │ Crop               │
      └─────────┬──────────┘    └──────────┬─────────┘
                │                          │
                │                          ▼
                │                ┌───────────────────┐
                │                │ EfficientNet-style│
                │                │ image preprocessing│
                │                └─────────┬─────────┘
                │                          │
                ▼                          ▼
      ┌──────────────────────────────────────────────┐
      │             TFLite Detector                  │
      │                                              │
      │  Image input + scaled morphometric features  │
      └──────────────────────┬───────────────────────┘
                             │
                             ▼
                  ┌────────────────────────┐
                  │ Glaucoma Probability  │
                  │       + Prediction     │
                  └────────────┬───────────┘
                               │
                ┌──────────────┼───────────────┐
                ▼              ▼               ▼
          ┌──────────┐   ┌──────────┐   ┌────────────┐
          │ Overlay  │   │  Masks   │   │ Disc Crop  │
          │ Heatmap  │   │ Disc/Cup │   │ Model Input│
          └──────────┘   └──────────┘   └────────────┘
```

---

# 🔬 How the Pipeline Works

## 1. Fundus image upload

The user uploads a retinal fundus image through the browser interface.

The frontend accepts common image formats including:

```text
JPG
PNG
```

The image is sent to:

```text
POST /analyze
```

The frontend creates a `FormData` object and places the selected image under the `image` field before making the request. 

---

## 2. Field-of-view detection

The application first creates a field-of-view mask from the grayscale image.

This helps restrict subsequent optic-disc detection to the relevant retinal region.

The implementation uses OpenCV thresholding and morphological operations.

---

## 3. Optic disc detection

The segmentation pipeline searches for a bright, disc-like region using image-processing operations including:

- Gaussian smoothing
- Thresholding
- Morphological opening
- Morphological closing
- Contour extraction
- Convex hull analysis
- Circularity/geometry checks

The detector uses a primary search followed by a relaxed search and finally a fallback method if necessary. 

---

## 4. Optic cup detection

After identifying the optic disc, the application searches inside the disc region for the optic cup.

This produces two binary masks:

```text
Disc mask
Cup mask
```

These masks are subsequently used both for visualization and for calculating morphometric features.

---

# 📐 Morphometric Features

The application extracts eight features from the optic disc and cup masks:

| Feature | Description |
|---|---|
| `CDR-V` | Vertical cup-to-disc ratio |
| `CDR-H` | Horizontal cup-to-disc ratio |
| `Disc%` | Optic disc area relative to image area |
| `Cup%` | Optic cup area relative to image area |
| `Rim%` | Neuro-retinal rim area relative to image area |
| `C/D-Area` | Cup area / disc area |
| `D-Circ` | Disc contour circularity |
| `Rim/D` | Rim area / disc area |

These are calculated directly from the detected contours and masks in the Flask pipeline. 

### Cup-to-disc ratio

The vertical and horizontal CDR values are calculated from the detected cup and disc bounding-box dimensions:

```text
CDR-V = cup height / disc height

CDR-H = cup width / disc width
```

The application UI describes a CDR value above approximately `0.65` as suspicious. This value is also used by the fallback heuristic described below. 

> **Important:** CDR is an established clinical measurement, but a single threshold should not be interpreted as a standalone medical diagnosis. Clinical assessment depends on the complete examination and patient context.

---

# 🤖 Machine-Learning Model

The deployed inference path uses **TensorFlow Lite**.

The detector initializes a TFLite interpreter and reads its input/output tensor definitions. It also loads a serialized feature scaler using `joblib`. 

The model receives two types of information:

### 1. Optic-disc image

The detected optic-disc region is cropped and resized to:

```text
224 × 224 × 3
```

The image is passed through TensorFlow/Keras EfficientNet preprocessing before being supplied to the TFLite model. 

### 2. Morphometric feature vector

The eight extracted features are transformed using the saved scaler:

```text
[CDR-V,
 CDR-H,
 Disc%,
 Cup%,
 Rim%,
 C/D-Area,
 D-Circ,
 Rim/D]
```

The scaled feature vector is then supplied to the second model input. 

---

# 🎯 Prediction

The model produces a glaucoma probability.

The application compares that probability against a configurable threshold:

```text
probability >= threshold
        │
        ├── True  → Glaucoma
        │
        └── False → Normal
```

The default command-line threshold is:

```text
0.40
```

The threshold can also be loaded from a `.npy` file. 

---

# 🧪 Fallback Mode

If the TFLite model or feature scaler cannot be found, the application does not necessarily stop.

Instead, it can operate in **segmentation-only / CDR heuristic mode**.

The fallback computes the morphometric features and derives a simple probability from the vertical CDR:

```text
probability =
clip((CDR-V - 0.30) / 0.50, 0, 1)
```

and classifies:

```text
CDR-V >= 0.65 → Glaucoma
CDR-V <  0.65 → Normal
```

This is explicitly a heuristic fallback, **not the trained machine-learning model**. 

---

# 🖥️ Web Application

The Flask interface is integrated directly into `app.py`.

The UI provides:

### 📤 Image upload

Users can:

- Click to select an image
- Drag and drop an image
- Preview the selected image

### 🔬 Analysis

Click:

```text
Analyse Image
```

to send the image to the backend.

### 📊 Results

The application displays:

- Prediction
- Glaucoma probability
- Visual overlay
- Cup/disc masks
- Model-input crop
- Original image
- Morphometric features

The current interface exposes tabs for **Overlay + Heatmap**, **Cup & Disc Masks**, **Disc Crop**, and **Original**. 

---

# 🌡️ Visualization

The application returns three generated visual assets in addition to the prediction:

```text
img_overlay
img_mask
img_crop
```

They are encoded as Base64 PNG strings in the JSON response. 

### Overlay

Shows the original fundus image with segmentation/activation information.

### Mask

Shows the detected optic disc and optic cup masks.

### Crop

Shows the optic-disc crop actually prepared for the image-model input.

These visualizations are intended to make the model pipeline easier to inspect and debug.

---

# 🔌 API Documentation

## `GET /`

Returns the main GlaucomaAI web interface.

---

## `POST /analyze`

Analyzes a fundus image.

### Request

Send a `multipart/form-data` request:

```text
image=<fundus-image>
```

### cURL example

```bash
curl -X POST \
  -F "image=@fundus.jpg" \
  http://127.0.0.1:5000/analyze
```

### Successful response

```json
{
  "prediction": "Glaucoma",
  "probability": 0.8234,
  "features": {
    "CDR-V": 0.7012,
    "CDR-H": 0.6123,
    "Disc%": 0.0421,
    "Cup%": 0.0198,
    "Rim%": 0.0223,
    "C/D-Area": 0.4702,
    "D-Circ": 0.8142,
    "Rim/D": 0.5298
  },
  "img_overlay": "...",
  "img_mask": "...",
  "img_crop": "..."
}
```

The exact feature values depend on the uploaded image.

### Error response

If no image is uploaded:

```json
{
  "error": "No image uploaded."
}
```

If the image cannot be decoded:

```json
{
  "error": "Could not decode image. Upload a valid JPG/PNG."
}
```

The endpoint returns HTTP `400` for these input errors and `500` if an unexpected processing exception occurs. 

---

# 📂 Repository Structure

The current repository contains:

```text
Glaucoma/
│
├── model/
│   ├── feat_scaler (1).pkl
│   └── glaucoma_fp16 (1).tflite
│
├── app.py
├── amdmodel.py
├── gradcamUpdated.py
├── pretest.py
├── st.py
├── config.json
│
├── dataset-card.jpg
├── original.jpg
├── result.png
├── result1.png
│
├── Figure_1.png
├── Figure_2.png
├── Figure_3.png
├── Figure_4.png
│
├── amdconfusion.png
├── amdroc.png
├── amdtflite.png
├── amdtfroc.png
│
├── requirements.txt
└── README.md
```

The repository currently includes both the deployment/inference path and additional model-development/evaluation artifacts. 

---

# 📜 Important Files

| File | Purpose |
|---|---|
| `app.py` | Main Flask application and inference pipeline |
| `model/glaucoma_fp16 (1).tflite` | TensorFlow Lite glaucoma model |
| `model/feat_scaler (1).pkl` | Feature scaler used by the model |
| `amdmodel.py` | Model development/training and TFLite conversion code |
| `gradcamUpdated.py` | Grad-CAM-related experimentation/visualization |
| `pretest.py` | Pre-testing/experimental processing |
| `st.py` | Additional application/experimental interface code |
| `config.json` | Configuration artifact |
| `requirements.txt` | Python dependencies |
| `Figure_*.png` | Experiment/result visualizations |
| `amdconfusion.png` | Confusion-matrix visualization |
| `amdroc.png` | ROC visualization |
| `amdtflite.png` | TFLite-related result |
| `amdtfroc.png` | TFLite ROC-related result |

---

# 📦 Requirements

The repository currently specifies:

```text
flask>=2.0
gunicorn>=20
opencv-python-headless>=4.7
numpy>=1.24
tensorflow-cpu>=2.12
joblib>=1.0
```

These dependencies are defined in the project's `requirements.txt`. 

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/Smaron47/Glaucoma.git
cd Glaucoma
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

The application can be started directly with:

```bash
python app.py
```

The default configuration uses:

```text
Host: 0.0.0.0
Port: 5000
```

Open:

```text
http://127.0.0.1:5000
```

The application also supports a custom port:

```bash
python app.py --port 8080
```

Debug mode can be enabled with:

```bash
python app.py --debug
```

The command-line interface and defaults are defined in `app.py`. 

---

# ⚙️ Command-Line Options

```text
--model
```

Path to the TensorFlow Lite model.

Default:

```text
model\glaucoma_fp16 (1).tflite
```

---

```text
--scaler
```

Path to the saved feature scaler.

Default:

```text
model\feat_scaler (1).pkl
```

---

```text
--thresh
```

Either:

- a numeric threshold, or
- a `.npy` file containing the threshold.

Default:

```text
0.40
```

---

```text
--host
```

Server bind address.

Default:

```text
0.0.0.0
```

---

```text
--port
```

Server port.

Default:

```text
5000
```

---

```text
--debug
```

Enables Flask debug mode.

---

# 🧰 Example Commands

### Normal

```bash
python app.py
```

### Custom threshold

```bash
python app.py --thresh 0.50
```

### Custom model

```bash
python app.py \
  --model model/glaucoma_fp16.tflite \
  --scaler model/feat_scaler.pkl \
  --thresh 0.45
```

### Custom port

```bash
python app.py --port 8080
```

---

# 🧠 Training / Model Development

The repository also contains a model-development script, `amdmodel.py`.

The training pipeline documented in that script uses a two-phase optimization strategy:

### Phase 1

```text
AdamW
learning rate = 1e-4
weight decay = 1e-4
```

### Phase 2

```text
SGD + Nesterov
learning rate = 1e-3
momentum = 0.9
Nesterov = True
```

The training code monitors:

- Accuracy
- AUC
- Precision
- Recall

and uses class weights calculated from the training dataset. 

The training script also includes early stopping, learning-rate reduction, and model checkpoints. 

---

# 📦 TFLite Conversion

The model-development code includes TensorFlow Lite conversion.

The conversion pipeline attempts:

1. Full INT8 quantization
2. Dynamic-range quantization fallback if the first conversion fails

A representative dataset is used for the quantization process. The script also configures TensorFlow Lite supported operations for compatibility with the model. 

The repository's deployed artifact is:

```text
model/glaucoma_fp16 (1).tflite
```

Therefore, the exact quantization/precision characteristics of the committed deployed artifact should be treated according to the actual TFLite model metadata rather than inferred solely from the training script.

---

# 📊 Evaluation Artifacts

The repository contains several experiment/evaluation figures, including:

```text
amdconfusion.png
amdroc.png
amdtflite.png
amdtfroc.png
Figure_1.png
Figure_2.png
Figure_3.png
Figure_4.png
```

These files are useful for documenting model development and evaluation.

However, this README intentionally does **not** claim a specific final accuracy, AUC, sensitivity, or specificity unless the corresponding verified numeric result is explicitly available from the experiment outputs.

For a research publication, report metrics from a clearly defined held-out test set and document the exact dataset split and evaluation protocol.

---

# 🔍 Explainability

GlaucomaAI emphasizes visual interpretability.

The system exposes:

```text
Original Fundus Image
        ↓
Optic Disc Mask
        ↓
Optic Cup Mask
        ↓
Optic Disc ROI
        ↓
Heatmap / Overlay
```

This allows users and developers to inspect what the pipeline detected before relying on the final classification.

The repository also contains `gradcamUpdated.py` for additional Grad-CAM-related experimentation.

> Visual heatmaps and segmentation masks are explanatory tools. They should not be interpreted as proof that the model has identified the pathological cause of glaucoma.

---

# 🩺 Clinical Interpretation

The system is designed as an **AI-assisted screening/research prototype**, not as an autonomous clinical diagnostic system.

A model prediction should not replace:

- Ophthalmologist assessment
- Intraocular pressure measurement
- Visual-field testing
- Optical coherence tomography
- Gonioscopy
- Clinical history
- Other appropriate ophthalmic examinations

The output should therefore be interpreted as a computational screening signal rather than a definitive diagnosis.

---

# ⚠️ Limitations

Important limitations include:

### Image quality

Poor-quality fundus images can make optic-disc and optic-cup segmentation unreliable.

### Segmentation dependency

The classifier depends on successful localization of the optic disc and cup.

If segmentation is wrong, the extracted features and model crop can also be wrong.

### Dataset generalization

Performance may vary across:

- Camera systems
- Hospitals
- Geographic populations
- Image resolutions
- Illumination
- Field-of-view differences
- Patient demographics
- Disease severity

### Threshold sensitivity

Changing the probability threshold changes the balance between false positives and false negatives.

The default threshold should therefore not be treated as a universal clinical threshold.

### Heuristic fallback

The fallback mode is not equivalent to the trained model and should be clearly distinguished from full-model inference.

---

# 🔐 Security & Production Deployment

Before exposing the service publicly:

- Disable Flask debug mode.
- Validate uploaded file types.
- Limit maximum upload size.
- Handle malformed images safely.
- Add authentication if the API is not public.
- Add rate limiting.
- Run behind HTTPS.
- Use a production WSGI server such as Gunicorn.
- Keep sensitive patient data out of logs.
- Do not expose unnecessary filesystem information.
- Review model/data licensing before commercial deployment.

The repository already includes `gunicorn` in its dependency specification, indicating support for production WSGI deployment. 

---

# 🚀 Future Development

Potential improvements include:

- [ ] Patient-level dataset splitting
- [ ] External validation dataset
- [ ] Better optic-disc/cup segmentation
- [ ] Automated image-quality assessment
- [ ] Confidence calibration
- [ ] Sensitivity/specificity optimization
- [ ] ROC/PR analysis in the application
- [ ] Batch inference
- [ ] Mobile deployment with TFLite
- [ ] Android integration
- [ ] Real-time camera capture
- [ ] Model versioning
- [ ] API authentication
- [ ] Docker deployment
- [ ] Clinical workflow integration
- [ ] More robust cross-dataset evaluation
- [ ] Prospective validation
- [ ] Explainability benchmarking

---

# 🧪 Research Recommendations

For research-grade evaluation, the following should be explicitly reported:

```text
Dataset
   ↓
Patient-level split
   ↓
Training
   ↓
Validation
   ↓
Threshold selection
   ↓
Locked test set
   ↓
External validation
```

Recommended metrics include:

- Accuracy
- ROC-AUC
- PR-AUC
- Sensitivity
- Specificity
- Precision
- F1-score
- Confusion matrix
- Calibration
- 95% confidence intervals

For medical-AI research, patient-level separation is particularly important when multiple images from the same patient are available.

---

# 📚 Reproducibility

To reproduce the deployed inference environment:

1. Clone the repository.
2. Install `requirements.txt`.
3. Keep the TFLite model and feature scaler in the `model/` directory.
4. Start `app.py`.
5. Upload a fundus image.
6. Send the image to `/analyze`.
7. Record the prediction, probability, morphometric features, and generated visualizations.

The model, scaler, preprocessing pipeline, feature extraction logic, and decision threshold should be versioned together for reproducible experiments.

---

# 📄 License

No explicit project license is currently identified in the repository.

Before distributing or using this project commercially, add an appropriate license and verify the licensing terms of:

- The source datasets
- Pretrained/model artifacts
- Third-party libraries
- Any externally sourced retinal images

---

# 👨‍💻 Project

**GlaucomaAI**

GitHub repository:

https://github.com/Smaron47/Glaucoma

Author/owner:

**Smaron47**

---

# ⚕️ Medical Disclaimer

**This project is for research, educational, and prototype screening purposes only.**

GlaucomaAI is not a certified medical device and must not be used as the sole basis for diagnosis, treatment, or other clinical decisions.

A qualified ophthalmologist or other appropriate healthcare professional should evaluate any suspected glaucoma case using accepted clinical examination and diagnostic procedures.

---

## ⭐ If You Find This Project Useful

Consider starring the repository and contributing improvements to the project.

```text
⭐ Star
🍴 Fork
🐛 Report issues
🔧 Submit pull requests
🧪 Share reproducible experiments
```

---

<p align="center">
  <strong>GlaucomaAI</strong><br>
  Explainable computer vision for glaucoma screening research
</p>
