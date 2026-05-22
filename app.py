# =============================================================================
#  GLAUCOMA DETECTION  —  Standalone Flask App (Single File, Inline HTML)
#
#  Usage:
#    pip install flask opencv-python-headless numpy tensorflow joblib
#    python glaucoma_app.py \
#        --model  glaucoma_fp16.tflite \
#        --scaler feat_scaler.pkl \
#        --thresh threshold.npy          # or a float like 0.48
#
#  Then open  http://localhost:5000  in your browser.
# =============================================================================

import argparse
import base64
import io
import math
import os
import sys
import tempfile
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from flask import Flask, request, jsonify

# ─────────────────────────────────────────────────────────────────────────────
# Inline HTML page (single-file, no templates folder needed)
# ─────────────────────────────────────────────────────────────────────────────
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>GlaucomaAI · Optic Disc & Cup Analyser</title>
<style>
  :root {
    --bg:       #0d1117;
    --surface:  #161b22;
    --card:     #21262d;
    --border:   #30363d;
    --accent:   #238636;
    --accent2:  #1f6feb;
    --danger:   #da3633;
    --warn:     #e3b341;
    --text:     #e6edf3;
    --muted:    #8b949e;
    --radius:   12px;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    min-height: 100vh;
  }

  /* ── Header ── */
  header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 18px 32px;
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .logo { font-size: 26px; }
  header h1 { font-size: 20px; font-weight: 700; letter-spacing: -.3px; }
  header p  { font-size: 13px; color: var(--muted); margin-top: 2px; }

  /* ── Layout ── */
  .container { max-width: 1100px; margin: 0 auto; padding: 32px 20px; }
  .grid { display: grid; grid-template-columns: 380px 1fr; gap: 24px; }
  @media(max-width:820px){ .grid{ grid-template-columns:1fr; } }

  /* ── Cards ── */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
  }
  .card-title {
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .6px;
    color: var(--muted);
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  /* ── Drop Zone ── */
  .dropzone {
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 36px 20px;
    text-align: center;
    cursor: pointer;
    transition: border-color .2s, background .2s;
    position: relative;
  }
  .dropzone:hover, .dropzone.over {
    border-color: var(--accent2);
    background: rgba(31,111,235,.06);
  }
  .dropzone input[type=file] {
    position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%;
  }
  .dropzone .icon { font-size: 36px; margin-bottom: 10px; }
  .dropzone p { color: var(--muted); font-size: 14px; }
  .dropzone strong { color: var(--text); }
  #preview {
    margin-top: 14px;
    border-radius: 8px;
    max-width: 100%;
    max-height: 220px;
    object-fit: contain;
    display: none;
    border: 1px solid var(--border);
  }

  /* ── Buttons ── */
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    border: none;
    transition: opacity .15s, transform .1s;
  }
  .btn:active { transform: scale(.97); }
  .btn-primary { background: var(--accent); color: #fff; width: 100%; justify-content: center; margin-top: 16px; font-size: 15px; padding: 12px; }
  .btn-primary:hover { opacity: .88; }
  .btn-primary:disabled { opacity: .45; cursor: not-allowed; transform: none; }

  /* ── Result Panel ── */
  #result-panel { display: none; }

  .verdict {
    border-radius: var(--radius);
    padding: 20px 24px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 18px;
    border: 1px solid;
  }
  .verdict.glaucoma { background: rgba(218,54,51,.12); border-color: var(--danger); }
  .verdict.normal   { background: rgba(35,134,54,.12);  border-color: var(--accent); }
  .verdict-icon { font-size: 40px; }
  .verdict-label { font-size: 26px; font-weight: 800; letter-spacing: -.5px; }
  .verdict-sub   { font-size: 13px; color: var(--muted); margin-top: 4px; }
  .prob-bar-wrap { margin-top: 10px; }
  .prob-bar-track {
    background: var(--border);
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
    margin-top: 6px;
  }
  .prob-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width .6s ease;
  }
  .prob-bar-fill.glaucoma { background: var(--danger); }
  .prob-bar-fill.normal   { background: var(--accent); }

  /* ── Image Tabs ── */
  .tabs { display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; }
  .tab {
    padding: 7px 16px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--muted);
    transition: all .15s;
  }
  .tab.active { background: var(--accent2); border-color: var(--accent2); color: #fff; }
  .img-panel {
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--border);
    background: #000;
    display: none;
    text-align: center;
  }
  .img-panel.active { display: block; }
  .img-panel img { max-width: 100%; max-height: 480px; object-fit: contain; display: block; margin: auto; }

  /* ── Metrics Grid ── */
  .metrics {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: 12px;
    margin-top: 20px;
  }
  .metric {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 12px;
    text-align: center;
  }
  .metric-val { font-size: 22px; font-weight: 800; }
  .metric-label { font-size: 11px; color: var(--muted); margin-top: 4px; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; }
  .high { color: var(--danger); }
  .low  { color: var(--accent); }
  .mid  { color: var(--warn);   }

  /* ── Spinner ── */
  #spinner {
    display: none;
    text-align: center;
    padding: 40px 0;
    color: var(--muted);
    font-size: 14px;
  }
  .spin {
    width: 40px; height: 40px;
    border: 4px solid var(--border);
    border-top-color: var(--accent2);
    border-radius: 50%;
    animation: spin .8s linear infinite;
    margin: 0 auto 16px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Error ── */
  #error-box {
    display: none;
    background: rgba(218,54,51,.12);
    border: 1px solid var(--danger);
    border-radius: var(--radius);
    padding: 16px 20px;
    font-size: 14px;
    color: #f08080;
    margin-top: 16px;
  }

  /* ── Legend ── */
  .legend { display: flex; gap: 16px; margin-top: 10px; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--muted); }
  .dot { width: 12px; height: 12px; border-radius: 50%; }
</style>
</head>
<body>

<header>
  <div class="logo">👁</div>
  <div>
    <h1>GlaucomaAI · Optic Disc &amp; Cup Analyser</h1>
    <p>TFLite-powered glaucoma screening with heatmap segmentation</p>
  </div>
</header>

<div class="container">
  <div class="grid">

    <!-- ── LEFT: Upload Panel ── -->
    <div>
      <div class="card">
        <div class="card-title">📤 Upload Fundus Image</div>

        <div class="dropzone" id="dropzone">
          <input type="file" id="file-input" accept="image/*"/>
          <div class="icon">🖼</div>
          <p><strong>Click or drag &amp; drop</strong></p>
          <p>JPG / PNG fundus image</p>
        </div>
        <img id="preview" alt="preview"/>

        <button class="btn btn-primary" id="analyze-btn" disabled onclick="runAnalysis()">
          🔬 Analyse Image
        </button>

        <div id="error-box"></div>
      </div>

      <!-- Feature key -->
      <div class="card" style="margin-top:16px; font-size:13px; color:var(--muted); line-height:1.9;">
        <div class="card-title">📖 Feature Guide</div>
        <div><strong style="color:var(--text)">CDR-V / CDR-H</strong> — Vertical &amp; Horizontal Cup-to-Disc Ratio. &gt;0.65 is suspicious.</div>
        <div><strong style="color:var(--text)">Disc%</strong> — Optic disc area as % of full image.</div>
        <div><strong style="color:var(--text)">Cup%</strong> — Cup area as % of full image.</div>
        <div><strong style="color:var(--text)">Rim%</strong> — Neuro-retinal rim area %.</div>
        <div><strong style="color:var(--text)">C/D-Area</strong> — Cup/Disc area ratio.</div>
        <div><strong style="color:var(--text)">D-Circ</strong> — Disc circularity (1 = perfect circle).</div>
        <div><strong style="color:var(--text)">Rim/D</strong> — Rim-to-disc ratio (lower = more glaucomatous).</div>
      </div>
    </div>

    <!-- ── RIGHT: Results Panel ── -->
    <div>
      <div id="spinner">
        <div class="spin"></div>
        Running segmentation &amp; inference…
      </div>

      <div id="placeholder" class="card" style="text-align:center; padding:60px 20px; color:var(--muted);">
        <div style="font-size:56px;margin-bottom:16px;">🔬</div>
        <div style="font-size:16px; font-weight:600; color:var(--text);">Upload an image to begin</div>
        <div style="font-size:13px; margin-top:8px;">Results will appear here after analysis</div>
      </div>

      <div id="result-panel">
        <!-- Verdict banner -->
        <div class="verdict" id="verdict-box">
          <div class="verdict-icon" id="verdict-icon"></div>
          <div style="flex:1">
            <div class="verdict-label" id="verdict-label"></div>
            <div class="verdict-sub"  id="verdict-sub"></div>
            <div class="prob-bar-wrap">
              <div style="font-size:12px; color:var(--muted);">Glaucoma probability</div>
              <div class="prob-bar-track">
                <div class="prob-bar-fill" id="prob-bar"></div>
              </div>
            </div>
          </div>
          <div style="text-align:right; min-width:70px;">
            <div style="font-size:30px; font-weight:900;" id="prob-text"></div>
            <div style="font-size:11px; color:var(--muted);">probability</div>
          </div>
        </div>

        <!-- Image viewer tabs -->
        <div class="card">
          <div class="card-title">🖼 Visualisation</div>
          <div class="tabs">
            <button class="tab active" onclick="showTab('overlay')">🎯 Overlay + Heatmap</button>
            <button class="tab"        onclick="showTab('mask')">🩺 Cup &amp; Disc Masks</button>
            <button class="tab"        onclick="showTab('crop')">✂ Disc Crop (Model Input)</button>
            <button class="tab"        onclick="showTab('original')">📷 Original</button>
          </div>

          <div id="tab-overlay"   class="img-panel active"><img id="img-overlay"  alt="heatmap overlay"/></div>
          <div id="tab-mask"      class="img-panel">        <img id="img-mask"    alt="mask"/></div>
          <div id="tab-crop"      class="img-panel">        <img id="img-crop"    alt="crop"/></div>
          <div id="tab-original"  class="img-panel">        <img id="img-original" alt="original"/></div>

          <div class="legend">
            <div class="legend-item"><div class="dot" style="background:#1e64ff"></div> Optic Disc</div>
            <div class="legend-item"><div class="dot" style="background:#ffd200"></div> Optic Cup</div>
            <div class="legend-item"><div class="dot" style="background:#00e5ff"></div> Heatmap activation</div>
          </div>
        </div>

        <!-- Metrics -->
        <div class="card" style="margin-top:16px;">
          <div class="card-title">📊 Morphometric Features</div>
          <div class="metrics" id="metrics-grid"></div>
        </div>
      </div><!-- /result-panel -->
    </div>
  </div>
</div>

<script>
const fileInput   = document.getElementById('file-input');
const dropzone    = document.getElementById('dropzone');
const preview     = document.getElementById('preview');
const analyzeBtn  = document.getElementById('analyze-btn');

let selectedFile = null;
let originalB64  = null;

// Drag & drop styling
dropzone.addEventListener('dragover',  e => { e.preventDefault(); dropzone.classList.add('over'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('over'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('over');
  if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) loadFile(fileInput.files[0]); });

function loadFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    const b64 = ev.target.result;
    originalB64 = b64;
    preview.src = b64;
    preview.style.display = 'block';
    analyzeBtn.disabled = false;
    document.getElementById('img-original').src = b64;
    hideError();
    hideResults();
  };
  reader.readAsDataURL(file);
}

async function runAnalysis() {
  if (!selectedFile) return;
  showSpinner(true);
  hideResults();
  hideError();

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const res  = await fetch('/analyze', { method: 'POST', body: formData });
    const data = await res.json();
    showSpinner(false);
    if (data.error) { showError(data.error); return; }
    renderResults(data);
  } catch(err) {
    showSpinner(false);
    showError('Request failed: ' + err.message);
  }
}

function renderResults(d) {
  // Verdict
  const isGL   = d.prediction === 'Glaucoma';
  const vBox   = document.getElementById('verdict-box');
  vBox.className = 'verdict ' + (isGL ? 'glaucoma' : 'normal');
  document.getElementById('verdict-icon').textContent  = isGL ? '⚠️' : '✅';
  document.getElementById('verdict-label').textContent = d.prediction;
  document.getElementById('verdict-sub').textContent   =
    isGL ? 'Elevated CDR detected — clinical review recommended.'
         : 'No significant glaucomatous features detected.';

  const pct = Math.round(d.probability * 100);
  document.getElementById('prob-text').textContent = pct + '%';
  const bar = document.getElementById('prob-bar');
  bar.style.width = pct + '%';
  bar.className = 'prob-bar-fill ' + (isGL ? 'glaucoma' : 'normal');

  // Images
  document.getElementById('img-overlay').src  = 'data:image/png;base64,' + d.img_overlay;
  document.getElementById('img-mask').src      = 'data:image/png;base64,' + d.img_mask;
  document.getElementById('img-crop').src      = 'data:image/png;base64,' + d.img_crop;

  // Metrics
  const grid = document.getElementById('metrics-grid');
  grid.innerHTML = '';
  const feats = d.features;
  const labels = {
    'CDR-V':   ['CDR Vertical', f => f > 0.65 ? 'high' : f > 0.5 ? 'mid' : 'low'],
    'CDR-H':   ['CDR Horizontal', f => f > 0.65 ? 'high' : f > 0.5 ? 'mid' : 'low'],
    'Disc%':   ['Disc Area %',    () => ''],
    'Cup%':    ['Cup Area %',     () => ''],
    'Rim%':    ['Rim Area %',     f => f < 0.01 ? 'high' : ''],
    'C/D-Area':['Cup/Disc Area',  f => f > 0.5 ? 'high' : 'low'],
    'D-Circ':  ['Disc Circularity', f => f < 0.7 ? 'mid' : 'low'],
    'Rim/D':   ['Rim/Disc Ratio', f => f < 0.3 ? 'high' : 'low'],
  };
  for (const [key, [label, colorFn]] of Object.entries(labels)) {
    if (feats[key] === undefined) continue;
    const val  = feats[key].toFixed(3);
    const cls  = colorFn(feats[key]);
    grid.innerHTML += `
      <div class="metric">
        <div class="metric-val ${cls}">${val}</div>
        <div class="metric-label">${label}</div>
      </div>`;
  }

  document.getElementById('placeholder').style.display = 'none';
  document.getElementById('result-panel').style.display = 'block';
  showTab('overlay');
}

function showTab(name) {
  document.querySelectorAll('.img-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}

function showSpinner(on) {
  document.getElementById('spinner').style.display = on ? 'block' : 'none';
}
function hideResults() {
  document.getElementById('result-panel').style.display = 'none';
  document.getElementById('placeholder').style.display = 'block';
}
function showError(msg) {
  const b = document.getElementById('error-box');
  b.textContent = '⚠ ' + msg;
  b.style.display = 'block';
}
function hideError() {
  document.getElementById('error-box').style.display = 'none';
}
</script>
</body>
</html>
"""

# =============================================================================
# Optic Disc / Cup Segmentation Pipeline  (merged from both your files)
# =============================================================================

FEAT_NAMES = ['CDR-V', 'CDR-H', 'Disc%', 'Cup%', 'Rim%', 'C/D-Area', 'D-Circ', 'Rim/D']
IMG_SIZE   = 224


@dataclass
class Cfg:
    max_spot:    int   = 6
    max_thresh:  int   = 15
    circ_strict: float = 0.58
    circ_relax:  float = 0.42
    edge_pct:    float = 0.05


class OpticPipeline:
    def __init__(self, cfg=None):
        self.cfg = cfg or Cfg()

    def run(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        fov  = self._fov(img_bgr)
        disc = self._hunt(img_bgr, fov, False)
        if disc is None:
            disc = self._hunt(img_bgr, fov, True)
        if disc is None:
            disc = self._fallback(img_bgr, fov)
        cup = self._cup(img_bgr, disc)
        return disc, cup

    def _fov(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, t = cv2.threshold(g, 15, 255, cv2.THRESH_BINARY)
        px = max(1, int(img.shape[1] * self.cfg.edge_pct))
        return cv2.erode(t, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px)))

    def _hunt(self, img, fov, relax):
        h, w = img.shape[:2]
        _, g, r = cv2.split(img)
        bk    = int(w * 0.08) | 1
        space = cv2.bitwise_and(cv2.GaussianBlur(g, (bk, bk), 0),
                                255 * np.ones_like(g), mask=fov)
        best, ba = None, 0
        for _ in range(self.cfg.max_spot):
            _, mx, _, loc = cv2.minMaxLoc(space)
            if mx < 20: break
            cx, cy = loc
            rw = int(w * 0.18)
            x1, y1 = max(0, cx - rw), max(0, cy - rw)
            x2, y2 = min(w, cx + rw), min(h, cy + rw)
            lr = r[y1:y2, x1:x2]
            if lr.size == 0: continue
            le = cv2.createCLAHE(2.0, (8, 8)).apply(lr)
            lb = cv2.GaussianBlur(le, (11, 11), 0)
            _, lmx, _, _ = cv2.minMaxLoc(lb)
            cands  = [cv2.threshold(lb, int(lmx * t), 255, cv2.THRESH_BINARY)[1]
                      for t in np.linspace(0.95, 0.40, self.cfg.max_thresh)]
            cands += [cv2.threshold(lb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]]
            k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            for c in cands:
                cl = cv2.morphologyEx(
                    cv2.morphologyEx(c, cv2.MORPH_OPEN, k11),
                    cv2.MORPH_CLOSE, k11, iterations=2)
                cs, _ = cv2.findContours(cl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cs: continue
                hull = cv2.convexHull(max(cs, key=cv2.contourArea))
                if self._ok(hull, lb.shape, (h, w), relax):
                    area = cv2.contourArea(hull)
                    if area > ba:
                        ba = area
                        gm = np.zeros((h, w), np.uint8)
                        cv2.drawContours(gm[y1:y2, x1:x2], [hull], -1, 255, cv2.FILLED)
                        best = gm
            if best is not None:
                return best
            cv2.circle(space, (cx, cy), int(w * 0.12), 0, -1)
        return None

    def _ok(self, cnt, ls, gs, relax):
        a  = cv2.contourArea(cnt)
        ga = gs[0] * gs[1]
        if a < ga * (0.001 if relax else 0.002): return False
        if a > ga * (0.08  if relax else 0.05 ): return False
        bx, by, bw, bh = cv2.boundingRect(cnt)
        em = 3
        if sum([bx <= em, bx + bw >= ls[1] - em,
                by <= em, by + bh >= ls[0] - em]) >= 2:
            return False
        p = cv2.arcLength(cnt, True)
        if p == 0: return False
        if 4 * math.pi * a / p ** 2 < (self.cfg.circ_relax if relax else self.cfg.circ_strict):
            return False
        if not (0.55 <= float(bw) / bh <= 1.55): return False
        return True

    def _fallback(self, img, fov):
        h, w = img.shape[:2]
        bl = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        _, _, _, loc = cv2.minMaxLoc(cv2.bitwise_and(bl, bl, mask=fov))
        m = np.zeros((h, w), np.uint8)
        cv2.circle(m, loc, int(w * 0.08), 255, -1)
        return m

    def _cup(self, img, disc):
        h, w = img.shape[:2]
        _, g, _ = cv2.split(img)
        bg = cv2.GaussianBlur(cv2.createCLAHE(2.0, (8, 8)).apply(g), (9, 9), 0)
        px = bg[disc > 0]
        if px.size == 0:
            return np.zeros((h, w), np.uint8)
        mn, sd = float(px.mean()), float(px.std())
        cs, _ = cv2.findContours(disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        da  = cv2.contourArea(cs[0]) if cs else 1.0
        cup = np.zeros((h, w), np.uint8)
        k5  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for sigma in [1.2, 1.0, 0.8, 0.5, 0.3, 0.1]:
            _, t = cv2.threshold(bg, int(mn + sigma * sd), 255, cv2.THRESH_BINARY)
            iso  = cv2.morphologyEx(cv2.bitwise_and(t, t, mask=disc), cv2.MORPH_OPEN, k5)
            cc, _ = cv2.findContours(iso, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cc:
                lc = max(cc, key=cv2.contourArea)
                ca = cv2.contourArea(lc)
                if da * 0.02 < ca < da * 0.90:
                    cv2.drawContours(cup, [cv2.convexHull(lc)], -1, 255, cv2.FILLED)
                    return cup
        if cs:
            (cx, cy), r = cv2.minEnclosingCircle(cs[0])
            cv2.circle(cup, (int(cx), int(cy)), int(r * 0.42), 255, -1)
        return cup


# =============================================================================
# Feature extraction
# =============================================================================

def morph_features(disc: np.ndarray, cup: np.ndarray) -> np.ndarray:
    h, w = disc.shape
    dc, _ = cv2.findContours(disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cc, _ = cv2.findContours(cup,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not dc:
        return np.zeros(8, np.float32)
    _, _, dw, dh = cv2.boundingRect(dc[0])
    da  = cv2.contourArea(dc[0])
    dp  = cv2.arcLength(dc[0], True)
    dcirc = 4 * math.pi * da / (dp ** 2 + 1e-6)
    if not cc:
        cdr_v = cdr_h = ca = 0.0
    else:
        _, _, cw, ch = cv2.boundingRect(cc[0])
        ca    = cv2.contourArea(cc[0])
        cdr_v = ch / (dh + 1e-6)
        cdr_h = cw / (dw + 1e-6)
    rim = max(0.0, da - ca)
    return np.array([cdr_v, cdr_h,
                     da / (h * w), ca / (h * w), rim / (h * w),
                     ca / (da + 1e-6), dcirc, rim / (da + 1e-6)], np.float32)


def extract_roi(img_bgr: np.ndarray, pipeline: OpticPipeline, sz=(IMG_SIZE, IMG_SIZE)):
    disc, cup = pipeline.run(img_bgr)
    if disc is None or disc.sum() == 0:
        raise ValueError("Disc segmentation failed")
    cs, _ = cv2.findContours(disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        raise ValueError("No disc contour found")
    x, y, w, h = cv2.boundingRect(cs[0])
    pad = int(max(w, h) * 0.25)
    H, W = disc.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    crop = img_rgb[max(0, y - pad):min(H, y + h + pad),
                   max(0, x - pad):min(W, x + w + pad)]
    if crop.size == 0:
        raise ValueError("Empty crop")
    return cv2.resize(crop, sz), morph_features(disc, cup), disc, cup


# =============================================================================
# Visualisation helpers
# =============================================================================

def make_mask_visual(img_bgr: np.ndarray, disc: np.ndarray, cup: np.ndarray) -> np.ndarray:
    """Coloured overlay showing disc (blue) and cup (yellow) filled regions + contours."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    ov  = rgb.copy()

    # Fill disc region (blue tint)
    disc_bool = disc > 0
    ov[disc_bool] = ov[disc_bool] * 0.45 + np.array([30, 100, 255]) * 0.55

    # Fill cup region on top (yellow tint)
    cup_bool = cup > 0
    ov[cup_bool] = ov[cup_bool] * 0.45 + np.array([255, 210, 0]) * 0.55

    ov = ov.clip(0, 255).astype(np.uint8)

    # Draw contours
    dc, _ = cv2.findContours(disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cc, _ = cv2.findContours(cup,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lw = max(2, int(img_bgr.shape[1] * 0.003))
    if dc and len(dc[0]) >= 5:
        cv2.ellipse(ov, cv2.fitEllipse(dc[0]), (30, 100, 255), lw)
    elif dc:
        cv2.drawContours(ov, dc, -1, (30, 100, 255), lw)
    if cc and len(cc[0]) >= 5:
        cv2.ellipse(ov, cv2.fitEllipse(cc[0]), (255, 210, 0), lw)
    elif cc:
        cv2.drawContours(ov, cc, -1, (255, 210, 0), lw)

    return ov


def make_heatmap_overlay(img_bgr: np.ndarray, disc: np.ndarray, cup: np.ndarray) -> np.ndarray:
    """
    JET heatmap on the disc+cup zone blended with original image.
    Contours + ellipse fits drawn on top.  CDR annotation added.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]

    # Build a combined binary for the heatmap zone
    zone = ((disc > 0) | (cup > 0)).astype(np.uint8) * 255

    # Activation canvas from green channel (bright structures = high activation)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    bf   = max(5, int(w * 0.015)) | 1
    blur = cv2.GaussianBlur(gray, (bf, bf), 0)

    canvas = np.zeros_like(blur)
    m = zone > 0
    if m.sum() > 0:
        vals = blur[m]
        mn_v, mx_v = vals.min(), vals.max()
        if mx_v > mn_v:
            canvas[m] = (vals - mn_v) / (mx_v - mn_v + 1e-6)

    hm_u8    = np.uint8(255 * canvas)
    hm_color = cv2.cvtColor(cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

    # Blend only inside the zone
    output = img_rgb.copy()
    blended = cv2.addWeighted(img_rgb, 0.35, hm_color, 0.65, 0)
    output[m] = blended[m]

    # Contour lines
    dc, _ = cv2.findContours(disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cc, _ = cv2.findContours(cup,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lw = max(2, int(w * 0.003))

    if dc and len(dc[0]) >= 5:
        cv2.ellipse(output, cv2.fitEllipse(dc[0]), (0, 255, 0), lw)
    elif dc:
        cv2.drawContours(output, dc, -1, (0, 255, 0), lw)

    if cc and len(cc[0]) >= 5:
        cv2.ellipse(output, cv2.fitEllipse(cc[0]), (255, 255, 255), max(1, lw - 1))
    elif cc:
        cv2.drawContours(output, cc, -1, (255, 255, 255), max(1, lw - 1))

    # CDR annotation
    feats = morph_features(disc, cup)
    cdr_v = feats[0]
    label = f"CDR-V: {cdr_v:.3f}"
    fs    = max(0.5, w / 800)
    th    = max(1, int(fs * 2))
    (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, th)
    pad_x, pad_y = 12, 10
    cv2.rectangle(output, (pad_x - 4, pad_y - 4), (pad_x + tw + 8, pad_y + tht + 8),
                  (0, 0, 0), -1)
    cv2.putText(output, label, (pad_x, pad_y + tht),
                cv2.FONT_HERSHEY_DUPLEX, fs, (0, 255, 160), th, cv2.LINE_AA)

    return output


def ndarray_to_b64png(arr: np.ndarray) -> str:
    _, buf = cv2.imencode('.png', cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode('utf-8')


def crop_to_b64png(crop_rgb: np.ndarray) -> str:
    _, buf = cv2.imencode('.png', cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode('utf-8')


# =============================================================================
# TFLite Detector  (loads lazily once on first request)
# =============================================================================

class GlaucomaDetector:
    def __init__(self, tflite_path: str, scaler_path: str, threshold: float):
        import tensorflow as tf
        import joblib
        self.interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        self.interpreter.allocate_tensors()
        self.in_det  = self.interpreter.get_input_details()
        self.out_det = self.interpreter.get_output_details()
        self.scaler  = joblib.load(scaler_path)
        self.thr     = threshold
        self._img_idx  = None
        self._feat_idx = None
        for d in self.in_det:
            if len(d['shape']) == 4:
                self._img_idx  = d['index']
            elif len(d['shape']) == 2:
                self._feat_idx = d['index']

    def predict(self, img_bgr: np.ndarray, pipeline: OpticPipeline):
        from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
        crop, feats, disc, cup = extract_roi(img_bgr, pipeline)

        img_in  = eff_pre(crop.astype(np.float32)[np.newaxis])
        feat_in = self.scaler.transform([feats]).astype(np.float32)

        for d in self.in_det:
            if d['index'] == self._img_idx:
                if d['dtype'] == np.uint8:
                    scale, zero = d['quantization']
                    img_in = (img_in / scale + zero).clip(0, 255).astype(np.uint8)
                self.interpreter.set_tensor(d['index'], img_in)
            elif d['index'] == self._feat_idx:
                if d['dtype'] == np.uint8:
                    scale, zero = d['quantization']
                    feat_in = (feat_in / scale + zero).clip(0, 255).astype(np.uint8)
                self.interpreter.set_tensor(d['index'], feat_in)

        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.out_det[0]['index']).flatten()
        od  = self.out_det[0]
        if od['dtype'] == np.uint8:
            scale, zero = od['quantization']
            prob = float((raw[0] - zero) * scale)
        else:
            prob = float(raw[0])

        return prob, 'Glaucoma' if prob >= self.thr else 'Normal', crop, disc, cup, feats


# =============================================================================
# Flask App
# =============================================================================

app      = Flask(__name__)
pipeline = OpticPipeline()
detector: Optional[GlaucomaDetector] = None  # set after arg parse


@app.route('/')
def index():
    return HTML_PAGE


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    file   = request.files['image']
    data   = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({'error': 'Could not decode image. Upload a valid JPG/PNG.'}), 400

    try:
        if detector is not None:
            # ── Full TFLite inference path ──────────────────────────────────
            prob, pred, crop, disc, cup, feats = detector.predict(img_bgr, pipeline)
        else:
            # ── Segmentation-only path (no model files provided) ────────────
            disc, cup = pipeline.run(img_bgr)
            feats = morph_features(disc, cup)
            cdr_v = float(feats[0])
            # Simple CDR-based heuristic when no model is loaded
            prob  = float(np.clip((cdr_v - 0.30) / 0.50, 0.0, 1.0))
            pred  = 'Glaucoma' if cdr_v >= 0.65 else 'Normal'
            cs, _ = cv2.findContours(disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cs:
                x, y, w, h = cv2.boundingRect(cs[0])
                pad = int(max(w, h) * 0.25)
                H, W = disc.shape
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                crop = img_rgb[max(0, y - pad):min(H, y + h + pad),
                               max(0, x - pad):min(W, x + w + pad)]
                crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)) if crop.size > 0 \
                       else np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
            else:
                crop = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)

        # ── Generate visuals ────────────────────────────────────────────────
        overlay_rgb = make_heatmap_overlay(img_bgr, disc, cup)
        mask_rgb    = make_mask_visual(img_bgr, disc, cup)

        return jsonify({
            'prediction':  pred,
            'probability': round(prob, 4),
            'features':    dict(zip(FEAT_NAMES, [round(float(v), 4) for v in feats])),
            'img_overlay': ndarray_to_b64png(overlay_rgb),
            'img_mask':    ndarray_to_b64png(mask_rgb),
            'img_crop':    crop_to_b64png(crop),
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


# =============================================================================
# Entry point
# =============================================================================

def load_threshold(path_or_val: str) -> float:
    p = Path(path_or_val)
    if p.exists():
        return float(np.load(str(p)).flat[0])
    return float(path_or_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GlaucomaAI Flask Web App',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # With model files (full TFLite inference):
  python glaucoma_app.py \\
      --model  glaucoma_fp16.tflite \\
      --scaler feat_scaler.pkl \\
      --thresh threshold.npy

  # Without model (segmentation + CDR heuristic only):
  python glaucoma_app.py

  # Custom port:
  python glaucoma_app.py --port 8080
"""
    )
    parser.add_argument('--model',  default=r"model\glaucoma_fp16 (1).tflite",
                        help='Path to glaucoma_fp16.tflite')
    parser.add_argument('--scaler', default=r"model\feat_scaler (1).pkl",
                        help='Path to feat_scaler.pkl')
    parser.add_argument('--thresh', default='0.40',
                        help='Path to threshold.npy OR a float (default: 0.50)')
    parser.add_argument('--host',   default='0.0.0.0',
                        help='Host to bind (default: 0.0.0.0)')
    parser.add_argument('--port',   type=int, default=5000,
                        help='Port (default: 5000)')
    parser.add_argument('--debug',  action='store_true',
                        help='Enable Flask debug mode')
    args = parser.parse_args()

    # Load model if all three paths are provided
    if args.model and args.scaler:
        if not Path(args.model).exists():
            print(f"[WARN] Model file not found: {args.model}  — running in segmentation-only mode.")
        elif not Path(args.scaler).exists():
            print(f"[WARN] Scaler file not found: {args.scaler} — running in segmentation-only mode.")
        else:
            thr = load_threshold(args.thresh)
            print(f"[INFO] Loading model: {args.model}")
            print(f"[INFO] Loading scaler: {args.scaler}")
            print(f"[INFO] Decision threshold: {thr:.3f}")
            detector = GlaucomaDetector(args.model, args.scaler, thr)
            print("[INFO] Model loaded successfully.")
    else:
        print("[INFO] No model/scaler provided — running in CDR-heuristic segmentation mode.")

    print(f"\n[INFO] Starting server on http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug)
