# RestoreAI

**Generative AI for Intelligent Image Restoration**
B.Tech CSE (AI & Data Science), Semester V — Ramdeobaba University
Team: Tanushri Nikhare, Om Bhitade, Safee Ali Khan, Siddhant Mustilwar · Guide: Prof. Ashish Chandak

A working web application that restores degraded photographs using a
restoration pipeline built from **two models the team trained from
scratch** (denoising + 2× super-resolution, trained on BSDS300 in
Colab) plus classical OpenCV techniques for scratch repair, deblurring,
and finishing — with before/after comparison, objective quality
metrics, and MySQL-backed restoration history.

---

## 1. Project overview

The system lets a user upload a degraded image, automatically analyzes
what's wrong with it (noise, blur, low resolution, scratches), runs the
relevant restoration steps — preferring the team's own trained models
where available — and shows a before/after comparison with downloadable
output. If the user also supplies the original clean image, the app
computes real PSNR, SSIM, and MSE against it; otherwise it clearly says
those metrics aren't available, rather than making numbers up.

## 2. Features

- Drag-and-drop upload (JPG, JPEG, PNG, WEBP, up to 16 MB)
- Automatic degradation analysis, or 6 manual restoration modes
- **Trained models**: a denoising CNN and a 2× super-resolution CNN,
  trained from scratch on BSDS300 (see `RestoreAI_Train_Models.ipynb`)
- Classical fallback: Non-Local Means denoising, pretrained ESPCN
  (3×/4×), scratch inpainting, detail-aware deblurring
- Live processing screen that waits on the actual backend response
- Before/after comparison slider, fullscreen preview
- Objective PSNR / SSIM / MSE when a reference image is supplied
- Downloadable restored image
- MySQL-backed restoration history with search, sort, and delete

## 3. Architecture

```
Browser (HTML/CSS/JS)
      │  fetch/AJAX
      ▼
Flask app (app.py) ── REST API
      │
      ▼
services/restoration.py  (pipeline orchestrator)
      │
      ├─ services/preprocessing.py   (validate, normalize, analyze degradation)
      ├─ models/restoration_model.py (chooses trained vs. classical per step)
      │      ├─ models/trained_models.py    (DenoiseNet / SRNet, ONNX via OpenCV DNN)
      │      └─ models/super_resolution.py  (pretrained ESPCN — fallback / 3x/4x)
      ├─ services/postprocessing.py  (contrast, color, artifact reduction, sharpen)
      └─ services/metrics.py         (PSNR / SSIM / MSE — only with a reference image)
      │
      ▼
database.py ── MySQL (restoration_history table)
```

**How the trained-vs-classical choice works:** `models/restoration_model.py`
calls `denoise()` and `super_resolve()`, both of which check whether the
team's trained ONNX model is present in `weights/` and runs it if so —
falling back automatically to the classical method if the file is
missing or fails to load. Every restoration response lists exactly
which method actually ran for each step.

## 4. Technology stack

| Layer | Choice |
|---|---|
| Language | Python 3.12 |
| Backend | Flask (local core web app) + FastAPI (Colab GPU service) |
| Frontend | HTML, custom CSS, vanilla JavaScript (Fetch API) |
| Database | MySQL |
| High-End Generative AI | **HYPIR** (Diffusion-Yielded Score Priors, SIGGRAPH 2025) running on dedicated Google Colab GPU runtime |
| Edge AI & Fallback | Real-ESRGAN GAN, FSRCNN, ESPCN via OpenCV DNN (`cv2.dnn.readNetFromONNX`) — runs locally without heavy VRAM requirements |
| Image libraries | OpenCV, NumPy, scikit-image, Pillow |

## 5. The restoration models

- **Real-ESRGAN (Generative AI / GAN)** — Generative Adversarial Network for photorealistic clarity, micro-texture recovery, and super-resolution up to 4×. Uses memory-efficient tiled inference for arbitrary image sizes.
- **FSRCNN & ESPCN** — Deep super-resolution sub-pixel CNN networks for fast multi-scale upscaling (2×, 3×, 4×).
- **DenoiseNet & SRNet** — Custom models trained from scratch on BSDS300 (see `RestoreAI_Train_Models.ipynb`).
- **High-Definition Finishing** — Dual-domain edge-preserving filtering, CLAHE contrast enhancement, perceptual color vibrance, and adaptive clarity sharpening.

## 6. Project structure

```
restore-ai/
├── app.py                    # Flask entrypoint & all routes
├── config.py                 # Environment-driven configuration
├── database.py                # MySQL access layer
├── requirements.txt
├── .env.example
├── database/
│   └── schema.sql
├── models/
│   ├── restoration_model.py  # Picks trained vs. classical per step
│   ├── trained_models.py     # ONNX wrappers for DenoiseNet / SRNet
│   └── super_resolution.py   # Pretrained ESPCN wrapper (fallback)
├── services/
│   ├── preprocessing.py
│   ├── restoration.py
│   ├── hypir_client.py       # Client connecting to Google Colab HYPIR GPU API
│   ├── postprocessing.py
│   └── metrics.py
├── utils/
│   └── image_utils.py
├── templates/                # index, restore, result, history, about
├── static/
│   ├── css/style.css
│   ├── js/{app,restore,result,history}.js
│   └── images/
├── weights/                  # Lightweight local edge weights (Real-ESRGAN ONNX, ESPCN)
│   ├── realesr-general-x4v3.onnx
│   └── ESPCN_x2.pb / x3 / x4
├── uploads/ / outputs/        (created at runtime)
├── Restore_AI_HYPIR.ipynb    # Dedicated Google Colab GPU notebook for HYPIR
└── RestoreAI_Train_Models.ipynb
```

## 7. Google Colab GPU Runtime for HYPIR

To deliver diffusion-grade photorealistic image restoration without overloading local hardware:
- All heavy model downloads (~850 MB LoRA + Stable Diffusion 2.1 base checkpoints) and CUDA inferences run exclusively inside **Google Colab**.
- Colab exposes a RESTful FastAPI server through a secure **Cloudflare HTTPS tunnel** (`trycloudflare.com`).
- Your local RestoreAI website transmits uploaded images to Colab and receives restored outputs in seconds.

### How to Start Colab HYPIR:
1. Upload and open `Restore_AI_HYPIR.ipynb` in [Google Colab](https://colab.research.google.com).
2. Set runtime hardware: `Runtime` → `Change runtime type` → select **T4 GPU** → Save.
3. Run all cells (`Ctrl+F9`).
4. Copy the generated public HTTPS URL from Step 6 (e.g., `https://xxxx.trycloudflare.com`).
5. Open your local RestoreAI website (`http://localhost:5000/restore`):
   - Click **⚙️ Colab Tunnel URL**, paste the URL, and click **Test & Connect**.
   - Or paste it into your local `.env`: `RESTORE_AI_API_URL=https://xxxx.trycloudflare.com`.

> **Note on Temporary Storage:** Colab storage is ephemeral. Checkpoints are kept in Colab during your session. If Colab disconnects, re-run Steps 6 & 7 to generate a new tunnel URL and paste it into the website.

## 8. Installation

### 7.1 Prerequisites
- Python 3.10+
- MySQL Server 8.x (or MariaDB 10.x)
- pip

### 7.2 Python environment setup

```bash
cd restore-ai
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 7.3 MySQL setup

```sql
CREATE USER 'restoreai'@'localhost' IDENTIFIED BY 'your-password';
GRANT ALL PRIVILEGES ON restore_ai.* TO 'restoreai'@'localhost';
FLUSH PRIVILEGES;
```

(Or `mysql -u root -p < database/schema.sql` to create it by hand.)

### 7.4 Environment configuration

```bash
cp .env.example .env
```

Edit `.env` and set `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `FLASK_SECRET_KEY`.

### 7.5 Model weights setup

The pretrained ESPCN weights (classical fallback) are already included
in `weights/`. If that folder is ever empty, re-fetch them with:

```bash
python scripts/download_weights.py
```

**Your trained models** (`denoise_model.onnx`, `sr_model_x2.onnx`) come
from running `RestoreAI_Train_Models.ipynb` in Colab and downloading
the exported files — place both in `weights/`. Without them, the app
still works correctly using the classical methods only.

## 8. Running the application

```bash
python app.py
```

Open **http://localhost:5000**. On startup the app logs which weights
it found; check the terminal for `[RestoreAI]` lines if a step seems to
be using the fallback method when you expected the trained one.

## 9. How to test it with a sample image

1. Open `/restore`, upload a degraded photo, leave mode on **Automatic**.
2. Click **Restore image** — this is a real network request.
3. On the Results page, check **"Steps actually applied"** — it will say
   `trained DenoiseNet model` / `trained SRNet model` if your weights
   loaded, or the classical equivalents if not.
4. Optionally upload the original clean image as a reference for real
   PSNR/SSIM/MSE.
5. Download, or save to history.

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| Steps always say "classical" even after adding .onnx files | Confirm the files are named exactly `denoise_model.onnx` / `sr_model_x2.onnx` and sit directly in `weights/`. Check the terminal for a loading error. |
| `onnx.onnx_cpp2py_export.checker.ValidationError` / weights missing when exporting from Colab | Recent PyTorch versions write weights to a separate `<name>.onnx.data` file by default. The notebook's export cells already reload and re-save as a single self-contained file — if you edited them, make sure that step is still there. |
| `Could not connect to MySQL` | Confirm MySQL is running and `.env` credentials are correct. Restoration still works without MySQL; only History is affected. |
| `Super-resolution weights not found` (ESPCN) | Run `python scripts/download_weights.py`. |
| `413 File too large` | Upload limit is 16 MB (`MAX_CONTENT_LENGTH` in `config.py`). |
| Port 5000 already in use | Change the port in `app.run(...)` at the bottom of `app.py`. |

## 11. Project limitations

- SRNet (trained) only covers 2× upscaling; 3×/4× use the pretrained ESPCN.
- Deblurring is an unsharp-mask approximation, not true blind deconvolution.
- Scratch repair handles thin scratches/marks, not large missing regions.
- Both trained models were trained on BSDS300 (300 natural images) —
  a larger/more diverse dataset (e.g. DIV2K) would likely improve
  generalization further.
- PSNR/SSIM/MSE are only computed when a genuine reference image is supplied.

## 12. Future enhancements

- Train SRNet at 3×/4× as well.
- Train on a larger dataset (DIV2K) for better generalization.
- Swap in a GAN-based restoration model.
- A learned inpainting model (e.g. LaMa) for large missing regions.
- Real-time video restoration, cloud deployment, user accounts.

---

## Honesty about the AI pipeline

| Technique | Status | What it actually is |
|---|---|---|
| Denoising | **Trained model** | DenoiseNet, trained from scratch on BSDS300 — falls back to classical Non-Local Means if weights are absent |
| Super-resolution (2×) | **Trained model** | SRNet, trained from scratch on BSDS300 — falls back to pretrained ESPCN if weights are absent |
| Super-resolution & Clarity | **Implemented** | Real-ESRGAN GAN (4×/3×/2×) + FSRCNN/ESPCN deep models |
| Deblurring | Implemented | Multi-scale high-frequency detail synthesis and unsharp masking |
| Scratch/damage repair | Implemented | Automatic detection + OpenCV Telea inpainting |
| PSNR / SSIM / MSE | Implemented, conditional | Real computation via scikit-image, only with a reference image |
| GAN-based restoration | **Implemented** | Real-ESRGAN (Generative Adversarial Network) via OpenCV DNN |
| Diffusion-based inpainting | Future work | — |
| Transformer-based restoration | Future work | — |

This mapping is also shown on the app's **About** page.

## Mapping to the project synopsis

- **Objectives 1–4** (restore, remove noise/blur/scratches, improve
  resolution, reconstruct damaged regions) → `models/trained_models.py`,
  `models/restoration_model.py`, `models/super_resolution.py`
- **"AI Model Development" (Week 5–10)** → `RestoreAI_Train_Models.ipynb`:
  two models trained from scratch, not just pretrained weights wired up
- **Objective 5** (easy-to-use interface) → `templates/`, `static/`
- **Objective 6** (evaluate with standard metrics) → `services/metrics.py`,
  plus the notebook's own PSNR evaluation of the trained models against
  baselines
- **Technology stack** → matched exactly; see Section 4
- **Workflow diagram** (upload → preprocessing → generative model →
  restoration → post-processing → evaluation → display → download →
  history) → implemented step-for-step in `services/restoration.py` and `app.py`
