# Local UI Analyzer

A local, privacy-first alternative to Attention Insight. Generates AI-powered attention heatmaps, scanpath predictions, scroll depth analysis, and accessibility audits for any website or screenshot.

**Stack:** EML-NET v2 (saliency) + Microsoft OmniParser V2 (UI detection) + Gemini (reports) + React frontend + Flask API

---

## Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.13 |
| Node.js | 18 | 20+ |
| GPU | CUDA-capable NVIDIA | 8 GB+ VRAM |
| OS | Windows 10/11 | Windows 11 |
| RAM | 8 GB | 16 GB+ |

> CPU-only mode works but is significantly slower. The saliency engine and OmniParser both benefit heavily from CUDA.

---

## Step 1 — Clone and enter the project

```bash
cd local-ui-analyzer
```

---

## Step 2 — Create a Python virtual environment

```bash
python -m venv venv
```

Activate it:

```powershell
# PowerShell
.\venv\Scripts\Activate.ps1

# CMD
venv\Scripts\activate.bat
```

---

## Step 3 — Install PyTorch with CUDA

PyTorch must be installed separately with the correct CUDA index URL. The included helper script handles this:

```bash
install_torch.bat
```

Or install manually:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

> Adjust `cu128` to match your CUDA version (e.g., `cu121` for CUDA 12.1). Check yours with `nvidia-smi`.

Verify the installation:

```bash
python test_installation.py
```

You should see `CUDA Available: True` and `Model Forward Pass: SUCCESS`.

---

## Step 4 — Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs: `opencv-python`, `numpy`, `scipy`, `matplotlib`, `jinja2`, `playwright`, `google-genai`, `python-dotenv`, `flask`, `flask-cors`, `albumentations`, `tqdm`, `Pillow`, and `ollama`.

---

## Step 5 — Install Playwright browsers

Playwright is used to capture full-page screenshots from URLs.

```bash
playwright install chromium
```

---

## Step 6 — Download model weights

The analyzer uses two AI models. Their weights must be placed in the correct directories.

### EML-NET (saliency prediction)

Place the trained model file at:

```
local-ui-analyzer/
  models/
    eml_net_hybrid.pth
```

This file is tracked via Git (or Git LFS if large). If missing, the engine falls back to Gaussian blob estimation.

### OmniParser V2 (UI element detection)

Clone Microsoft's OmniParser into the models directory:

```bash
cd models
git clone https://github.com/microsoft/OmniParser.git
cd OmniParser
```

Download the required weights into `models/OmniParser/weights/`:

```
models/OmniParser/
  weights/
    icon_detect/
      model.pt          # YOLOv8 icon detection model
    icon_caption/
      model.safetensors  # Florence-2 captioning model
      (+ config/tokenizer files)
```

Follow the [OmniParser README](https://github.com/microsoft/OmniParser) for weight download instructions.

---

## Step 7 — Set up environment variables

Copy the example and add your Gemini API key:

```bash
copy .env.example .env
```

Edit `.env`:

```
GEMINI_API_KEY=your_google_gemini_api_key_here
```

The Gemini API key is used for:

- AI-generated accessibility audit reports
- UX overview analysis (when page context is provided)

> Get a key at [aistudio.google.com](https://aistudio.google.com/). The tool works without it, but the AI report tabs will be empty.

---

## Step 8 — Build the React frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

This compiles the React app into `frontend/dist/`, which Flask serves as static files.

---

## Step 9 — Run the analyzer

### Option A: Web UI (recommended)

Start the Flask server:

```bash
venv\Scripts\python.exe server.p
```

Open **<http://localhost:5000>** in your browser. From there you can:

- Paste a URL or upload a screenshot
- Choose device type (mobile / tablet / desktop)
- Provide optional page context for AI-powered UX overview
- View attention heatmaps, scanpath, scroll depth, and reports

### Option B: CLI with HTML report

```bash
# Analyze a website (desktop)
python main.py https://example.com

# Analyze with mobile viewport
python main.py https://example.com --mobile

# Analyze a local screenshot
python main.py screenshot.png

# Custom viewport
python main.py https://example.com --viewport 1440x900

# JSON output (for programmatic use)
python main.py https://example.com --json
```

The CLI generates an interactive HTML report in the `output/` directory and opens it in your browser automatically. Use `--no-open` to suppress that.

---

## Project structure

```
local-ui-analyzer/
├── server.py              # Flask API server (serves React frontend)
├── main.py                # CLI entry point
├── analyze.py             # Core analysis pipeline (heatmaps, scores, metrics)
├── screenshot.py          # Playwright-based website screenshot capture
├── saliency_bridge.py     # EML-NET inference engine
├── eml_net_model.py       # EML-NET architecture (EfficientNet-V2 + FPN)
├── hybrid_loader.py       # Dataset loader for model training
├── test_installation.py   # CUDA / model verification script
├── test_model.py          # Model inference test
├── install_torch.bat      # PyTorch + CUDA installer (Windows)
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── .gitignore
│
├── models/
│   ├── eml_net_hybrid.pth # Trained EML-NET weights
│   └── OmniParser/        # Microsoft OmniParser V2 (git submodule/clone)
│       └── weights/
│
├── utils/
│   └── omniparser_wrapper.py  # OmniParser V2 integration
│
├── templates/
│   └── report.html        # Jinja2 template for CLI HTML reports
│
├── frontend/              # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx        # Main application component
│   │   ├── App.css        # Styles
│   │   └── index.css      # Global styles
│   ├── package.json
│   └── vite.config.js
│
└── output/                # Generated analysis results (gitignored)
```

---

## What it produces

| Output | Description |
|---|---|
| **Attention Heatmap** | EML-NET saliency + OmniParser boxes fused into a JET colormap overlay |
| **Areas of Interest** | Bounding boxes around detected UI elements ranked by attention |
| **Scanpath** | Deterministic WTA (Winner-Take-All) predicted eye movement path |
| **Scroll Depth** | Per-fold attention distribution and engagement drop-off |
| **Mouse Movement** | Simulated cursor traces based on saliency-weighted element paths |
| **Focus Score** | 0-100 — how concentrated attention is on key elements |
| **Clarity Score** | 0-100 — visual complexity and scannability |
| **ATF Attention** | Percentage of total attention captured above the fold |
| **Accessibility Report** | AI-generated WCAG audit (via Gemini) |
| **UX Overview** | AI-generated UX strategy report with recommendations (via Gemini) |

---

## Troubleshooting

**`CUDA not available`** — Make sure you installed the correct PyTorch build for your CUDA version. Run `nvidia-smi` to check your driver's CUDA version, then reinstall with the matching index URL.

**`OmniParser failed`** — Ensure the OmniParser weights are downloaded into `models/OmniParser/weights/`. The YOLO model (`icon_detect/model.pt`) and Florence model (`icon_caption/`) are both required.

**`EML-NET model not found`** — Place `eml_net_hybrid.pth` in the `models/` directory. Without it, heatmaps use a Gaussian fallback that is less accurate.

**`playwright._impl._errors.Error: Executable doesn't exist`** — Run `playwright install chromium` to download browser binaries.

**`GEMINI_API_KEY not found`** — Copy `.env.example` to `.env` and add your key. The analyzer still works without it, but AI-generated reports will be unavailable.

**Port 5000 already in use** — Another process is using port 5000. Kill it or edit `server.py` to use a different port.
