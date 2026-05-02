# XTTS Long-Form TTS Pipeline

A comprehensive text-to-speech (TTS) pipeline for generating high-quality, long-form audio using Coqui XTTS-v2 and fine-tuned transformer models. This project demonstrates both state-of-the-art TTS synthesis and advanced voice cloning techniques.

## Features

- **Long-Form TTS Synthesis** — Generate natural-sounding audio for extended text using single-shot synthesis with intelligent fallback strategies
- **Voice Cloning** — Clone voice characteristics from reference audio samples
- **Robust Text Processing** — Automatic text cleaning and normalization
- **Intelligent Segmentation** — Two-segment fallback strategy for handling long content
- **Audio Quality Control** — Cross-fade joining for seamless multi-segment synthesis
- **LoRA Fine-Tuning** — Fine-tune CSM-1B model on custom voice datasets (LJSpeech example included)
- **Lightweight Demo Runner** — Easy-to-use Python CLI for quick synthesis

## Project Contents

### Notebooks
- **`xtts_longform_pipeline.ipynb`** — Complete long-form TTS workflow using Coqui XTTS-v2 with text cleaning, two-segment fallback, and quality checks
- **`Updated csm-1b.ipynb`** — LoRA fine-tuning of `unsloth/csm-1b` model on LJSpeech dataset and inference for ~5 minute audio composition

### Python Modules
- **`xtts_pipeline.py`** — Reusable helper functions:
  - `clean_text()` — Smart text normalization (quotes, punctuation, whitespace)
  - `cross_fade_join()` — Seamless audio segment merging with fade effects
  - `tts_single_shot()` — Single-pass long-form synthesis
  - `tts_two_segment_fallback()` — Robust fallback for longer content
- **`run_inference.py`** — Lightweight CLI demo runner
- **`smoke_test.py`** — Quick validation tests (no heavy model dependencies)

### Assets
- `reference.wav` — Example reference voice sample for voice cloning
- Example outputs: `xtts_single.wav`, `xtts_segA.wav`, `xtts_segB.wav`, `xtts_joined.wav`

## Installation

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU recommended (CPU mode supported but slower)

### Setup (Windows PowerShell)

1. **Clone the repository**
```powershell
git clone https://github.com/yourusername/xtts-longform-pipeline.git
cd xtts-longform-pipeline
```

2. **Create and activate a virtual environment**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. **Install dependencies**
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Quick Demo
Generate audio from text using the reference voice:

```powershell
python run_inference.py --reference reference.wav --out output.wav
```

### Custom Text Input
Process text from a file:

```powershell
python run_inference.py --reference reference.wav --out output.wav --script-file input.txt
```

### Run Tests
Validate text processing utilities (no TTS dependencies required):

```powershell
python smoke_test.py
```

### Interactive Notebooks
For detailed workflows and fine-tuning experiments:

1. Launch Jupyter:
```powershell
jupyter notebook
```

2. Open either notebook to explore:
   - `xtts_longform_pipeline.ipynb` for TTS synthesis
   - `Updated csm-1b.ipynb` for model fine-tuning

## Technical Details

### Dependencies
Core dependencies (from `requirements.txt`):
- **PyTorch** (2.5.1) — Deep learning framework with CUDA support
- **TTS** (0.22.x) — Coqui TTS library with XTTS-v2 models
- **Transformers** (4.39+) — Hugging Face transformers for model inference
- **Librosa** — Audio processing and analysis
- **Soundfile** — Audio file I/O

### Model Versions
- **XTTS-v2** — Zero-shot multilingual TTS from Coqui
- **CSM-1B** — Lightweight transformer-based TTS for fine-tuning

### Important Notes
- Notebooks pin specific versions (e.g., `torch==2.5.1`) for reproducibility
- Adjust versions if using different CUDA versions or non-NVIDIA GPUs
- On Windows with fresh environments, you may need to explicitly install `scipy` and `networkx` for `scikit-fuzzy`

## Performance & Quality

- **Single-shot synthesis** handles most use cases efficiently
- **Two-segment fallback** automatically engages for content exceeding single-model limits
- **Cross-fade joining** ensures imperceptible audio transitions between segments
- Typical latency: 10-30 seconds for 2-3 minutes of audio (GPU-dependent)

## Development

### Running Tests
```powershell
python smoke_test.py
```

### Extending the Pipeline
The modular design allows easy customization:

```python
from xtts_pipeline import clean_text, tts_single_shot

# Process and synthesize
clean = clean_text("Your text here...")
audio_path = tts_single_shot(clean, "reference.wav", "output.wav")
```

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions, issues, and feature requests are welcome! Please feel free to open a GitHub issue or submit a pull request.

## Acknowledgments

- [Coqui TTS](https://github.com/coqui-ai/TTS) — XTTS-v2 model and library
- [Unsloth](https://github.com/unslothai/unsloth) — Efficient fine-tuning framework
- [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/) — Training data
