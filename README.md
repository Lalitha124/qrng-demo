# Quantum Random Number Generator (QRNG) — Gradio Demo

Author: Lalitha Lakkaraju

This repository contains a Gradio app that simulates / runs a QRNG (uses Qiskit Aer if available) and performs basic statistical analysis (chi-square, Shannon entropy). It is intended to be deployed on Hugging Face Spaces.

## Files
- `app.py` — Gradio app and QRNG logic
- `requirements.txt` — Python dependencies

## Deploy to Hugging Face Spaces (from this GitHub repo)
1. Create a Hugging Face account (https://huggingface.co).
2. Go to **Spaces → Create new Space**.
   - Choose **Gradio** as the SDK.
   - Under "Repository", select **Use a public Git repository** and paste:
     `https://github.com/<your-username>/<this-repo-name>`
3. The Space will build and host the app. You will get a permanent public URL you can embed in your PPT.

## Quick local test
```bash
python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py