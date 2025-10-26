# Quantum Random Number Generator (QRNG)

This project implements a Quantum Random Number Generator using Qiskit and Gradio. It leverages quantum superposition to produce unbiased random numbers and includes built-in tools for analyzing the randomness statistically through Chi-square tests and Shannon entropy.

## Features

- Quantum circuit construction and calibration
- Random number generation based on quantum measurement outcomes
- Error mitigation through inverse calibration matrices
- Statistical analysis using Chi-square and Shannon entropy
- Interactive Gradio interface for user control

## Functionality

The project generates random numbers using quantum circuits simulated via Qiskit’s Aer backend. If the AerSimulator is unavailable, it falls back to a classical uniform simulator to ensure consistent functionality.

## Files in Repository

| File | Description |
|------|-------------|
| app.py | Main application file containing the QRNG logic and Gradio interface |
| requirements.txt | List of Python dependencies required to run the application |
| presentation.pptx | Project presentation slides (included in the repository) |
| demo_video.mp4 | Project video demonstration (included in the repository) |

## Requirements

- gradio>=3.34
- qiskit==0.46.3
- qiskit-aer==0.12.0
- numpy
- scipy
- matplotlib

## Additional Resources

- PowerPoint presentation explaining the project’s background, methodology, and results.
- Video presentation demonstrating the application’s functionality and results.

## Online Demo

A live interactive demo of this QRNG project is hosted on Hugging Face Spaces:

https://huggingface.co/spaces/lvsl/qrng-demo