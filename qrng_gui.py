# Quantum Random Number Generator (QRNG)
# Hackathon Submission
# Author: Lalitha Lakkaraju

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import numpy as np
from scipy.stats import chisquare, entropy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# Calibration Circuit Generator
def make_calibration_circuits(n_qubits):
    circs, labels = [], []
    for i in range(2 ** n_qubits):
        bits = format(i, f"0{n_qubits}b")
        qc = QuantumCircuit(n_qubits, n_qubits)
        for q, b in enumerate(reversed(bits)):
            if b == "1":
                qc.x(q)
        qc.measure(range(n_qubits), range(n_qubits))
        circs.append(qc)
        labels.append(bits)
    return circs, labels

# Quantum Random Number Generator with Statistical Analysis
def qrng_with_analysis(n_qubits=3, shots=2048, num_samples=5):
    backend = AerSimulator()

    # Calibration
    cal_circs, labels = make_calibration_circuits(n_qubits)
    cal_job = backend.run(transpile(cal_circs, backend), shots=max(256, int(shots/4)))
    cal_result = cal_job.result()

    n_states = 2 ** n_qubits
    M = np.zeros((n_states, n_states))
    for i, label in enumerate(labels):
        counts = cal_result.get_counts(i)
        for j in range(n_states):
            M[i, j] = counts.get(labels[j], 0)
        M[i] /= M[i].sum() if M[i].sum() > 0 else 1
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        M_inv = np.linalg.pinv(M)

    # Quantum Circuit
    qc = QuantumCircuit(n_qubits, n_qubits)
    for q in range(n_qubits):
        qc.h(q)
    qc.measure(range(n_qubits), range(n_qubits))
    res = backend.run(transpile(qc, backend), shots=shots).result()
    raw_counts = res.get_counts(0)

    state_labels = [format(i, f"0{n_qubits}b") for i in range(n_states)]
    raw_probs = np.array([raw_counts.get(lbl, 0) for lbl in state_labels], dtype=float)
    raw_probs /= raw_probs.sum()

    mitigated_probs = M_inv.dot(raw_probs)
    mitigated_probs = np.maximum(mitigated_probs, 0)
    mitigated_probs /= mitigated_probs.sum()

    mitigated_counts = {state_labels[i]: int(mitigated_probs[i] * shots)
                        for i in range(n_states) if mitigated_probs[i] > 0}

    # Generate Random Numbers
    samples = np.random.choice(state_labels, size=num_samples, p=mitigated_probs)
    random_numbers = [int(s, 2) for s in samples]

    # Statistical Tests
    expected = np.full(n_states, shots / n_states)
    chi_stat, p_value = chisquare([mitigated_counts.get(lbl, 0) for lbl in state_labels],
                                  f_exp=expected)
    shannon_entropy = entropy(mitigated_probs, base=2)
    max_entropy = n_qubits

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_histogram(mitigated_counts, ax=ax, title="Quantum Measurement Distribution")
    plt.tight_layout()

    result = (
        f"Quantum Random Numbers: {random_numbers}\n\n"
        f"Chi-square: {chi_stat:.3f}\n"
        f"p-value: {p_value:.4f}\n"
        f"Shannon Entropy: {shannon_entropy:.3f} / {max_entropy:.3f} bits\n\n"
    )
    result += "Statistically uniform." if p_value > 0.05 else "Bias detected."
    return result, fig

# --- Tkinter UI ---
def run_qrng():
    n_qubits = int(n_qubits_var.get())
    shots = int(shots_var.get())
    samples = int(samples_var.get())

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Generating... please wait...\n")
    root.update_idletasks()

    result, fig = qrng_with_analysis(n_qubits, shots, samples)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, result)

    for widget in frame_plot.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    canvas.draw()
    canvas.get_tk_widget().pack()

root = tk.Tk()
root.title("Quantum Random Number Generator")

tk.Label(root, text="Qubits:").grid(row=0, column=0)
n_qubits_var = tk.StringVar(value="3")
tk.Entry(root, textvariable=n_qubits_var, width=5).grid(row=0, column=1)

tk.Label(root, text="Shots:").grid(row=0, column=2)
shots_var = tk.StringVar(value="2048")
tk.Entry(root, textvariable=shots_var, width=8).grid(row=0, column=3)

tk.Label(root, text="Samples:").grid(row=0, column=4)
samples_var = tk.StringVar(value="5")
tk.Entry(root, textvariable=samples_var, width=5).grid(row=0, column=5)

tk.Button(root, text="Generate", command=run_qrng, bg="#4CAF50", fg="white").grid(row=0, column=6, padx=10)

result_text = tk.Text(root, height=10, width=80)
result_text.grid(row=1, column=0, columnspan=7, pady=10)

frame_plot = tk.Frame(root)
frame_plot.grid(row=2, column=0, columnspan=7)

root.mainloop()
