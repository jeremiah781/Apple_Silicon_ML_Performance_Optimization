# Stable Diffusion **Edge Optimizer** for Apple Silicon

<!-- headline badges – scaled to 28 px tall -->
<p>
  <img src="https://img.shields.io/badge/Target-Apple%20M1-blue"                 height="28">
  <img src="https://img.shields.io/badge/Latency-Reduced%20by%2035%25-green"    height="28">
  <img src="https://img.shields.io/badge/Status-Ongoing%20Research-informational" height="28">
</p>


> **Result:** End-to-end inference latency for Stable Diffusion reduced from 110s to **71s** on a baseline Apple M1.

---

## The Mission: Democratizing Generative AI

Generative models like Stable Diffusion are GPU-hungry. On consumer hardware like an Apple M1, the standard PyTorch+MPS path is slow and leaves the GPU under-utilized. **This project's mission is to democratize AI by creating a framework that squeezes professional-grade performance out of widely available hardware.**

---

## Technical Approach

| Area | Techniques Applied |
| --- | --- |
| **Profiling Stack** | Systematic analysis using Py-Spy, `cProfile`, macOS Instruments, and MPS performance counters to identify and measure bottlenecks. |
| **Hardware Tuning** | `torch.amp` for **FP16 Mixed Precision** · JIT **Operator Fusion** with `torch.jit.script` · Asynchronous CPU/GPU overlap via `asyncio`. |
| **Automated Optimization**| `Optuna` hyperparameter search to find the optimal balance between inference steps, guidance scale, and performance. |
| **Reusable Framework** | Designed with clean, modular hooks for profiling, tuning, and experimenting on other edge devices. |

---

## Key Performance Metrics  (Apple M1 • 512×512 • 50 steps)

| Stage                        | Baseline | Optimized | Δ (improvement) |
|------------------------------|----------|-----------|-----------------|
| **Text encoding**            |   5.2 s  |   3.1 s   | -40 % |
| **UNet diffusion pass**      |  95.0 s  |  62.9 s   | -34 % |
| **VAE decoding**             |   8.0 s  |   4.0 s   | -50 % |
| **End-to-end latency**       | 110.0 s  | **71.0 s** | **-35 %** |
| **Avg GPU util.**            |   30 %   |   64 %    | +34 pp |


## Measured on: <!-- Tested-on badges -->
<p>
  <img src="https://img.shields.io/badge/macOS-14.4-black?logo=apple&logoColor=white"  height="28">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" height="28">
  <img src="https://img.shields.io/badge/PyTorch-2.3%20(MPS)-ee4c2c?logo=pytorch&logoColor=white" height="28">
</p

---

## Validation & Research Status

This work was selected for presentation as a research poster at **Penn State AI Week**. The project is ongoing and is currently being prepared for formal academic publication under the guidance of my faculty mentors.

[**View the Full Research Poster (PDF)**](https://github.com/jeremiah781/Apple_Silicon_ML_Performance_Optimization/blob/main/Research%20Poster)

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/jeremiah781/Apple-Silicon-ML-Performance-Optimization.git
cd Apple-Silicon-ML-Performance-Optimization

# 2. Create and activate the Conda environment
conda env create -f environment.yml
conda activate m1_sd_opt

# Add your Hugging Face token
cp .env.example .env                    # copy template
echo 'HF_TOKEN=YOUR_TOKEN_HERE' >> .env
source .env                             # or let your shell load it

# Install a secret-scanning pre-commit hook
pip install pre-commit
pre-commit install                      # enables detect-secrets on every commit


Optional M-series GPU utilisation
The scripts call powermetrics for real-time GPU %.
Run with sudo -E python stable_diffusion.py … if you want this metric.
Without sudo, the value is silently recorded as 0 %—everything else still works.

# 3. Run an optimized inference pass
python optimize_inference.py \
  --prompt "A photo of the Nittany Lion statue in a cyberpunk city" \
  --steps 50
