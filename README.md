
# Apple Silicon ML Performance Optimization

**35% faster inference on Stable Diffusion models through profiling, hardware-aware tuning, and async parallelism on Apple an M1 laptop.**

## The Problem

Generative models like Stable Diffusion are compute-heavy. On edge devices like Apple M1 Macs, PyTorch often underutilizes the GPU (via MPS), leading to slow inference times. This project tackles that head-on.

## Approach

I treated inference as a full-stack system optimization problem. Key steps:

- **Systematic Profiling:** Used Py-Spy, cProfile, and Instruments to identify bottlenecks in function calls and memory operations.
- **Mixed Precision (FP16):** Applied `torch.amp` to reduce memory usage and use M1’s FP16 acceleration.
- **Operator Fusion:** Used `torch.jit.script` to fuse adjacent ops into fewer GPU launches.
- **Async Scheduling:** Leveraged `asyncio` to overlap CPU preprocessing with GPU inference.
- **Auto-Tuning:** Integrated Optuna to tune steps and guidance scale for optimal latency vs quality.

## Results

- **35% latency reduction** (95s → 62s)
- Balanced CPU-GPU load, eliminating idle periods
- Built a reusable pipeline for optimizing PyTorch models on Apple Silicon

## Tools & Stack

- **Languages**: Python
- **Frameworks**: PyTorch, Hugging Face Diffusers
- **Hardware API**: Metal Performance Shaders (MPS)
- **Optimization**: Optuna, NumPy, Py-Spy, cProfile
- **Dev Tools**: Git, Docker, Conda

## Reproducing Results

```bash
git clone https://github.com/jeremiah781/Apple-Silicon-ML-Performance-Optimization.git
cd Apple-Silicon-ML-Performance-Optimization
conda env create -f environment.yml
conda activate ml_optimization
python optimize_inference.py
```

## Note

The project is ongoing. This poster, presented at Penn State AI Week, captures a meaningful milestone in its development.
