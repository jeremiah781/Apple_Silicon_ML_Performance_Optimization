# Apple Silicon ML Performance Optimization

A systematic investigation into optimizing Stable Diffusion inference on Apple Silicon, achieving a **35% latency reduction** through advanced profiling, asynchronous execution, and hardware-aware techniques.

## The Problem

Generative AI models like Stable Diffusion are computationally expensive. While powerful, deploying them on consumer-grade edge devices like an Apple M1 Mac presents significant challenges in latency and resource utilization. The standard PyTorch implementation often underutilizes the M1's GPU (via the Metal Performance Shaders backend), leading to slow inference times that are impractical for real-time applications. This project tackles that problem head-on.

## My Solution & Technical Approach

This project systematically identifies and resolves these bottlenecks by treating the entire inference pipeline as a system to be optimized. The goal was not just to run a model, but to make it run efficiently on specific, constrained hardware.

My approach involved several key actions:

*   **Systematic Profiling:** I used a combination of `Py-Spy`, `cProfile`, and macOS Instruments to conduct a deep analysis of the entire inference pipeline, identifying specific function calls and memory transfer operations that were creating bottlenecks.

*   **Hardware-Aware Optimization:** I engineered solutions that specifically leverage the capabilities of the M1 architecture:
    *   **Mixed Precision (FP16):** Implemented `torch.amp` to use half-precision floating-point numbers, reducing memory footprint and leveraging the M1's hardware acceleration for FP16 operations.
    *   **Operator Fusion:** Utilized PyTorch's JIT compiler (`torch.jit.script`) to fuse multiple adjacent operations into a single kernel, significantly reducing the overhead of launching multiple small computations on the GPU.

*   **Asynchronous Execution & Scheduling:** I orchestrated an adaptive scheduling system using Python's `asyncio` library to run data preprocessing on the CPU concurrently with model inference on the GPU, minimizing idle time for both processors.

*   **Automated Hyperparameter Tuning:** I integrated `Optuna` to automate the process of finding the optimal inference steps and guidance scale, balancing output quality with raw performance.

## Key Results & Quantified Impact

This multi-faceted optimization strategy yielded significant, measurable improvements:

*   **35% Reduction in End-to-End Inference Latency:** The primary goal was achieved, reducing the time to generate an image from a baseline of ~95 seconds to ~62 seconds.

*   **Improved Hardware Utilization:** Profiling confirmed that these techniques improved the distribution of work between the CPU and GPU, moving from a state of GPU underutilization to more parallel and efficient processing.

*   **Developed a Reusable Framework:** The code serves as a robust framework for profiling and optimizing other complex PyTorch models for deployment on Apple Silicon and other edge devices.

## Core Technologies & Tools

*   **Language:** Python
*   **ML Framework:** PyTorch, Hugging Face Diffusers
*   **Hardware Backend:** Apple Metal Performance Shaders (MPS)
*   **Optimization & Profiling:** Optuna, NumPy, cProfile, Py-Spy
*   **Developer Tools:** Git, Conda, Docker

## How to Run

1.  Clone the repository:
    ```sh
    git clone https://github.com/jeremiah781/Apple-Silicon-ML-Performance-Optimization.git
    ```
2.  Navigate to the project directory:
    ```sh
    cd Apple-Silicon-ML-Performance-Optimization
    ```
3.  Set up the environment using Conda:
    ```sh
    conda env create -f environment.yml
    ```
4.  Activate the environment:
    ```sh
    conda activate ml_optimization
    ```
5.  Run the main script:
    ```sh
    python optimize_inference.py
    ``` reducing memory footprint and leveraging the M1's hardware acceleration for FP16 operations.
