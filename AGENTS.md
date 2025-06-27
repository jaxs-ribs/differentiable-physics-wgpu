# Differentiable Physics in Tinygrad

## I. Core Mission

Our objective is to build a "Brax-like" rigid-body physics engine deeply integrated into the `tinygrad` ecosystem. This engine will be:

1.  **Massively Parallel:** Designed from the ground up to run entirely on a hardware accelerator (GPU, etc.) without host-CPU roundtrips during simulation rollouts. This enables the simulation of thousands of environments simultaneously.
2.  **Batch Differentiable:** The entire physics step will be differentiable end-to-end, allowing gradients to flow back through time and simulation states.
3.  **Multi-Backend:** By leveraging `tinygrad`'s infrastructure, we will target multiple backends, with a primary focus on WebGPU for maximum portability and reach.

The ultimate purpose of this project is to create a foundational tool for next-generation machine learning research, particularly in reinforcement learning, evolutionary robotics, and creating self-debugging, self-improving systems.

## II. Guiding Principles

This project will be executed with an unwavering commitment to quality and clarity. Our engineering philosophy is defined by three core pillars:

1.  **Extreme Observability:** We don't just write code; we build systems we can see. Every component, from the lowest-level kernel to the highest-level API, will be designed with telemetry, logging, and tracing as first-class citizens. The goal is to provide a "live control center" view of the system's state, enabling both human and AI agents to debug and analyze its behavior with high fidelity.

2.  **Rigorous Test-Driven Development (TDD):** No feature is considered complete until it is covered by a comprehensive suite of tests. We will write tests first to precisely define the behavior of a component, then write the code to make the tests pass. This disciplined approach ensures correctness at every stage, from the Python oracle to the final GPU kernels.

3.  **Clean Architecture & Readability:** We will adhere to the principles of Clean Architecture ("Uncle Bob" Martin) to create a system that is readable, maintainable, and minimizes cognitive entropy. The code will be self-documenting, the modules will have clear responsibilities, and the dependencies will be strictly managed. This ensures that the codebase remains a valuable and understandable asset for both human and AI developers.

## III. The Four-Phase Master Plan

We have established a clear, sequential roadmap to manage the complexity of this project.

-   **Phase 1: The Python Oracle (The "Golden Reference")**
    Create a 100% functionally correct physics engine using pure `tinygrad.Tensor` and `numpy` operations. It will be slow, but it will be our unimpeachable ground truth for validation.

-   **Phase 2: The Monolithic WGSL Kernel (The "GPU Workhorse")**
    Port the exact logic from the Python Oracle into a single, high-performance WGSL kernel. This kernel will execute the entire physics step on the GPU.

-   **Phase 3: The `Ops.CUSTOM` Bridge (The "Tinygrad Integration")**
    Hook our WGSL kernel into `tinygrad`'s computation graph via `Ops.CUSTOM`. This will make our engine a first-class, JIT-compilable citizen within the `tinygrad` ecosystem.

-   **Phase 4: The Backward Pass (The "Differentiability")**
    Implement the backward pass for our custom op, likely as a second, hand-written WGSL kernel. This will unlock end-to-end differentiability.

## IV. Current Status & Immediate Objective

-   **Current Phase:** Phase 1
-   **Immediate Goal:** Complete the `physics_engine.py` oracle. The next step is to implement the full collision pipeline (Broadphase, Narrowphase, and a basic impulse-based Solver) according to the detailed specification we have drafted. This will provide the validated, ground-truth implementation required for all subsequent work. 

# tinygrad agents

Hello agent. You are one of the most talented programmers of your generation.

You are looking forward to putting those talents to use to improve tinygrad.

## philosophy

tinygrad is a **tensor** library focused on beauty and minimalism, while still matching the functionality of PyTorch and JAX.

Every line must earn its keep. Prefer readability over cleverness. We believe that if carefully designed, 10 lines can have the impact of 1000.

Never mix functionality changes with whitespace changes. All functionality changes must be tested.

## style

Use **2-space indentation**, and keep lines to a maximum of **150 characters**. Match the existing style.