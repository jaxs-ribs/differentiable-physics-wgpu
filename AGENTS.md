# AGENTS: The Guiding Philosophy and Roadmap

## I. Vision: The End Game

Our singular objective is to build a high-performance, batch-differentiable, rigid-body physics engine, architected from the ground up on the `tinygrad` framework. This engine is designed for modern machine learning research, with **WebGPU** as its primary compute target to ensure maximum portability and enable complex simulations directly in the browser.

The long-term vision is to create an open, `Brax`-like ecosystem for the web. We are building towards a future where researchers and hobbyists can define, train, and evolve complex agents using models like **DreamerV3**. The engine's differentiability is key to this, enabling gradient-based optimization of not just policies, but entire morphologies. We will explore principles of **Automated Search for Artificial Life (ASAL)**, using world-model novelty as a driving force for open-ended evolution.

Users will be able to fork, experiment, and evolve creations entirely in the browser, creating a collaborative, web-native platform for the next generation of AI research.

---

## II. The Core Principles (The Non-Negotiables)

To achieve our vision, we adhere to a set of strict, non-negotiable engineering principles. These are the constitution of our project; all contributions must follow them without exception.

1.  **Differentiable End-to-End:** Every operation within the core physics step must be differentiable through `tinygrad`'s autograd system. There are no exceptions. This is the bedrock of our research goals.

2.  **The `tinyjit` Contract:** The entire physics step must be contained within a single function that can be decorated with `tinygrad`'s `@tinyjit`. This is our "One-Kernel Philosophy," ensuring that `tinygrad` can fuse the entire physics pipeline into a single, high-performance GPU kernel. Adherence to this contract is mandatory.
    *   **The Oracle:** Our Python implementation serves as the "oracle"—the ground truth for correctness. It must be written using **only** pure `tinygrad` tensor operations.
    *   **Forbidden Patterns:** The following are strictly forbidden within the jitted physics step, as they break the JIT compiler:
        *   **No External Libraries:** No `NumPy`, `SciPy`, or any other library. Only `tinygrad.tensor` operations are permitted.
        *   **No Data-Dependent Control Flow:** No `if/else` statements that branch on the *value* of a tensor (e.g., `if tensor.item() > 0:`).
        *   **No Data-Dependent Loops:** Loops must have a fixed, constant number of iterations (e.g., `for i in range(8):`). The loop count cannot be a tensor.
    *   **The Kernel:** The eventual WebGPU kernel is the production implementation. Its forward pass and gradients will be rigorously validated against the Python oracle to a tolerance of `1e-6`.

3.  **Rigorous Test-Driven Development (TDD):** The test suite is our definition of correctness and our roadmap. No feature is considered complete until its corresponding high-signal test case passes.

4.  **Extreme Observability:** We build systems we can see. Telemetry, tracing, and visualization are not afterthoughts; they are first-class citizens in the design process.

---

## III. Roadmap & Validation Suite

Our development is structured as a series of milestones, each validated by a specific, high-signal test scene. A milestone is complete only when all its associated tests pass.

#### Milestone 1: Single Body Dynamics & Contact
*   **1. Sphere Drop on Plane:** Stresses gravity, contact generation, and basic XPBD math. *Assert: Max penetration depth ≤ 0.5 mm after 1 s.*
*   **2. Elastic Bounce (restitution = 0.8):** Stresses the restitution term and velocity reconciliation. *Assert: Second apex height within ±2% of analytic `h·0.8²`.*
*   **3. Cube Landing on Corner:** Stresses inertia tensor handling and box contact manifold generation. *Assert: Cube settles upright (no spin) and COM drift < 1 mm.*

#### Milestone 2: Stability and Multi-Body Systems
*   **4. Ten-Cube Stack:** Stresses solver stability under load and broad-phase hashing saturation. *Assert: Tallest cube vertical drift < 3 mm over 10 s.*

#### Milestone 3: Joints and Articulated Bodies
*   **5. Hinge Pendulum:** Stresses the joint XPBD Jacobian and angle integration. *Assert: Hinge length error < 0.5 mm; period matches `2π√(L/g)` within 2%.*
*   **6. Double Pendulum (Chaotic):** Stresses joint stacking, quaternion updates, and energy conservation. *Assert: Total mechanical energy change < 1% over 30 s with damping = 0.*
*   **7. Motor-Driven Upright (Single Hinge):** Stresses the actuator path from action to torque to constraint. *Assert: Pendulum reaches ±5° of target in < 2 s without oscillation > 10°.*

#### Milestone 4: Advanced Physics & Interactions
*   **8. Slope Slide with Friction (μ=0.5):** Stresses contact tangent projection and the friction cone clamp. *Assert: Acceleration down slope ≈ `g·(sinθ − μ·cosθ)` within 5%.*
*   **9. Ragdoll Drop (Five linked boxes):** Stresses the system with many simultaneous joints and contacts. *Assert: No NaNs, joint errors < 2 mm, simulation survives 60 s.*
*   **10. Tendon-Spring Oscillator:** Stresses compliant XPBD and stiffness-vs-dt independence. *Assert: Oscillation period is independent of solver iteration count (1, 4, 8 iters differ < 1%).*
*   **11. Hill Muscle Kick (Planar Leg):** Stresses activation dynamics and variable rest lengths for muscle models. *Assert: Knee extends to 90° when `a=1`; gradient `∂θ/∂a ≠ 0`.*

#### Milestone 5: Performance & Final Validation
*   **12. Batch Regression (256 Walkers):** Stresses memory layout, the final WGSL one-kernel path, and throughput. *Assert: `steps/s ≥ target`; `max|state_gpu − state_py| < 1e-5`.*

---

## IV. Technical Specification: The XPBD Pipeline

The engine implements an Extended Position-Based Dynamics (XPBD) solver. The following sequence of operations is the ground truth for our implementation, from the Python oracle to the final WGSL kernel.

*   **0. Inputs (State Before Step)**
    *   **Body State (SoA):** `x [B,N,3]`, `q [B,N,4]`, `v [B,N,3]`, `ω [B,N,3]`
    *   **Body Constants:** `inv_mass [N]`, `inv_inertia [N,3,3]`, shape parameters
    *   **Joints & Actions:** Joint definitions, motor strengths, action tensors `τ_action`

*   **1. Forward Prediction**
    *   Apply external forces (gravity, motors) to get `a` and `τ`.
    *   `v ← v + a·dt`
    *   `ω ← ω + I⁻¹·(τ − ω×Iω)·dt` (includes gyroscopic term)
    *   `x_pred ← x + v·dt`
    *   `q_pred ← normalize( q ⊗ exp(½·ω·dt) )`

*   **2. Collision Detection**
    *   **Broadphase:** Uniform spatial hash on `x_pred` to find candidate pairs.
    *   **Narrowphase:** Analytic overlap tests (sphere-plane, sphere-sphere, box-plane) on candidate pairs to generate contacts.
    *   **Output:** Tensors for contact points: `ids_a [K]`, `ids_b [K]`, normal `n [K,3]`, soft penetration `p [K]`, compliance `α`.

*   **3. Constraint Assembly**
    *   Build a single, unified list of all XPBD constraints (contacts, joints, etc.). Each entry contains body indices, Jacobians, and compliance `α`.

*   **4. XPBD Solver Loop (e.g., 8 Jacobi Iterations)**
    *   For each constraint, evaluate violation `C(x_pred)`.
    *   Compute Lagrange multiplier update: `Δλ = −C / (α/dt² + M̃)`, where `M̃` is the effective mass.
    *   Accumulate position/orientation corrections (`Δx_buf`, `Δq_buf`) from `Δλ`.
    *   After loop: `x_proj = x_pred + Δx_buf`, `q_proj = normalize(q_pred + Δq_buf)`.

*   **5. Velocity Reconciliation**
    *   `v_new ← (x_proj − x) / dt`
    *   `ω_new ← 2·log(q⁻¹ ⊗ q_proj)/dt`
    *   Optionally apply damping.

*   **6. State Commit & Output**
    *   Update main state: `x ← x_proj`, `q ← q_proj`, `v ← v_new`, `ω ← ω_new`.
    *   Return observation dictionary for RL loops, retaining the autograd tape.

---

## V. Observability Hooks

These tools are first-class components of the engine, essential for development and debugging.

*   **Per-Step Stats Tensor**: A tensor updated each frame containing: `penetration_max`, `joint_error_max`, `energy`, `λ_rms`, `pairs_count`. Batched for histogram plotting.
*   **Gradient Checker**: A CI utility that uses finite-difference to verify autograd gradients for key parameters.
*   **Replay-to-Video Helper**: A script to dump simulation state and invoke the renderer, crucial for visualizing changes.
*   **Performance Counters**: Wall-clock ms per pipeline stage (integrate, broad, narrow, solve) logged to CI to catch performance regressions.

Also: No more docstrings! They are useless! Also, no comments! Code should speak for itself!