# Project Phoenix: The XPBD Physics Engine

## I. Core Mission & Philosophy

Our objective is to build a high-performance, batch-differentiable rigid-body physics engine, "Phoenix," deeply integrated into the `tinygrad` ecosystem. This engine is designed for modern machine learning research, with a primary focus on WebGPU for maximum portability.

Our engineering philosophy is defined by three pillars:
1.  **Extreme Observability:** We build systems we can see, with telemetry and tracing as first-class citizens.
2.  **Rigorous Test-Driven Development (TDD):** The test suite is our definition of correctness. No feature is complete until its corresponding high-signal test case passes.
3.  **One-Kernel Philosophy:** The Python implementation (the "oracle") must be a pure tensor-based function that directly maps to a single, fused WGSL compute kernel later. This ensures performance is designed in from day one.

---

## II. The XPBD Pipeline: Technical Specification

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

## III. Validation Suite: High-Signal Test Scenes

This suite defines the validation benchmarks for the engine. Each milestone must pass all preceding tests.

1.  **Sphere Drop on Plane**
    *   *Stresses*: Gravity, contact generation, basic XPBD math.
    *   *Assert*: Max penetration depth ≤ 0.5 mm after 1 s.

2.  **Elastic Bounce** (restitution = 0.8)
    *   *Stresses*: Restitution term, velocity reconciliation.
    *   *Assert*: Second apex height within ±2% of analytic `h·0.8²`.

3.  **Cube Landing on Corner**
    *   *Stresses*: Inertia tensor handling, box contact manifold.
    *   *Assert*: Cube settles upright (no spin) and COM drift < 1 mm.

4.  **Ten-Cube Stack**
    *   *Stresses*: Solver stability, broad-phase hashing saturation.
    *   *Assert*: Tallest cube vertical drift < 3 mm over 10 s.

5.  **Hinge Pendulum**
    *   *Stresses*: Joint XPBD Jacobian, angle integration.
    *   *Assert*: Hinge length error < 0.5 mm; period matches `2π√(L/g)` within 2%.

6.  **Double Pendulum (Chaotic)**
    *   *Stresses*: Joint stacking, quaternion update, energy conservation.
    *   *Assert*: Total mechanical energy change < 1% over 30 s with damping = 0.

7.  **Motor-Driven Upright** (Single Hinge)
    *   *Stresses*: Actuator path, action→torque→constraint flow.
    *   *Assert*: Pendulum reaches ±5° of target in < 2 s without oscillation > 10°.

8.  **Slope Slide with Friction** (μ=0.5)
    *   *Stresses*: Contact tangent projection, friction clamp.
    *   *Assert*: Acceleration down slope ≈ `g·(sinθ − μ·cosθ)` within 5%.

9.  **Ragdoll Drop** (Five linked boxes)
    *   *Stresses*: Many joints + contacts + cross-body inertia.
    *   *Assert*: No NaNs, joint errors < 2 mm, simulation survives 60 s.

10. **Tendon-Spring Oscillator**
    *   *Stresses*: Compliant XPBD, stiffness-vs-dt independence.
    *   *Assert*: Oscillation period independent of solver iteration count (1, 4, 8 iters differ < 1%).

11. **Hill Muscle Kick** (Planar Leg)
    *   *Stresses*: Activation dynamics, variable rest length.
    *   *Assert*: Knee extends to 90° when `a=1`; gradient `∂θ/∂a ≠ 0`.

12. **Batch Regression** (256 Walkers)
    *   *Stresses*: Memory layout, WGSL one-kernel path, throughput.
    *   *Assert*: `steps/s ≥ target`; `max|state_gpu − state_py| < 1e-5`.

---

## IV. Observability Hooks

These tools are first-class components of the engine, essential for development and debugging.

*   **Per-Step Stats Tensor**: A tensor updated each frame containing: `penetration_max`, `joint_error_max`, `energy`, `λ_rms`, `pairs_count`. Batched for histogram plotting.
*   **Gradient Checker**: A CI utility that uses finite-difference to verify autograd gradients for key parameters.
*   **Replay-to-Video Helper**: A script to dump simulation state and invoke the renderer, crucial for visualizing changes.
*   **Performance Counters**: Wall-clock ms per pipeline stage (integrate, broad, narrow, solve) logged to CI to catch performance regressions.

Also: No more docstrings! They are useless! Also, no comments! Code should speak for itself!