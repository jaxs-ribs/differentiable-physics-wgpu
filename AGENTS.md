# AGENTS.md - Development History & Strategic Context

**SUMMARY:** Physics engine works (630M body-steps/s achieved!) but wireframe viz only shows background. Fixed matrix math, consolidated scripts to 3. ENTRYPOINT: Debug why wireframes aren't rendering despite correct vertex generation.

This document contains the development history and, more importantly, the strategic and technical context for AI development assistants. It is the single source of truth for the project's goals, constraints, and multi-phase roadmap.

## 1. Core Mission & Philosophy

**Mission:** Build a WebGPU-first, batch-differentiable rigid-body simulator that can run tens of thousands of bodies entirely on the GPU. The end goal is to plug this engine into `tinygrad` to create a platform for novel research in evolutionary robotics and physically-grounded AI.

**Philosophy & Ground Rules:**
- **Tests First, Perf Second:** Behavior is locked with a NumPy oracle and an exhaustive `pytest`/`Hypothesis` test suite before any micro-optimization.
- **Single Source of Truth:** One SoA buffer for all body states, padded to 16 bytes. The Rust `#[repr(C)]` struct must always exactly match the WGSL `struct Body`.
- **No Hidden Host Hops:** Every feature must be proven to run entirely on the GPU until a queue submission. No implicit data transfers.
- **Strict Cost Ceiling:** This project operates on a sabbatical budget. Development relies on local consumer GPUs. Cloud compute (e.g., A100 bursts) is reserved for final, high-impact paper-related sweeps and must be explicitly justified, staying under a total budget of $200/month.

## 2. The Multi-Phase Master Plan

This roadmap outlines the concrete, time-boxed deliverables from the initial engine to a shippable, research-grade showcase.

### Phase 1: Stand-alone Engine (COMPLETE)
*   **Goal:** A high-performance, stand-alone Rust + WGSL demo. `tinygrad` is not touched.
*   **Success Criteria:** All tests pass, performance exceeds 10,000 body-steps/s, and WGSL kernels are stable and portable.
*   **Key Deliverables:** `physics_core` binary, comprehensive test suite, wireframe visualization, and validated performance benchmarks.

---

### Phase 2: "Hello tinygrad" Smoke Test (Next Week)
1.  **Expose WGSL in tinygrad:** Patch the `tinygrad` WebGPU runtime to accept a `"WGSL_RAW"` tuple (`src`, `name`, `wgsize`). This should be a minimal, non-invasive change (~70 LOC).
2.  **Register Custom Ops:** Register `PHYSICS_STEP` and `CONTACT_SOLVE` as custom operations.
3.  **Python Shim:** Create a `tinyphys.step(state_tensor, dt)` that dispatches the raw WGSL op and returns the updated state as a new tensor.
4.  **Sanity RL Demo:** Implement a simple cart-pole-style balancing task. Use a built-in `tinygrad` optimizer (e.g., RMSProp) to learn a linear policy. The goal is to create a smoke test to prove gradients are flowing through the system.
5.  **Milestone:** `v0.2.0` (“autograd path live”).

---

### Phase 3: Differentiable Evolution Loop (Next Month)
1.  **Analytic Jacobians:** Implement the backward pass for the integrator and contact solver directly in WGSL. Expose this as an `adjoint_step` kernel.
2.  **Gradient Validation:** Validate the analytic gradients against finite-difference approximations across at least 100 randomized scenes.
3.  **DreamerV3 Integration:** Swap raw pixel observations for the latent state vector `z_t` from a pre-trained DreamerV3 world model. This treats the physics engine as a component within a larger learned model.
4.  **Quality-Diversity (QD) Loop:**
    *   **Outer Loop (CPU):** Mutate a morphology genotype → upload the new `Body` layout to the GPU.
    *   **Inner Loop (GPU):** Run N simulation rollouts.
    *   **Novelty Metric (GPU):** Compute novelty as `‖z_t – μ_seen‖₂` on-GPU.
    *   **Archive (CPU):** Log `(reward, novelty)` pairs and maintain a Hall-of-Fame archive of morphologies based on non-domination.
5.  **Milestone:** `v0.3.0` (“differentiable evolution loop”).

---

### Phase 4: Live Control & Telemetry (3 Months)
1.  **Telemetry Streamer:** Implement a Rust WebSocket server to stream simulation telemetry (step, energy, contact counts) as JSON blobs.
2.  **Minimal Web Dashboard:** Create a lightweight web interface (React or HTMX) to visualize the live telemetry.
3.  **LLM Scaffolding:** Allow an LLM to subscribe to the telemetry stream and issue commands (`pause`, `mutate`, `resume`). If the LLM detects an invariant violation (e.g., energy explosion), it should be able to auto-generate a skeleton for a new unit test capturing the failure case.
4.  **Milestone:** `v0.5.0` (“live control center”).

---

### Phase 5: Flagship Demo & Paper (6 Months)
1.  **Public Showcase:** Create a livestream or interactive web demo of the "ideal body" evolution, showing novel morphologies emerging in real-time.
2.  **Publishable Artifacts:** Prepare an 8-page arXiv draft detailing the methodology, performance, and emergent results.
3.  **One-Click Deploy:** Package the demo in a Docker image with a minimal WebUI for public release.
4.  **Handoff & Future Work:** Tag `v0.9.0`. Roadmap soft-body physics, differentiable fluids, and on-GPU rendering.

## 3. Phase 1 Specification (For Posterity)

*This was the original specification used to guide Phase 1 development.*

- **Mission:** Build a stand-alone Rust + WGSL demo.
- **Success Criteria:**
    1. All automated tests pass on CI.
    2. `cargo run --release --bin benchmark` reports ≥ 10,000 bodies × steps / s.
    3. WGSL kernels in `src/shaders/*.wgsl` expose a stable `physics_step` signature.
    4. Moving WGSL strings to tinygrad custom ops is expected to need < 1 day of glue work.
- **Implementation Plan:**
    - Step 0: Project scaffolding.
    - Step 1: CPU golden reference in Python.
    - Step 2: Data layout lock-in.
    - Step 3: Kernel TDD loop (Integrator, Narrow, Contact, Broad).
    - Step 4: Benchmarking and optimization.
    - Step 5: Minimal wire-frame debug view.
- **Test Suite:**
    - Unit maths, golden single step, golden 1,000-step free-fall, property fuzzing, stability stress test, performance guardrail.

## 4. Quick Development Reference

### Essential Scripts (Phase 1 Complete)
```bash
./pc-test    # Run all tests (unit, GPU, Python, fuzz)
./pc-bench   # Run benchmarks (add body count as arg)
./pc-demo    # Run visualization (add: simple, ascii, viz)
```

### Common Tasks
```bash
# Quick validation
./pc-test && ./pc-bench 10000

# See the engine in action
./pc-demo viz

# Full benchmark suite
cargo run --release --bin benchmark_full

# Specific Python tests
python3 tests/test_energy.py
python3 tests/test_sdf_fuzz.py
```

## 5. Current Status & Technical Details (Phase 1 Complete)

### Performance Achieved
- **~630M body×steps/s** with 20,000 bodies.
- Memory: 112 bytes per body (GPU-optimized alignment).

### Body Structure (112 bytes, 16-byte aligned)
```rust
struct Body {
    position: [f32; 4],      // xyz + padding
    velocity: [f32; 4],      // xyz + padding
    orientation: [f32; 4],   // quaternion
    angular_vel: [f32; 4],   // xyz + padding
    mass_data: [f32; 4],     // mass, inv_mass, padding
    shape_data: [u32; 4],    // type, flags (static=1), padding
    shape_params: [f32; 4],  // radius/half_extents
}
```

### Recently Fixed Issues
- ✅ `SimParams` uniform buffer fixed (proper WGSL alignment).
- ✅ All bodies now animate correctly (removed hardcoded shader limits).
- ✅ Wireframe visualization implemented.
- ✅ Comprehensive test suite implemented and passing.
- ✅ Unified CLI implemented.
- ✅ Matrix transformations fixed (row-major to column-major for GPU).
- ✅ Shell scripts consolidated to 3 simple scripts (pc-test, pc-bench, pc-demo).

### Current Issue - Wireframe Not Visible
**ENTRYPOINT:** The wireframe visualization shows only background color despite physics working correctly.

**Status:** Physics simulation runs (bodies fall), 288 vertices generated for 12 bodies, but wireframes not rendering.

**Debug Info:**
- Vertex data is being generated (24 vertices per AABB)
- Matrix math fixed with transpose for GPU column-major format
- Camera at (0, 20, 50) looking at (0, 5, 0)
- Pipeline configured for LineList topology
- Colors set (green=dynamic, gray=static)

**Next Steps:**
1. Verify vertex positions are in reasonable world space coordinates
2. Check if view-projection matrix correctly transforms to NDC [-1,1]
3. Test with simpler scene (single box at origin)
4. Add depth buffer to pipeline (currently None)
5. Check if lines are being culled or clipped