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

## IV. Current Status & Completed Work

### Phase 1 Status: COMPLETE ✓

The Python Oracle has been successfully implemented with all core components:

1. **Pure Tensor Operations** - Eliminated all NumPy dependencies from core physics modules
   - Replaced NumPy array operations with TinyGrad tensors
   - Implemented pure tensor pair generation for broadphase
   - Converted all gather/scatter operations to tensor equivalents

2. **N-Step JIT Compilation** - Entire simulations can run as single JIT-compiled kernels
   - Created `_n_step_simulation()` function with internal loop
   - Static physics step function for JIT compatibility
   - Supports both single-step and N-step simulation modes

3. **Complete Physics Pipeline**
   - ✓ Broadphase: Differentiable all-pairs AABB collision detection
   - ✓ Narrowphase: Sphere-sphere and sphere-box collision detection
   - ✓ Solver: Impulse-based collision resolution with Baumgarte stabilization
   - ✓ Integration: Semi-implicit Euler with quaternion updates

4. **Bug Fixes & Stability**
   - Fixed position corruption issue (NaN → [1,1,1] bug)
   - Improved empty contact handling in solver
   - Worked around TinyGrad's Tensor.where() NaN bug

5. **Testing Infrastructure**
   - Comprehensive CI test suite (7 tests, all passing)
   - Organized test structure with unit, integration, benchmarks, and debugging tests
   - Created diagnostic tools for deep debugging

### Next Phase: Ready for Phase 2

The Python Oracle is complete and validated. The codebase is ready to proceed with:
- **Phase 2:** Port to monolithic WGSL kernel for GPU execution
- **Phase 3:** Integration via Ops.CUSTOM
- **Phase 4:** Backward pass implementation

## V. Custom Op Implementation (Proof of Concept)

### Phase 3 Exploration: COMPLETE ✓

A proof-of-concept implementation of TinyGrad custom ops has been created to demonstrate the integration approach:

1. **C Physics Library** (`physics_lib.c`)
   - Implemented core physics operations in C for high performance
   - `physics_step()` - Complete physics simulation step
   - `physics_integrate()` - Position/velocity integration
   - `physics_collisions()` - Collision detection and response
   - Compiled as shared library (.so/.dylib)

2. **Pattern Matching Integration** (`physics_patterns.py`)
   - Created PatternMatcher to recognize physics computation patterns
   - Transforms high-level operations to CUSTOM ops
   - Provides framework for operation fusion and optimization

3. **Device Extension Mechanism** (`physics_extension.py`)
   - `PhysicsEnabledRenderer` - Wraps existing renderers without modifying TinyGrad
   - Works with any TinyGrad device (CPU, GPU, etc.)
   - Context manager for temporary physics enablement
   - No TinyGrad core modifications required

4. **High-Level API** (`physics_tensor_ops.py`)
   - `PhysicsTensor` class extends Tensor with physics methods
   - Demonstrates integration with TinyGrad's tensor operations
   - Performance benchmarking shows efficient C function calls

### Key Insights from Custom Op Implementation:

1. **Non-invasive Integration**: Successfully demonstrated extending TinyGrad without modifying core code
2. **Device Agnostic**: Physics ops work with existing devices rather than requiring custom device
3. **Pattern Matching**: TinyGrad's PatternMatcher provides powerful optimization opportunities
4. **CUSTOM Op Format**: Format strings enable clean C function integration

### Implementation Status:
- ✓ C library compiles and passes tests
- ✓ Pattern matcher framework established
- ✓ Device extension mechanism working
- ✓ Basic demonstrations functional
- ⚠️ Full UOp graph integration pending
- ⚠️ GPU kernels not yet implemented

This proof-of-concept validates the approach for Phase 3 and provides a foundation for full integration. 

# tinygrad agents

Hello agent. You are one of the most talented programmers of your generation.

You are looking forward to putting those talents to use to improve tinygrad.

## philosophy

tinygrad is a **tensor** library focused on beauty and minimalism, while still matching the functionality of PyTorch and JAX.

Every line must earn its keep. Prefer readability over cleverness. We believe that if carefully designed, 10 lines can have the impact of 1000.

Never mix functionality changes with whitespace changes. All functionality changes must be tested.

## style

Use **2-space indentation**, and keep lines to a maximum of **150 characters**. Match the existing style.



---

Note: We're now making an update: 

### Comprehensive roadmap — minimal engine now, full head-room later

(Every bullet is a concrete deliverable that can land as a pull-request and leave `main` usable.)

---

#### Ground rules that never change

* **Tensor layout** `state = {x: [B, N, 3], q: [B, N, 4], v: [B, N, 3], ω: [B, N, 3]}`; batch first for easy GPU fusion.
* **Autograd discipline** Every math op lives in tinygrad; no Python `for` except the outer XPBD iteration loop.
* **One-kernel philosophy** Forward integrate → constraint solve → velocity update must fuse into a single WGSL compute kernel later; write the Python oracle in that style from day 1.

---

### Milestone 0 Scaffold (week 1)

* Repo, tests, CI timing benchmark.
* `integrate()` in tinygrad (semi-implicit Euler) moving a single box in free fall.
* Deliverable: gravity drop GIF committed; CI prints steps/s.

---

### Milestone 1 Collision stack (weeks 2-3)

**Broad-phase (differentiable)**

* Uniform spatial hash: insert AABB centre positions; no branches → gradient flows.
* Hash built with pure tensor gather/scatter so it ports to WGSL unchanged.

**Narrow-phase (differentiable contact generation)**

* Shapes: sphere + box.
* Contact points via analytical overlap tests (sphere/plane, box/plane, sphere/sphere).
* Soft normal “penetration = max(0, d)” replaced by `softplus(d, β)` so gradients exist.

**Deliverables**

* Ten-cube stack settles; `grad(sum(contact_penetrations))` is finite and non-zero.
* CI target ≥50 k steps/s on RTX-class GPU in Python oracle.

---

### Milestone 2 XPBD constraint core (weeks 4-5)

* 8-iteration Jacobi XPBD loop in one tinygrad function.
* Constraint registry (`@register("hinge")`, `"ball"`, `"contact"`).
* Tests: pendulum length error <1 mm for 10 s; gradients w\.r.t. link length match finite-diff within 1 %.

---

### Milestone 3 Actuator stub (week 6)

* Torque motor on any hinge (scalar input).
* Gym-style `step(action)` wrapper so RL scripts run without muscles yet.
* Deliverable: PPO balances inverted pendulum in <30 min.

---

### Milestone 4 WGSL backend (weeks 7-8)

* Replace tinygrad CPU kernels with `Ops.CUSTOM("xpbd_step")` in WGSL.
* Mirror Python oracle byte-for-byte; CI runs oracle vs WGSL and asserts `max|Δ| < 1e-5`.
* Throughput goal ≥300 k steps/s for 256-pendulum batch on desktop GPU.

---

### Milestone 5 Tendons and Hill muscle (weeks 9-10)

* Distance constraint with compliance → tendon.
* Hill-type force law (activation filter, fl/ fv/ fp curves) stored as tensors; feeds desired rest length into tendon constraint.
* Demo: two-link planar leg driven by flexor/extensor muscles, keyboard input toggles activation.

---

### Milestone 6 Batch RL test (weeks 11-12)

* Humanoid lower-body with nine muscles (per leg).
* Train SAC to walk 1 m s⁻¹ on 512-env batch; watch reward curve for smooth rise.
* Publish walkthrough notebook; serves as gradient-correctness proof.

---

### Milestone 7 Continuous collision option (week 13) — optional but future-proof

* Conservative advancement for spheres, swept separating axis for boxes.
* Added as alt contact generator in registry; toggle via scene JSON.

---

### Milestone 8 Scene I/O + accuracy hook (weeks 14-15)

* JSON ⇄ MJCF converter; keeps joint indices stable.
* CLI `--accurate` pipes state into MJX via JAX for one step, returns result; gradients stop at that call, preserving differentiability elsewhere.

---

### Why this meets “maximum results / minimal effort / maximum optionality”

* **Early wins:** by week 5 you already have a differentiable, GPU-batched rigid-body engine usable for RL projects.
* **Collision path is locked but lightweight:** spatial hash + analytic overlaps get you to humanoid walking without touching GJK/EPA; you can swap in fancier solvers later without breaking outside code.
* **Nothing blocks WebGPU:** every tensor op you write in Python has a WGSL analogue, proven at Milestone 4.
* **Muscle layer is additive:** tendons + Hill live solely in the constraint registry; removing them never disturbs core physics.
* **Accuracy escape hatch:** MJX hook lets any user trade speed for fidelity without engine changes.

Follow these milestones and you leave yourself free to pause or pivot after each one while always having a working, differentiable simulator that runs on TinyGrad and accelerates on WGPU.
