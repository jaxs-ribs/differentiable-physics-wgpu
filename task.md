# Master Plan: Unifying the Physics Engine Runner & CI

## I. Project Goal & Context

Our objective is to refactor the project's entry points to create a clean, unified, and extensible system for running and comparing different physics simulation backends. The current structure is fragmented, with logic scattered across shell scripts and example files. This plan will centralize control into two primary Python scripts:

1.  `scripts/run_simulation.py`: A powerful script to run any combination of our physics engines (Python Oracle, C-accelerated, and future WebGPU) and save their results.
2.  `run_ci.py`: A single Python script to handle all Continuous Integration checks, replacing the current mix of shell scripts and Python test runners.

This refactor will make the project significantly easier to use, test, and extend.

## II. Guiding Principles

-   **Single Entry Points:** Interactions will be funneled through a simple `./run` dispatcher and a `python run_ci.py` command.
-   **Clear API Abstraction:** All physics engines will conform to a standard Python interface, making them interchangeable "backends".
-   **Configuration over Code:** The behavior of the runner will be controlled by command-line arguments, not by changing the code.
-   **Python-First CI:** Consolidating CI logic into a single Python file improves portability and maintainability.

---

## III. The Four-Phase Execution Plan

### Phase 1: Implement the Unified Simulation Runner

This phase focuses on creating the core tooling to run, save, and compare simulations.

#### **Task 1.1: Create the `./run` Command Dispatcher**

This is a simple user-facing shell script that delegates tasks to the more complex Python scripts.

-   **File to Create:** `./run`
-   **Permissions:** The file must be executable (`chmod +x ./run`).
-   **Specification:**
    -   It will be a `bash` script (`#!/bin/bash`).
    -   It will parse these command-line arguments:
        -   `--naive`: Flag to run the pure Python "Oracle" engine.
        -   `--c`: Flag to run the C-accelerated engine.
        -   `--webgpu`: Flag to run the (future) WebGPU engine.
        -   `--steps <N>`: Integer specifying the number of simulation steps. **Default: `200`**.
        -   `--save-all-frames`: A flag. If present, every simulation frame is saved. If absent (the default behavior), only the **first and last frames** are saved for comparison.
        -   `--no-render`: A flag to prevent the renderer from running automatically after the simulation.
    -   **Behavior:**
        1.  **Simulation Mode:** If any engine flag (`--naive`, `--c`, `--webgpu`) is provided, the script will:
            -   Construct and execute a command to the main Python workhorse: `python scripts/run_simulation.py --modes naive c --steps 200 ...` (passing along all relevant arguments).
            -   After the simulation finishes, and if `--no-render` is **not** present, it will automatically call the renderer to visualize the newly generated `.npy` file(s).
        2.  **Render-Only Mode:** If no engine flags are provided, the script will find the most recently created `.npy` file(s) in `artifacts/` and run the renderer to visualize them. This provides a quick "replay" functionality.

#### **Task 1.2: Create the `scripts/run_simulation.py` Workhorse**

This script contains the main logic for setting up scenes and running the different physics engines.

-   **File to Create:** `scripts/run_simulation.py`
-   **Specification:**
    -   Use Python's `argparse` to handle the arguments passed from `./run`.
    -   **Engine Abstraction Layer:** Define a common interface that all engine wrappers will implement. This can be a formal `abc.ABC` or an informal convention.
        ```python
        class PhysicsEngine:
            def __init__(self, bodies: np.ndarray, dt: float = 0.01):
                raise NotImplementedError
            def step(self) -> None:
                raise NotImplementedError
            def get_state(self) -> np.ndarray:
                raise NotImplementedError
        ```
    -   **Engine Wrappers:** Implement wrappers for each engine that conform to the `PhysicsEngine` interface.
        -   `NaiveEngine(PhysicsEngine)`: Wraps the existing `TensorPhysicsEngine` from `physics/engine.py`.
        -   `CEngine(PhysicsEngine)`: Wraps the C-accelerated engine (logic to be refactored in Task 1.3).
        -   `WebGPUEngine(PhysicsEngine)`: A placeholder that prints a "not implemented" message.
    -   **Scene Management:** Include a function like `create_default_scene() -> np.ndarray` that generates a consistent set of initial bodies for all simulations to use.
    -   **Main Logic:**
        1.  For each engine specified in the `--modes` argument:
        2.  Initialize the engine with the default scene.
        3.  Loop for the specified number of `--steps`, calling `engine.step()` each time.
        4.  Collect the body states (as NumPy arrays) for each frame.
        5.  Based on the `--save-all-frames` flag, either keep all collected frames or just the first and last ones.
        6.  Save the final array of states to a uniquely named file in the `artifacts/` directory (e.g., `artifacts/naive_sim_200steps_16776_all.npy`).

#### **Task 1.3: Refactor the C-Op Engine for Unified Access**

Isolate the C-accelerated engine logic from the example script into a reusable module.

-   **File to Create:** `physics/engine_c.py`
-   **Action:**
    -   Move the core logic for running the C-op from `custom_ops/examples/basic_demo.py` into this new file.
    -   Encapsulate this logic within a class that implements our `PhysicsEngine` interface, ready to be imported and used by `scripts/run_simulation.py`.

---

### Phase 2: Enhance the Renderer for Comparison Visualization

The renderer must be updated to display multiple simulation results simultaneously with transparency.

-   **Files to Modify:** `renderer/src/main.rs` (and potentially other Rust/WGSL files).
-   **Specification:**
    1.  **Argument Parsing:** Update the argument parser (e.g., `clap`) to accept a list of input files and their corresponding visual properties. The `./run` script will construct a command like this:
        ```bash
        ./renderer \
          artifacts/naive_sim.npy --color "0.2 0.5 1.0" --alpha 0.5 \
          artifacts/c_sim.npy --color "1.0 0.9 0.2" --alpha 0.5
        ```
    2.  **GPU Resource Management:** For each input `.npy` file, allocate a separate GPU buffer for its body data and a separate uniform buffer for its color and alpha values.
    3.  **Render Loop:** In the main render loop, iterate through each simulation's data, bind its corresponding vertex and uniform buffers, and issue a draw call. The GPU's alpha blending will handle the transparent rendering automatically.
    4.  **Color Scheme:** The `./run` script will be responsible for providing the correct colors:
        -   **Oracle (Naive):** Blue-ish `(0.2, 0.5, 1.0)`
        -   **C-Op:** Yellow-ish `(1.0, 0.9, 0.2)`
        -   **WebGPU:** Purple-ish `(0.8, 0.3, 1.0)`

---

### Phase 3: Simplify and Consolidate the CI Pipeline

Replace the fragmented CI execution with a single, clean Python script.

-   **File to Create:** `run_ci.py` (at the project root).
-   **Action:**
    1.  Create this new file to serve as the single entry point for all CI tasks.
    2.  **Consolidate Logic:** Port all the setup, test discovery, and execution logic from the existing `ci` shell script and the `tests/run_ci.py` script into this one file.
    3.  **Functionality:** The script should perform all the same checks as before: linting, running the custom test functions, and invoking `pytest` on the required test files.
    4.  **Execution:** The CI workflow in the cloud (e.g., GitHub Actions) will be updated to simply run `python run_ci.py`.

-   **Files to Delete:**
    -   `./ci` (the old shell script).
    -   `tests/run_ci.py` (its logic is now in the root `run_ci.py`).

---

### Phase 4: Finalize and Document

-   **Update `README.md`:** The main project `README.md` should be updated to document the new, simplified `run` and `run_ci.py` commands, explaining how to use them. Remove references to the old, deleted scripts.
-   **Review and Cleanup:** Perform a final review to remove any now-unused scripts or files. 