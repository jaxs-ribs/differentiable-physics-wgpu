# Physics Engine Architecture

A modular rigid body physics engine built with tinygrad, following Clean Code principles with tinygrad's horizontal coding style.

## Module Structure

### Core Modules

- **`types.py`**: Data schemas and type definitions
  - `BodySchema`: Column indices for the state tensor (SoA layout)
  - `ShapeType`: Supported collision shapes (Sphere, Box, Capsule)
  - `Contact`: Contact point information
  - `create_body_array()`: Helper to create body state arrays

- **`math_utils.py`**: Mathematical utilities
  - Quaternion operations (multiply, rotate)
  - Coordinate transformations
  - Inertia tensor transformations

### Physics Pipeline (executed in order each timestep)

1. **`broadphase.py`**: Sweep and Prune algorithm
   - Computes AABBs for all bodies
   - O(n log n) collision pair detection
   - Returns potentially colliding pairs

2. **`narrowphase.py`**: Exact collision detection
   - Sphere-sphere collision
   - Sphere-box collision
   - Generates contact points with normals and depths

3. **`solver.py`**: Impulse-based collision resolution
   - Calculates impulses from contacts
   - Updates velocities to resolve collisions
   - Applies position correction for stability

4. **`integration.py`**: Semi-implicit Euler integration
   - Updates velocities from forces (gravity)
   - Updates positions from velocities
   - Updates orientations from angular velocities
   - Normalizes quaternions to prevent drift

### Entry Points

- **`engine.py`**: Main physics engine class that orchestrates the pipeline
- **`main.py`**: Example simulation that creates a scene and dumps states to numpy

## Usage

From the `physics_core` directory:
```bash
python3 run_physics.py --steps 200 --output oracle_dump.npy
```

## Design Philosophy

- **Clean Architecture**: Each module has a single responsibility
- **Horizontal Style**: Dense, functional code following tinygrad conventions
- **GPU-Ready**: Designed for future WGSL kernel implementation
- **Testable**: Pure functions with clear inputs/outputs