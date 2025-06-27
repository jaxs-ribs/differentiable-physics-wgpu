# TinyGrad Custom Op Implementation

This directory contains a proof-of-concept implementation showing how to add custom physics operations to TinyGrad using the CUSTOM op mechanism, without modifying TinyGrad's core code.

## Overview

This implementation demonstrates:
1. Creating custom C operations for physics simulation
2. Integrating them with TinyGrad's existing devices (CPU, GPU)
3. Using pattern matching to recognize and optimize physics computations
4. Extending TinyGrad without forking or modifying the core library

## Architecture

### Components

1. **physics_lib.c** - C implementation of physics operations
   - `physics_step()` - Complete physics simulation step
   - `physics_integrate()` - Position/velocity integration  
   - `physics_collisions()` - Collision detection and response

2. **physics_patterns.py** - Pattern matching for physics operations
   - Defines patterns to recognize physics computations
   - Transforms them into CUSTOM ops that call C functions
   - Provides high-level API for physics operations

3. **physics_extension.py** - Device extension mechanism
   - `PhysicsEnabledRenderer` - Wraps existing renderers with physics support
   - `enable_physics_on_device()` - Enables physics on any TinyGrad device
   - Context manager for temporary physics enablement

4. **physics_tensor_ops.py** - High-level tensor operations
   - `PhysicsTensor` - Extended tensor class with physics methods
   - Demonstrates integration with TinyGrad's tensor API

## Building

```bash
cd physics_core
make
```

This compiles the C physics library into a shared library (.so or .dylib).

## Usage

### Basic Usage

```python
from physics_extension import enable_physics_on_device
from tinygrad import Tensor

# Enable physics on CPU device
enable_physics_on_device("CPU")

# Create rigid bodies tensor [N, 8]
# Format: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, mass, radius]
bodies = Tensor.randn(100, 8)

# Perform physics step (conceptual - full integration pending)
# bodies = physics_step(bodies, dt=0.016)
```

### Context Manager

```python
from physics_extension import physics_enabled

with physics_enabled("CPU"):
    # Physics operations available here
    pass
# Physics operations disabled after context
```

### Extended Tensor Operations

```python
from physics_tensor_ops import PhysicsTensor, create_physics_world

# Create a physics world
bodies = create_physics_world(n_bodies=100)

# Integrate one time step
bodies = bodies.integrate(dt=0.016)
```

## How It Works

### Pattern Matching

The implementation uses TinyGrad's `PatternMatcher` to recognize physics patterns:

```python
# Original computation
pos = pos + vel * dt
vel = vel + gravity * dt

# Recognized and replaced with
CUSTOM("physics_integrate({0}, {1}, {2})")
```

### Device Extension

Instead of creating a new device, we extend existing devices:

```python
# Wrap the existing renderer
original = Device["CPU"].renderer
Device["CPU"].renderer = PhysicsEnabledRenderer(original)
```

### CUSTOM Op Integration

The C functions are called through TinyGrad's CUSTOM op:
- Format strings specify the C function call
- TinyGrad handles memory management
- Works with any device supporting CUSTOM ops

## Examples

Run the demonstrations:

```bash
# Basic demo
python3 physics_demo.py

# Full simulation with benchmarks
python3 physics_tensor_ops.py
```

## Technical Details

### Memory Layout

Bodies are stored as tensors with shape [N, 8]:
- Positions: columns 0-2 (x, y, z)
- Velocities: columns 3-5 (vx, vy, vz)
- Mass: column 6
- Radius: column 7

### UOp Integration

Physics operations create `Ops.CUSTOM` UOps:

```python
UOp(Ops.CUSTOM, dtype, (input_tensor,), "physics_step({0}, dt, {1})")
```

### Performance

- Direct C function calls minimize overhead
- Pattern matching enables operation fusion
- Compatible with TinyGrad's existing optimizations

## Limitations

1. **Pattern Recognition**: Currently uses placeholder patterns
2. **GPU Support**: Requires CUDA/OpenCL kernels
3. **Shape Information**: Need better tensor metadata passing
4. **Integration**: Full UOp graph integration pending

## Future Work

1. Complete pattern matching for real physics computations
2. GPU kernel implementations
3. More physics operations (constraints, soft bodies)
4. Performance optimizations (SIMD, parallelization)
5. Upstream integration with TinyGrad

## Relation to Main Physics Engine

This custom op implementation is complementary to the main physics engine:
- Main engine: Pure TinyGrad implementation for portability
- Custom ops: High-performance native operations for production

Both approaches can coexist, with custom ops as an optional acceleration layer.