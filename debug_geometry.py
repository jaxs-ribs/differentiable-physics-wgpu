import sys
sys.path.insert(0, ".")
sys.path.insert(0, "external/tinygrad")

import numpy as np
from tinygrad import Tensor
from physics.types import ShapeType

# Ground plane: box at (0, -1, 0) with half-extents (10, 0.05, 10)
# Top surface is at y = -1 + 0.05 = -0.95

# Sphere: radius 0.5
# For contact, sphere center must be at y <= -0.95 + 0.5 = -0.45

print("Ground plane (box):")
print("  Center: (0, -1, 0)")
print("  Half-extents: (10, 0.05, 10)")
print("  Top surface: y = -0.95")

print("\nSphere:")
print("  Radius: 0.5")

print("\nFor contact:")
print("  Sphere center must be at y <= -0.45")
print("  For 0.01 penetration: y = -0.44")
print("  For 0.05 penetration: y = -0.40")
print("  For 0.10 penetration: y = -0.35")

# Test the narrowphase logic directly
from physics.xpbd.narrowphase import sphere_plane_test
from physics.math_utils import apply_quaternion_to_vector

# Create test data
x_sphere = Tensor([[0.0, -0.40, 0.0]])  # 0.05 penetration
x_plane = Tensor([[0.0, -1.0, 0.0]])
q_plane = Tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity quaternion
params_sphere = Tensor([[0.5, 0.0, 0.0]])  # Radius 0.5
params_plane = Tensor([[10.0, 0.05, 10.0]])  # Box half-extents

pen, norm, cp = sphere_plane_test(x_sphere, x_plane, q_plane, params_sphere, params_plane)

print("\nDirect sphere-plane test:")
print(f"  Sphere at: {x_sphere.numpy()}")
print(f"  Plane at: {x_plane.numpy()}")
print(f"  Penetration: {pen.numpy()}")
print(f"  Normal: {norm.numpy()}")
print(f"  Contact point: {cp.numpy()}")