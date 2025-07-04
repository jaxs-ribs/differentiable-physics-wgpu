import sys
sys.path.insert(0, ".")
sys.path.insert(0, "external/tinygrad")

import numpy as np
from tinygrad import Tensor
from physics.types import ShapeType
from physics.xpbd.narrowphase import sphere_plane_test

# Test sphere-plane collision directly
# Sphere: center at (0, 0.54, 0), radius 0.5
# Plane: center at (0, 0, 0), normal (0, 1, 0), half-thickness 0.05

x_sphere = Tensor([[0.0, 0.54, 0.0]])
x_plane = Tensor([[0.0, 0.0, 0.0]])
q_plane = Tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity quaternion
params_sphere = Tensor([[0.5, 0.0, 0.0]])  # Radius 0.5
params_plane = Tensor([[10.0, 0.05, 10.0]])  # Box half-extents

pen, norm, cp = sphere_plane_test(x_sphere, x_plane, q_plane, params_sphere, params_plane)

print("Direct sphere-plane test:")
print(f"  Sphere at: {x_sphere.numpy()}")
print(f"  Plane at: {x_plane.numpy()}")
print(f"  Sphere radius: {params_sphere.numpy()[0][0]}")
print(f"  Plane half-thickness: {params_plane.numpy()[0][1]}")
print()
print(f"  Penetration: {pen.numpy()[0]}")
print(f"  Normal: {norm.numpy()[0]}")
print(f"  Contact point: {cp.numpy()[0]}")
print()
print("Expected:")
print(f"  Sphere bottom: {0.54 - 0.5} = 0.04")
print(f"  Plane top: {0.0 + 0.05} = 0.05")
print(f"  Expected penetration: {0.05 - 0.04} = 0.01")

# Test with different positions
print("\n=== Testing different sphere positions ===")
positions = [0.54, 0.55, 0.56, 0.50, 0.45]
for y in positions:
    x_sphere = Tensor([[0.0, y, 0.0]])
    pen, _, _ = sphere_plane_test(x_sphere, x_plane, q_plane, params_sphere, params_plane)
    sphere_bottom = y - 0.5
    plane_top = 0.05
    expected_pen = plane_top - sphere_bottom
    print(f"Y={y}: penetration={pen.numpy()[0]:.4f}, expected={expected_pen:.4f}")