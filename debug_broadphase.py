import sys
sys.path.insert(0, ".")
sys.path.insert(0, "external/tinygrad")

import numpy as np
from tinygrad import Tensor
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder
from physics.xpbd.broadphase import uniform_spatial_hash

# Simple test case
builder = SceneBuilder()

# Ground plane
builder.add_body(
    position=[0, -1, 0],
    shape_type=ShapeType.BOX,
    shape_params=[10, 0.05, 10],
    mass=float('inf'),
    friction=0.5
)

# Sphere on ground
builder.add_body(
    position=[0, -0.44, 0],
    velocity=[5, 0, 0],
    shape_type=ShapeType.SPHERE,
    shape_params=[0.5, 0, 0],
    mass=1.0,
    friction=0.5
)

scene_data = builder.build()

# Create tensors
x = Tensor(scene_data['x'].astype(np.float32))
shape_type = Tensor(scene_data['shape_type'].astype(np.int32))
shape_params = Tensor(scene_data['shape_params'].astype(np.float32))

print("Positions:")
print(f"  Ground: {x.numpy()[0]}")
print(f"  Sphere: {x.numpy()[1]}")

print("\nShape types:")
print(f"  Ground: {shape_type.numpy()[0]} (BOX={ShapeType.BOX})")
print(f"  Sphere: {shape_type.numpy()[1]} (SPHERE={ShapeType.SPHERE})")

print("\nShape params:")
print(f"  Ground: {shape_params.numpy()[0]} (box half-extents)")
print(f"  Sphere: {shape_params.numpy()[1]} (radius)")

# Check broadphase
candidate_pairs = uniform_spatial_hash(x, shape_type, shape_params)

print("\nBroadphase results:")
print(f"  Candidate pairs shape: {candidate_pairs.shape}")
print(f"  Pairs:")
for i in range(candidate_pairs.shape[0]):
    pair = candidate_pairs.numpy()[i]
    if pair[0] != -1 or pair[1] != -1:
        print(f"    [{pair[0]}, {pair[1]}]")