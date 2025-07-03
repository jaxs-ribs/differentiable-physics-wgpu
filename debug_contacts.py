import sys
sys.path.insert(0, ".")
sys.path.insert(0, "external/tinygrad")

import numpy as np
from tinygrad import Tensor
from physics.xpbd.broadphase import uniform_spatial_hash
from physics.xpbd.narrowphase import generate_contacts
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder

# Create simple test scene
builder = SceneBuilder()

# Ground plane
builder.add_body(
    position=[0, -1, 0],
    shape_type=ShapeType.BOX,
    shape_params=[10, 0.05, 10],
    mass=float('inf')
)

# Sphere penetrating ground
builder.add_body(
    position=[0, -0.5, 0],  # Should be penetrating
    shape_type=ShapeType.SPHERE,
    shape_params=[0.5, 0, 0],
    mass=1.0
)

scene_data = builder.build()

# Convert to tensors
x = Tensor(scene_data['x'].astype(np.float32))
q = Tensor(scene_data['q'].astype(np.float32))
shape_type = Tensor(scene_data['shape_type'].astype(np.int32))
shape_params = Tensor(scene_data['shape_params'].astype(np.float32))

print("Scene setup:")
print(f"Ground position: {x.numpy()[0]}")
print(f"Sphere position: {x.numpy()[1]}")
print(f"Sphere radius: {shape_params.numpy()[1][0]}")

# Calculate expected penetration
plane_top = -0.95
sphere_bottom = x.numpy()[1][1] - shape_params.numpy()[1][0]
expected_pen = plane_top - sphere_bottom
print(f"\nExpected penetration: {expected_pen} m = {expected_pen*1000} mm")

# Run collision detection
candidate_pairs = uniform_spatial_hash(x, shape_type, shape_params)
print(f"\nBroadphase output:")
print(f"Candidate pairs shape: {candidate_pairs.shape}")
print(f"Candidate pairs: {candidate_pairs.numpy()}")

# Check valid pairs
valid_pairs = (candidate_pairs[:, 0] >= 0) & (candidate_pairs[:, 1] >= 0)
print(f"Valid pairs: {valid_pairs.numpy()}")

contacts = generate_contacts(x, q, candidate_pairs, shape_type, shape_params, 0.001)

print(f"\nContacts generated: {len(contacts['ids_a'].numpy())}")
valid_contacts = (contacts['ids_a'] != -1).numpy()
print(f"Valid contacts: {valid_contacts.sum()}")

if valid_contacts.sum() > 0:
    # Find first valid contact
    for i in range(len(contacts['ids_a'].numpy())):
        if contacts['ids_a'].numpy()[i] != -1:
            print(f"\nFirst valid contact:")
            print(f"  ids: {contacts['ids_a'].numpy()[i]} - {contacts['ids_b'].numpy()[i]}")
            print(f"  normal: {contacts['normal'].numpy()[i]}")
            print(f"  penetration (physical): {contacts['p'].numpy()[i]}")
            print(f"  penetration (soft): {contacts['p_soft'].numpy()[i]}")
            print(f"  compliance: {contacts['compliance'].numpy()[i]}")
            break