import sys
sys.path.insert(0, ".")
sys.path.insert(0, "external/tinygrad")

import numpy as np
from tinygrad import Tensor
from physics.types import ShapeType
from scripts.scene_builder import SceneBuilder
from physics.xpbd.broadphase import uniform_spatial_hash
from physics.xpbd.narrowphase import generate_contacts
from physics.xpbd.solver import solve_constraints, solver_iteration

# Simple test - sphere on ground
builder = SceneBuilder()

# Ground plane  
builder.add_body(
    position=[0, 0, 0],
    shape_type=ShapeType.BOX,
    shape_params=[10, 0.05, 10],
    mass=float('inf'),
    friction=0.5
)

# Sphere with penetration
builder.add_body(
    position=[0, 0.52, 0],  # 0.03 penetration
    velocity=[0, 0, 0],
    shape_type=ShapeType.SPHERE,
    shape_params=[0.5, 0, 0],
    mass=1.0,
    friction=0.5
)

scene_data = builder.build()

# Create tensors
x = Tensor(scene_data['x'].astype(np.float32))
q = Tensor(scene_data['q'].astype(np.float32))
shape_type = Tensor(scene_data['shape_type'].astype(np.int32))
shape_params = Tensor(scene_data['shape_params'].astype(np.float32))
friction = Tensor(scene_data['friction'].astype(np.float32))
inv_mass = Tensor(scene_data['inv_mass'].astype(np.float32))
dt = 0.016

# Get contacts
candidate_pairs = uniform_spatial_hash(x, shape_type, shape_params)
contacts = generate_contacts(x, q, candidate_pairs, shape_type, shape_params, friction, 0.0001)

print("Initial penetration:", contacts['p'].numpy())

# Manually iterate to see lambda accumulation
ids_a = contacts['ids_a']
ids_b = contacts['ids_b']
normals = contacts['normal']
penetrations = contacts['p']
compliance = contacts['compliance']
valid_mask = ids_a != -1

num_contacts = ids_a.shape[0]
lambda_acc = Tensor.zeros((num_contacts,))

x_corrected = x.detach()

print("\nSolver iterations:")
for i in range(8):
    old_lambda = lambda_acc.numpy().copy()
    x_corrected, lambda_acc = solver_iteration(
        x_corrected, ids_a, ids_b, normals, penetrations, 
        compliance, inv_mass, lambda_acc, dt, valid_mask
    )
    new_lambda = lambda_acc.numpy()
    delta = new_lambda - old_lambda
    print(f"  Iteration {i+1}: lambda = {new_lambda}, delta = {delta}")

print(f"\nFinal position: {x_corrected.numpy()[1]}")
print(f"Position change: {x_corrected.numpy()[1] - x.numpy()[1]}")