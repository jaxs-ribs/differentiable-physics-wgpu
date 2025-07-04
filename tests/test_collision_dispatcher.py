"""Tests for the collision dispatcher."""
import numpy as np
from tinygrad import Tensor
from physics.types import ShapeType
from physics.xpbd.narrowphase import generate_contacts as generate_contacts_old
from physics.xpbd.narrowphase_clean import generate_contacts as generate_contacts_clean
from physics.xpbd.collision_dispatcher_v2 import create_collision_dispatcher


def test_dispatcher_equivalence():
    # Create test data
    n_bodies = 10
    
    # Random positions and quaternions
    x = Tensor(np.random.randn(n_bodies, 3).astype(np.float32))
    q = Tensor(np.random.randn(n_bodies, 4).astype(np.float32))
    q = q / (q * q).sum(axis=-1, keepdim=True).sqrt()  # Normalize quaternions
    
    # Random shape types
    shape_types_np = np.random.choice([ShapeType.SPHERE, ShapeType.BOX, ShapeType.CAPSULE], n_bodies)
    shape_type = Tensor(shape_types_np.astype(np.int32))
    
    # Random shape parameters
    shape_params = Tensor(np.random.rand(n_bodies, 3).astype(np.float32) * 0.5 + 0.1)
    
    # Random friction
    friction = Tensor(np.random.rand(n_bodies).astype(np.float32))
    
    # Create some collision pairs
    pairs = []
    for i in range(n_bodies):
        for j in range(i+1, n_bodies):
            if np.random.rand() > 0.7:  # 30% chance of collision pair
                pairs.append([i, j])
    
    if len(pairs) == 0:
        pairs = [[0, 1], [2, 3]]  # Ensure we have some pairs
        
    candidate_pairs = Tensor(np.array(pairs, dtype=np.int32))
    
    # Test with old implementation
    contacts_old = generate_contacts_old(x, q, candidate_pairs, shape_type, shape_params, friction)
    
    # Test with new clean implementation
    dispatcher = create_collision_dispatcher()
    contacts_clean = generate_contacts_clean(x, q, candidate_pairs, shape_type, shape_params, friction, dispatcher=dispatcher)
    
    # Compare results
    print(f"Testing with {len(pairs)} collision pairs")
    print(f"Shape types: {shape_types_np}")
    
    # Check that we get the same number of contacts
    old_valid = (contacts_old['ids_a'] != -1).numpy()
    clean_valid = (contacts_clean['ids_a'] != -1).numpy()
    
    print(f"Old implementation found {old_valid.sum()} contacts")
    print(f"Clean implementation found {clean_valid.sum()} contacts")
    
    # Compare penetrations where both found contacts
    if old_valid.sum() > 0 and clean_valid.sum() > 0:
        old_pen = contacts_old['p'].numpy()[old_valid]
        clean_pen = contacts_clean['p'].numpy()[clean_valid]
        
        print(f"Old penetrations: {old_pen}")
        print(f"Clean penetrations: {clean_pen}")
        
        # Check normals
        old_norm = contacts_old['normal'].numpy()[old_valid]
        clean_norm = contacts_clean['normal'].numpy()[clean_valid]
        
        print(f"Normal difference: {np.abs(old_norm - clean_norm).max()}")
        
    print("Test completed!")


if __name__ == "__main__":
    test_dispatcher_equivalence()
