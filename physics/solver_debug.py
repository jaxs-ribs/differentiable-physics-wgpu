"""Debug version of solver to trace the issue."""

from .solver import *

# Store original resolve function
_original_resolve = resolve_collisions

def resolve_collisions_debug(bodies, pair_indices, contact_normals, contact_depths,
                           contact_points, contact_mask, restitution=0.1):
    """Debug version of resolve_collisions."""
    num_contacts = contact_mask.sum().numpy()
    
    if num_contacts > 0:
        print(f"\n=== RESOLVE DEBUG ===")
        print(f"Restitution: {restitution}")
        
        # Get first active contact
        mask_np = contact_mask.numpy()
        idx = mask_np.nonzero()[0][0]
        pair = pair_indices.numpy()[idx]
        
        # Get body velocities before
        vel_a_before = bodies[pair[0], 3:6].numpy()
        vel_b_before = bodies[pair[1], 3:6].numpy()
        
        print(f"Contact {idx}: bodies ({pair[0]}, {pair[1]})")
        print(f"  Before: vel_a={vel_a_before}, vel_b={vel_b_before}")
        
        # Call original
        result = _original_resolve(bodies, pair_indices, contact_normals, contact_depths,
                                 contact_points, contact_mask, restitution)
        
        # Get velocities after
        vel_a_after = result[pair[0], 3:6].numpy()
        vel_b_after = result[pair[1], 3:6].numpy()
        
        print(f"  After: vel_a={vel_a_after}, vel_b={vel_b_after}")
        print(f"  Change: Δvel_a={vel_a_after - vel_a_before}, Δvel_b={vel_b_after - vel_b_before}")
        
        return result
    else:
        return _original_resolve(bodies, pair_indices, contact_normals, contact_depths,
                               contact_points, contact_mask, restitution)

# Monkey patch for debugging
import physics.solver
physics.solver.resolve_collisions = resolve_collisions_debug