"""Debug version 2 of solver to find the 12x issue."""

from .solver import *
import numpy as np

# Store original resolve function
_original_resolve = resolve_collisions

def resolve_collisions_debug2(bodies, pair_indices, contact_normals, contact_depths,
                            contact_points, contact_mask, restitution=0.1):
    """Debug version 2."""
    num_contacts = contact_mask.sum().numpy()
    
    if num_contacts > 0:
        # Get first active contact
        mask_np = contact_mask.numpy()
        idx = mask_np.nonzero()[0][0]
        pair = pair_indices.numpy()[idx]
        
        # Check if this is our ground-ball collision
        bodies_np = bodies.numpy()
        if bodies_np[pair[1], 1] < 0:  # Ball is near ground
            print(f"\n=== DETAILED SOLVER DEBUG ===")
            print(f"Contact {idx}: bodies ({pair[0]}, {pair[1]})")
            
            # Get all the data
            normal = contact_normals.numpy()[idx]
            depth = contact_depths.numpy()[idx]
            
            # Extract velocities
            vel_a = bodies_np[pair[0], 3:6]
            vel_b = bodies_np[pair[1], 3:6]
            inv_mass_a = bodies_np[pair[0], BodySchema.INV_MASS]
            inv_mass_b = bodies_np[pair[1], BodySchema.INV_MASS]
            
            print(f"  Normal: {normal}")
            print(f"  Depth: {depth:.3f}")
            print(f"  vel_a: {vel_a}")
            print(f"  vel_b: {vel_b}")
            print(f"  inv_mass_a: {inv_mass_a}, inv_mass_b: {inv_mass_b}")
            
            # Calculate what should happen
            v_rel = vel_a - vel_b
            v_rel_n = np.dot(v_rel, normal)
            
            print(f"\n  v_rel = {v_rel}")
            print(f"  v_rel_n = {v_rel_n:.3f}")
            
            j = (1 + restitution) * (-v_rel_n) / (inv_mass_a + inv_mass_b)
            print(f"\n  j = (1 + {restitution}) * (-{v_rel_n:.3f}) / ({inv_mass_a} + {inv_mass_b})")
            print(f"  j = {j:.3f}")
            
            delta_v_b = -j * normal * inv_mass_b
            expected_vel_b = vel_b + delta_v_b
            
            print(f"\n  Expected delta_v_b = {delta_v_b}")
            print(f"  Expected vel_b after = {expected_vel_b}")
            
            # Call original
            result = _original_resolve(bodies, pair_indices, contact_normals, contact_depths,
                                     contact_points, contact_mask, restitution)
            
            # Check actual result
            actual_vel_b = result.numpy()[pair[1], 3:6]
            print(f"\n  Actual vel_b after = {actual_vel_b}")
            print(f"  ERROR FACTOR: {actual_vel_b[1] / expected_vel_b[1]:.2f}x")
            
            return result
    
    return _original_resolve(bodies, pair_indices, contact_normals, contact_depths,
                           contact_points, contact_mask, restitution)

# Monkey patch for debugging
import physics.solver
physics.solver.resolve_collisions = resolve_collisions_debug2