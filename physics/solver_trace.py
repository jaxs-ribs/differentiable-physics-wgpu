"""Detailed trace of solver calculations."""

from .solver import *
import numpy as np

# Store original resolve function
_original_resolve = resolve_collisions

def resolve_collisions_trace(bodies, pair_indices, contact_normals, contact_depths,
                           contact_points, contact_mask, restitution=0.1):
    """Trace version of resolve_collisions."""
    print("TRACE: resolve_collisions called")
    
    # Check both contact_mask sum and active_mask
    if contact_mask.shape[0] > 0:
        print(f"\n=== COLLISION CHECK ===")
        print(f"Contact mask shape: {contact_mask.shape}")
        print(f"Contact mask sum: {contact_mask.sum().numpy()}")
        
    num_contacts = contact_mask.sum().numpy()
    
    if num_contacts > 0:
        print(f"\n=== DETAILED SOLVER TRACE ===")
        print(f"Restitution: {restitution}")
        
        # Get first active contact
        mask_np = contact_mask.numpy()
        idx = mask_np.nonzero()[0][0]
        pair = pair_indices.numpy()[idx]
        normal = contact_normals.numpy()[idx]
        
        # Extract body data
        indices_a = pair_indices[:, 0]
        indices_b = pair_indices[:, 1]
        
        # Get body data
        inv_mass_a = bodies[indices_a, BodySchema.INV_MASS].numpy()[idx]
        inv_mass_b = bodies[indices_b, BodySchema.INV_MASS].numpy()[idx]
        
        vel_a = bodies[indices_a, BodySchema.VEL_X:BodySchema.VEL_Z+1].numpy()[idx]
        vel_b = bodies[indices_b, BodySchema.VEL_X:BodySchema.VEL_Z+1].numpy()[idx]
        
        print(f"\nContact {idx}: bodies ({pair[0]}, {pair[1]})")
        print(f"  Normal: {normal}")
        print(f"  inv_mass_a: {inv_mass_a}, inv_mass_b: {inv_mass_b}")
        print(f"  vel_a: {vel_a}, vel_b: {vel_b}")
        
        # Calculate relative velocity
        v_rel = vel_a - vel_b
        v_rel_normal = np.dot(v_rel, normal)
        
        print(f"\nRelative velocity:")
        print(f"  v_rel: {v_rel}")
        print(f"  v_rel_normal: {v_rel_normal}")
        
        # Calculate impulse magnitude manually
        numerator = (1.0 + restitution) * (-v_rel_normal)
        denominator = inv_mass_a + inv_mass_b
        j_magnitude = numerator / denominator
        
        print(f"\nImpulse calculation:")
        print(f"  numerator = (1 + {restitution}) * (-{v_rel_normal}) = {numerator}")
        print(f"  denominator = {inv_mass_a} + {inv_mass_b} = {denominator}")
        print(f"  j_magnitude = {j_magnitude}")
        
        # Calculate expected velocity changes
        delta_v_a = j_magnitude * normal * inv_mass_a
        delta_v_b = -j_magnitude * normal * inv_mass_b
        
        print(f"\nExpected velocity changes:")
        print(f"  delta_v_a = {j_magnitude} * {normal} * {inv_mass_a} = {delta_v_a}")
        print(f"  delta_v_b = -{j_magnitude} * {normal} * {inv_mass_b} = {delta_v_b}")
        
        print(f"\nExpected final velocities:")
        print(f"  vel_a_final = {vel_a + delta_v_a}")
        print(f"  vel_b_final = {vel_b + delta_v_b}")
        
        # Call original
        result = _original_resolve(bodies, pair_indices, contact_normals, contact_depths,
                                 contact_points, contact_mask, restitution)
        
        # Get actual velocities after
        vel_a_after = result[pair[0], BodySchema.VEL_X:BodySchema.VEL_Z+1].numpy()
        vel_b_after = result[pair[1], BodySchema.VEL_X:BodySchema.VEL_Z+1].numpy()
        
        print(f"\nActual final velocities:")
        print(f"  vel_a_after: {vel_a_after}")
        print(f"  vel_b_after: {vel_b_after}")
        
        print(f"\nDiscrepancy:")
        print(f"  Expected vel_a[1]: {(vel_a + delta_v_a)[1]:.3f}")
        print(f"  Actual vel_a[1]: {vel_a_after[1]:.3f}")
        print(f"  Ratio: {vel_a_after[1] / (vel_a + delta_v_a)[1]:.2f}x")
        
        return result
    else:
        return _original_resolve(bodies, pair_indices, contact_normals, contact_depths,
                               contact_points, contact_mask, restitution)

# Monkey patch for tracing
import physics.solver
physics.solver.resolve_collisions = resolve_collisions_trace