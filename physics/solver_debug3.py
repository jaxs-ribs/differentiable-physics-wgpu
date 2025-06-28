"""Debug solver with detailed tracing."""

from .solver import *
import numpy as np

# Store original functions
_original_resolve = resolve_collisions
_original_scatter = Tensor.scatter_reduce

# Track scatter calls
scatter_log = []

def scatter_reduce_debug(self, dim, index, src, reduce="sum", **kwargs):
    """Debug version of scatter_reduce."""
    result = _original_scatter(self, dim, index, src, reduce, **kwargs)
    
    # Log the call
    if len(index.numpy()) > 0:
        scatter_log.append({
            'self_shape': self.shape,
            'index_shape': index.shape,
            'src_shape': src.shape,
            'indices': index.numpy()[:10],  # First 10 indices
            'src_values': src.numpy()[:10],  # First 10 values
        })
    
    return result

# Monkey patch scatter_reduce
Tensor.scatter_reduce = scatter_reduce_debug

def resolve_collisions_debug3(bodies, pair_indices, contact_normals, contact_depths,
                            contact_points, contact_mask, restitution=0.1):
    """Debug version 3 with scatter tracking."""
    global scatter_log
    scatter_log = []
    
    num_contacts = contact_mask.sum().numpy()
    
    if num_contacts > 0:
        # Get first contact
        idx = 0
        pair = pair_indices.numpy()[idx]
        bodies_np = bodies.numpy()
        
        if bodies_np[pair[1], 1] < 0:  # Ball near ground
            print(f"\n=== SOLVER DEBUG 3 ===")
            print(f"Number of active contacts: {num_contacts}")
            
            vel_before = bodies_np[pair[1], 3:6]
            print(f"Ball velocity before: {vel_before}")
            
            # Call original
            result = _original_resolve(bodies, pair_indices, contact_normals, contact_depths,
                                     contact_points, contact_mask, restitution)
            
            vel_after = result.numpy()[pair[1], 3:6]
            print(f"Ball velocity after: {vel_after}")
            print(f"Change: {vel_after - vel_before}")
            
            # Print scatter log
            print(f"\nScatter operations: {len(scatter_log)}")
            for i, call in enumerate(scatter_log):
                print(f"\nScatter {i}:")
                print(f"  Target shape: {call['self_shape']}")
                print(f"  Index shape: {call['index_shape']}")
                print(f"  First few indices: {call['indices'][:5]}")
                print(f"  First few values: {call['src_values'][:5]}")
            
            return result
    
    return _original_resolve(bodies, pair_indices, contact_normals, contact_depths,
                           contact_points, contact_mask, restitution)

# Monkey patch for debugging
import physics.solver
physics.solver.resolve_collisions = resolve_collisions_debug3