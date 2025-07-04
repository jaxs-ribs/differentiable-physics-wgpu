"""
Narrow-phase collision detection with table-driven dispatcher.
All collision test functions and the dispatcher in one place.
"""
from typing import Callable, Dict, Tuple
import numpy as np
from tinygrad import Tensor, dtypes
from physics.types import ShapeType
from physics.math_utils import apply_quaternion_to_vector


# Configuration constants
PLANE_THICKNESS_THRESHOLD = 0.1  # Boxes thinner than this are treated as planes
MAX_SHAPE_TYPE = 10  # Must be larger than the maximum ShapeType enum value

# Global collision function registry
COLLISION_TABLE: Dict[Tuple[int, int], Callable] = {}


def register(shape_a: int, shape_b: int):
    """Decorator to register collision test functions.
    
    The function is stored under (min(a,b), max(a,b)) for canonical ordering.
    """
    def decorator(func: Callable) -> Callable:
        key = (min(shape_a, shape_b), max(shape_a, shape_b))
        COLLISION_TABLE[key] = func
        return func
    return decorator


def softplus(x: Tensor, beta: float = 10.0) -> Tensor:
    """Smooth approximation of ReLU for differentiable penetration."""
    return (1.0 / beta) * ((beta * x).exp() + 1).log()


# ============================================================================
# Collision Test Functions
# ============================================================================

@register(ShapeType.SPHERE, ShapeType.SPHERE)
def sphere_sphere_test(x_a: Tensor, x_b: Tensor, q_a: Tensor, q_b: Tensor,
                      params_a: Tensor, params_b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Test collision between two spheres. Quaternions are ignored."""
    delta = x_a - x_b
    dist = (delta * delta).sum(axis=-1, keepdim=True).sqrt()
    
    radius_a = params_a[:, 0:1]
    radius_b = params_b[:, 0:1]
    radii_sum = radius_a + radius_b
    
    penetration = radii_sum - dist
    
    epsilon = 1e-8
    normal = delta / (dist + epsilon)
    
    contact_point = x_b + normal * radius_b
    
    return penetration.squeeze(-1), normal, contact_point


@register(ShapeType.BOX, ShapeType.BOX)
def box_box_test(x_a: Tensor, x_b: Tensor, q_a: Tensor, q_b: Tensor,
                 params_a: Tensor, params_b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """SAT-based collision test for oriented boxes."""
    half_extents_a = params_a[:, :3]
    half_extents_b = params_b[:, :3]
    
    local_axes = Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    
    # Get world-space axes for both boxes
    axes_a_list = []
    axes_b_list = []
    for i in range(3):
        axis_local = local_axes[i:i+1].expand(q_a.shape[0], -1)
        axes_a_list.append(apply_quaternion_to_vector(q_a, axis_local))
        axes_b_list.append(apply_quaternion_to_vector(q_b, axis_local))
    
    axes_a = Tensor.stack(*axes_a_list, dim=1)
    axes_b = Tensor.stack(*axes_b_list, dim=1)
    
    center_diff = x_b - x_a
    
    # Find separating axis with minimum overlap
    min_overlap = Tensor.full((x_a.shape[0],), float('inf'))
    best_axis = Tensor.zeros((x_a.shape[0], 3))
    
    # Test axes from box A
    for i in range(3):
        axis = axes_a[:, i]
        projection_a = half_extents_a[:, i]
        abs_projections = (axes_b * axis.unsqueeze(1)).abs().sum(axis=2)
        projection_b = (abs_projections * half_extents_b).sum(axis=-1)
        
        center_proj = (center_diff * axis).sum(axis=-1)
        overlap = projection_a + projection_b - center_proj.abs()
        
        update_mask = overlap < min_overlap
        min_overlap = update_mask.where(overlap, min_overlap)
        best_axis = update_mask.unsqueeze(-1).where(axis, best_axis)
    
    # Test axes from box B
    for i in range(3):
        axis = axes_b[:, i]
        abs_projections = (axes_a * axis.unsqueeze(1)).abs().sum(axis=2)
        projection_a = (abs_projections * half_extents_a).sum(axis=-1)
        projection_b = half_extents_b[:, i]
        
        center_proj = (center_diff * axis).sum(axis=-1)
        overlap = projection_a + projection_b - center_proj.abs()
        
        update_mask = overlap < min_overlap
        min_overlap = update_mask.where(overlap, min_overlap)
        best_axis = update_mask.unsqueeze(-1).where(axis, best_axis)
    
    penetration = min_overlap
    
    # Ensure normal points from A to B
    center_proj_on_best = (center_diff * best_axis).sum(axis=-1, keepdim=True)
    normal = (center_proj_on_best < 0).where(-best_axis, best_axis)
    
    contact_point = x_a + center_diff * 0.5
    
    return penetration, normal, contact_point


@register(ShapeType.BOX, ShapeType.SPHERE)
def box_sphere_test(x_a: Tensor, x_b: Tensor, q_a: Tensor, q_b: Tensor,
                    params_a: Tensor, params_b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Test collision between box and sphere."""
    half_extents = params_a[:, :3]
    radius = params_b[:, 0:1]
    
    box_to_sphere = x_b - x_a
    
    local_axes = Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    
    # Transform to box space and clamp
    axes_list = []
    for i in range(3):
        axis_local = local_axes[i:i+1].expand(q_a.shape[0], -1)
        axes_list.append(apply_quaternion_to_vector(q_a, axis_local))
    
    axes = Tensor.stack(*axes_list, dim=1)
    
    local_sphere = Tensor.zeros((x_a.shape[0], 3))
    for i in range(3):
        projection = (box_to_sphere * axes[:, i]).sum(axis=-1)
        clamped = projection.clip(-half_extents[:, i], half_extents[:, i])
        local_sphere = local_sphere + clamped.unsqueeze(-1) * axes[:, i]
    
    closest_point = x_a + local_sphere
    
    sphere_to_closest = x_b - closest_point
    distance = (sphere_to_closest * sphere_to_closest).sum(axis=-1, keepdim=True).sqrt()
    
    penetration = radius - distance
    
    epsilon = 1e-8
    normal = sphere_to_closest / (distance + epsilon)
    
    contact_point = x_b - normal * radius
    
    return penetration.squeeze(-1), normal, contact_point


@register(ShapeType.CAPSULE, ShapeType.CAPSULE)
def capsule_capsule_test(x_a: Tensor, x_b: Tensor, q_a: Tensor, q_b: Tensor,
                         params_a: Tensor, params_b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Test collision between two capsules."""
    radius_a = params_a[:, 0:1]
    half_height_a = params_a[:, 1:2]
    radius_b = params_b[:, 0:1]
    half_height_b = params_b[:, 1:2]
    
    local_up = Tensor([[0.0, 1.0, 0.0]]).expand(q_a.shape[0], -1)
    axis_a = apply_quaternion_to_vector(q_a, local_up)
    axis_b = apply_quaternion_to_vector(q_b, local_up)
    
    # End points of capsule line segments
    p1a = x_a + axis_a * half_height_a
    p2a = x_a - axis_a * half_height_a
    p1b = x_b + axis_b * half_height_b
    p2b = x_b - axis_b * half_height_b
    
    # Closest points on line segments
    d1 = p2a - p1a
    d2 = p2b - p1b
    r = p1a - p1b
    
    a = (d1 * d1).sum(axis=-1)
    e = (d2 * d2).sum(axis=-1)
    f = (d2 * r).sum(axis=-1)
    
    epsilon = 1e-8
    denom = a * e - (d1 * d2).sum(axis=-1)**2
    
    s = ((d1 * d2).sum(axis=-1) * f - e * (d1 * r).sum(axis=-1)) / (denom + epsilon)
    t = ((d1 * d2).sum(axis=-1) * (d1 * r).sum(axis=-1) - a * f) / (denom + epsilon)
    
    s = s.clip(0.0, 1.0)
    t = t.clip(0.0, 1.0)
    
    closest_a = p1a + s.unsqueeze(-1) * d1
    closest_b = p1b + t.unsqueeze(-1) * d2
    
    delta = closest_a - closest_b
    distance = (delta * delta).sum(axis=-1, keepdim=True).sqrt()
    
    penetration = radius_a + radius_b - distance
    
    normal = delta / (distance + epsilon)
    
    contact_point = closest_b + normal * radius_b
    
    return penetration.squeeze(-1), normal, contact_point


@register(ShapeType.CAPSULE, ShapeType.SPHERE)
def capsule_sphere_test(x_a: Tensor, x_b: Tensor, q_a: Tensor, q_b: Tensor,
                        params_a: Tensor, params_b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Test collision between capsule and sphere."""
    capsule_radius = params_a[:, 0:1]
    half_height = params_a[:, 1:2]
    sphere_radius = params_b[:, 0:1]
    
    local_up = Tensor([[0.0, 1.0, 0.0]]).expand(q_a.shape[0], -1)
    capsule_axis = apply_quaternion_to_vector(q_a, local_up)
    
    p1 = x_a + capsule_axis * half_height
    p2 = x_a - capsule_axis * half_height
    
    # Project sphere center onto capsule line
    capsule_vec = p2 - p1
    sphere_to_p1 = x_b - p1
    
    t = (sphere_to_p1 * capsule_vec).sum(axis=-1) / ((capsule_vec * capsule_vec).sum(axis=-1) + 1e-8)
    t = t.clip(0.0, 1.0)
    
    closest_on_line = p1 + t.unsqueeze(-1) * capsule_vec
    
    sphere_to_closest = x_b - closest_on_line
    distance = (sphere_to_closest * sphere_to_closest).sum(axis=-1, keepdim=True).sqrt()
    
    penetration = capsule_radius + sphere_radius - distance
    
    epsilon = 1e-8
    normal = sphere_to_closest / (distance + epsilon)
    
    contact_point = x_b - normal * sphere_radius
    
    return penetration.squeeze(-1), normal, contact_point


@register(ShapeType.CAPSULE, ShapeType.BOX)
def capsule_box_test(x_a: Tensor, x_b: Tensor, q_a: Tensor, q_b: Tensor,
                     params_a: Tensor, params_b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Test collision between capsule and box."""
    radius = params_a[:, 0:1]
    half_height = params_a[:, 1:2]
    half_extents = params_b[:, :3]
    
    local_up = Tensor([[0.0, 1.0, 0.0]]).expand(q_a.shape[0], -1)
    capsule_axis = apply_quaternion_to_vector(q_a, local_up)
    
    p1 = x_a + capsule_axis * half_height
    p2 = x_a - capsule_axis * half_height
    
    local_axes = Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    
    axes_list = []
    for i in range(3):
        axis_local = local_axes[i:i+1].expand(q_b.shape[0], -1)
        axes_list.append(apply_quaternion_to_vector(q_b, axis_local))
    
    axes = Tensor.stack(*axes_list, dim=1)
    
    # Find closest point on box to capsule endpoints
    closest_on_capsule = Tensor.zeros_like(x_a)
    min_distance = Tensor.full((x_a.shape[0],), float('inf'))
    
    for endpoint in [p1, p2]:
        box_to_point = endpoint - x_b
        
        local_point = Tensor.zeros((x_b.shape[0], 3))
        for i in range(3):
            projection = (box_to_point * axes[:, i]).sum(axis=-1)
            clamped = projection.clip(-half_extents[:, i], half_extents[:, i])
            local_point = local_point + clamped.unsqueeze(-1) * axes[:, i]
        
        closest_on_box = x_b + local_point
        dist = ((endpoint - closest_on_box) * (endpoint - closest_on_box)).sum(axis=-1).sqrt()
        
        update_mask = dist < min_distance
        min_distance = update_mask.where(dist, min_distance)
        closest_on_capsule = update_mask.unsqueeze(-1).where(endpoint, closest_on_capsule)
    
    penetration = radius.squeeze(-1) - min_distance
    
    box_to_capsule = closest_on_capsule - x_b
    normal = box_to_capsule / (min_distance.unsqueeze(-1) + 1e-8)
    
    contact_point = closest_on_capsule - normal * radius
    
    return penetration, normal, contact_point


@register(ShapeType.SPHERE, ShapeType.PLANE)
def sphere_plane_test(x_a: Tensor, x_b: Tensor, q_a: Tensor, q_b: Tensor,
                      params_a: Tensor, params_b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Test collision between sphere and plane."""
    local_normal = Tensor([[0.0, 1.0, 0.0]]).expand(q_b.shape[0], -1)
    plane_normal = apply_quaternion_to_vector(q_b, local_normal)
    
    sphere_to_plane = x_a - x_b
    signed_distance = (sphere_to_plane * plane_normal).sum(axis=-1, keepdim=True)
    
    radius = params_a[:, 0:1]
    half_thickness = params_b[:, 1:2]
    
    distance_to_surface = signed_distance - half_thickness
    
    penetration = radius - distance_to_surface
    
    normal = plane_normal
    
    contact_point = x_a - normal * radius
    
    return penetration.squeeze(-1), normal, contact_point


@register(ShapeType.CAPSULE, ShapeType.PLANE)
def capsule_plane_test(x_a: Tensor, x_b: Tensor, q_a: Tensor, q_b: Tensor,
                       params_a: Tensor, params_b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Test collision between capsule and plane."""
    radius = params_a[:, 0:1]
    half_height = params_a[:, 1:2]
    
    local_normal = Tensor([[0.0, 1.0, 0.0]]).expand(q_b.shape[0], -1)
    plane_normal = apply_quaternion_to_vector(q_b, local_normal)
    
    local_up = Tensor([[0.0, 1.0, 0.0]]).expand(q_a.shape[0], -1)
    capsule_axis = apply_quaternion_to_vector(q_a, local_up)
    
    p1 = x_a + capsule_axis * half_height
    p2 = x_a - capsule_axis * half_height
    
    dist1 = ((p1 - x_b) * plane_normal).sum(axis=-1)
    dist2 = ((p2 - x_b) * plane_normal).sum(axis=-1)
    
    closest_on_line = (dist1 <= dist2).unsqueeze(-1).where(p1, p2)
    min_dist = (dist1 <= dist2).where(dist1, dist2)
    
    half_thickness = params_b[:, 1]
    distance_to_surface = min_dist - half_thickness
    
    penetration = radius.squeeze(-1) - distance_to_surface
    
    normal = plane_normal
    contact_point = closest_on_line - normal * radius
    
    return penetration, normal, contact_point


# ============================================================================
# Main Collision Detection Entry Point
# ============================================================================

def generate_contacts(x: Tensor, q: Tensor, candidate_pairs: Tensor, 
                     shape_type: Tensor, shape_params: Tensor, friction: Tensor, 
                     compliance: float = 0.001, plane_threshold: float = None) -> dict:
    """
    Generate contact information for collision pairs using table-driven dispatch.
    
    Args:
        x: Positions (N, 3)
        q: Quaternions (N, 4)
        candidate_pairs: Potential collision pairs from broadphase (M, 2)
        shape_type: Shape type for each body (N,)
        shape_params: Shape parameters for each body (N, 3)
        friction: Friction coefficient for each body (N,)
        compliance: Contact compliance parameter
        plane_threshold: Thickness threshold for treating boxes as planes
        
    Returns:
        Dictionary with contact information
    """
    if plane_threshold is None:
        plane_threshold = PLANE_THICKNESS_THRESHOLD
    
    # Early exit for empty pairs
    valid_mask = candidate_pairs[:, 0] != -1
    if candidate_pairs.shape[0] == 0 or not valid_mask.any().numpy():
        return {
            'ids_a': Tensor.zeros((0,), dtype=dtypes.int32),
            'ids_b': Tensor.zeros((0,), dtype=dtypes.int32),
            'normal': Tensor.zeros((0, 3)),
            'p': Tensor.zeros((0,)),
            'compliance': Tensor.zeros((0,)),
            'friction': Tensor.zeros((0,))
        }
    
    # Extract pair indices
    ids_a = candidate_pairs[:, 0]
    ids_b = candidate_pairs[:, 1]
    
    # Gather shape data
    shape_type_a = shape_type.gather(0, ids_a)
    shape_type_b = shape_type.gather(0, ids_b)
    
    # Pre-compute plane detection and rewrite shape types
    params_a = shape_params.gather(0, ids_a.unsqueeze(-1).expand(-1, 3))
    params_b = shape_params.gather(0, ids_b.unsqueeze(-1).expand(-1, 3))
    
    # Validate shape parameters width
    assert params_a.shape[-1] >= 3, f"Shape params must have width >= 3, got {params_a.shape[-1]}"
    assert params_b.shape[-1] >= 3, f"Shape params must have width >= 3, got {params_b.shape[-1]}"
    
    is_plane_a = (shape_type_a == ShapeType.BOX) & (params_a[:, 1] <= plane_threshold)
    is_plane_b = (shape_type_b == ShapeType.BOX) & (params_b[:, 1] <= plane_threshold)
    
    shape_type_a = is_plane_a.where(ShapeType.PLANE, shape_type_a)
    shape_type_b = is_plane_b.where(ShapeType.PLANE, shape_type_b)
    
    # Gather remaining data
    x_a = x.gather(0, ids_a.unsqueeze(-1).expand(-1, 3))
    x_b = x.gather(0, ids_b.unsqueeze(-1).expand(-1, 3))
    q_a = q.gather(0, ids_a.unsqueeze(-1).expand(-1, 4))
    q_b = q.gather(0, ids_b.unsqueeze(-1).expand(-1, 4))
    
    # Calculate contact friction
    friction_a = friction.gather(0, ids_a)
    friction_b = friction.gather(0, ids_b)
    contact_friction = friction_a * friction_b
    
    # Initialize outputs
    batch_size = x_a.shape[0]
    penetration = Tensor.zeros((batch_size,))
    normal = Tensor.zeros((batch_size, 3))
    contact_point = Tensor.zeros((batch_size, 3))
    
    # Process all pairs - avoiding nonzero and scatter which don't exist in tinygrad
    # For each possible shape type combination, test all pairs
    for type_a_val in range(MAX_SHAPE_TYPE):
        for type_b_val in range(MAX_SHAPE_TYPE):
            # Get canonical ordering
            type_lo = min(type_a_val, type_b_val)
            type_hi = max(type_a_val, type_b_val)
            lookup_key = (type_lo, type_hi)
            
            if lookup_key not in COLLISION_TABLE:
                continue
            
            # Build mask for this pair type
            mask = (shape_type_a == type_a_val) & (shape_type_b == type_b_val)
            
            if not mask.any().numpy():
                continue
            
            # Check if we need to swap inputs
            swapped = type_a_val > type_b_val
            
            # Process with swapping if needed
            if swapped:
                x_a_test = x_b
                x_b_test = x_a
                q_a_test = q_b
                q_b_test = q_a
                params_a_test = params_b
                params_b_test = params_a
            else:
                x_a_test = x_a
                x_b_test = x_b
                q_a_test = q_a
                q_b_test = q_b
                params_a_test = params_a
                params_b_test = params_b
            
            # Call collision test on all pairs
            test_func = COLLISION_TABLE[lookup_key]
            pen_test, norm_test, cp_test = test_func(
                x_a_test, x_b_test, q_a_test, q_b_test, 
                params_a_test, params_b_test
            )
            
            # Flip normal if swapped
            if swapped:
                norm_test = -norm_test
            
            # Use mask to update results
            mask_expanded = mask.unsqueeze(-1)
            penetration = mask.where(pen_test, penetration)
            normal = mask_expanded.where(norm_test, normal)
            contact_point = mask_expanded.where(cp_test, contact_point)
    
    # Apply contact processing
    contact_mask = penetration > 0
    soft_penetration = contact_mask.where(softplus(penetration, beta=10.0), Tensor.zeros_like(penetration))
    
    # Combine with validity mask
    final_mask = contact_mask & valid_mask
    
    # Create final outputs
    final_ids_a = final_mask.where(ids_a, -1)
    final_ids_b = final_mask.where(ids_b, -1)
    final_normal = final_mask.unsqueeze(-1).where(normal, Tensor.zeros_like(normal))
    final_physical_penetration = final_mask.where(penetration, Tensor.zeros_like(penetration))
    
    compliance_tensor = Tensor.full((ids_a.shape[0],), compliance)
    final_compliance = final_mask.where(compliance_tensor, Tensor.zeros_like(compliance_tensor))
    final_friction = final_mask.where(contact_friction, Tensor.zeros_like(contact_friction))
    
    return {
        'ids_a': final_ids_a,
        'ids_b': final_ids_b,
        'normal': final_normal,
        'p': final_physical_penetration,
        'compliance': final_compliance,
        'friction': final_friction
    }