"""
Narrow-phase collision detection with table-driven dispatcher.
All collision test functions and the dispatcher in one place.
"""
from typing import Callable, Dict, Tuple
import numpy as np
from tinygrad import Tensor, dtypes
from physics.types import ShapeType
from physics.math_utils import apply_quaternion_to_vector
from .broadphase_consts import MAX_CONTACTS_PER_STEP


# Configuration constants
PLANE_THICKNESS_THRESHOLD = 0.1  # Boxes thinner than this are treated as planes
MAX_SHAPE_TYPE = 5  # Must be larger than the maximum ShapeType enum value

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
    Generate contact information for collision pairs using vectorized dispatch.
    
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
    
    # First, ensure candidate_pairs is exactly MAX_CONTACTS_PER_STEP in size
    input_size = candidate_pairs.shape[0]
    if input_size < MAX_CONTACTS_PER_STEP:
        # Pad with invalid pairs
        pad_size = MAX_CONTACTS_PER_STEP - input_size
        pad_pairs = Tensor.full((pad_size, 2), -1, dtype=dtypes.int32)
        candidate_pairs = candidate_pairs.cat(pad_pairs, dim=0)
    else:
        # Truncate to MAX_CONTACTS_PER_STEP
        candidate_pairs = candidate_pairs[:MAX_CONTACTS_PER_STEP]
    
    # Now all operations work on fixed-size tensors
    valid_mask = candidate_pairs[:, 0] != -1
    
    # Extract pair indices
    ids_a = candidate_pairs[:, 0]
    ids_b = candidate_pairs[:, 1]
    
    # For invalid pairs, use index 0 (will be masked out later)
    safe_ids_a = valid_mask.where(ids_a, 0)
    safe_ids_b = valid_mask.where(ids_b, 0)
    
    # Gather shape data using safe indices
    shape_type_a = shape_type.gather(0, safe_ids_a)
    shape_type_b = shape_type.gather(0, safe_ids_b)
    
    # Pre-compute plane detection and rewrite shape types
    params_a = shape_params.gather(0, safe_ids_a.unsqueeze(-1).expand(-1, 3))
    params_b = shape_params.gather(0, safe_ids_b.unsqueeze(-1).expand(-1, 3))
    
    is_plane_a = (shape_type_a == ShapeType.BOX) & (params_a[:, 1] <= plane_threshold)
    is_plane_b = (shape_type_b == ShapeType.BOX) & (params_b[:, 1] <= plane_threshold)
    
    shape_type_a = is_plane_a.where(ShapeType.PLANE, shape_type_a)
    shape_type_b = is_plane_b.where(ShapeType.PLANE, shape_type_b)
    
    # Gather remaining data
    x_a = x.gather(0, safe_ids_a.unsqueeze(-1).expand(-1, 3))
    x_b = x.gather(0, safe_ids_b.unsqueeze(-1).expand(-1, 3))
    q_a = q.gather(0, safe_ids_a.unsqueeze(-1).expand(-1, 4))
    q_b = q.gather(0, safe_ids_b.unsqueeze(-1).expand(-1, 4))
    
    # Calculate contact friction
    friction_a = friction.gather(0, safe_ids_a)
    friction_b = friction.gather(0, safe_ids_b)
    contact_friction = friction_a * friction_b
    
    # Initialize outputs with fixed size MAX_CONTACTS_PER_STEP
    penetration = Tensor.zeros((MAX_CONTACTS_PER_STEP,))
    normal = Tensor.zeros((MAX_CONTACTS_PER_STEP, 3))
    contact_point = Tensor.zeros((MAX_CONTACTS_PER_STEP, 3))
    
    # VECTORIZED DISPATCH: Execute all collision tests in parallel and mask results
    
    # Sphere-Sphere
    ss_mask = (shape_type_a == ShapeType.SPHERE) & (shape_type_b == ShapeType.SPHERE)
    if (ShapeType.SPHERE, ShapeType.SPHERE) in COLLISION_TABLE:
        pen_ss, norm_ss, cp_ss = sphere_sphere_test(x_a, x_b, q_a, q_b, params_a, params_b)
        penetration = ss_mask.where(pen_ss, penetration)
        normal = ss_mask.unsqueeze(-1).where(norm_ss, normal)
        contact_point = ss_mask.unsqueeze(-1).where(cp_ss, contact_point)
    
    # Sphere-Box (and Box-Sphere)
    sb_mask = (shape_type_a == ShapeType.SPHERE) & (shape_type_b == ShapeType.BOX)
    bs_mask = (shape_type_a == ShapeType.BOX) & (shape_type_b == ShapeType.SPHERE)
    if (ShapeType.BOX, ShapeType.SPHERE) in COLLISION_TABLE:
        # Forward case: sphere-box
        pen_sb, norm_sb, cp_sb = box_sphere_test(x_b, x_a, q_b, q_a, params_b, params_a)
        norm_sb = -norm_sb  # Flip normal since we swapped arguments
        penetration = sb_mask.where(pen_sb, penetration)
        normal = sb_mask.unsqueeze(-1).where(norm_sb, normal)
        contact_point = sb_mask.unsqueeze(-1).where(cp_sb, contact_point)
        
        # Reverse case: box-sphere
        pen_bs, norm_bs, cp_bs = box_sphere_test(x_a, x_b, q_a, q_b, params_a, params_b)
        penetration = bs_mask.where(pen_bs, penetration)
        normal = bs_mask.unsqueeze(-1).where(norm_bs, normal)
        contact_point = bs_mask.unsqueeze(-1).where(cp_bs, contact_point)
    
    # Box-Box
    bb_mask = (shape_type_a == ShapeType.BOX) & (shape_type_b == ShapeType.BOX)
    if (ShapeType.BOX, ShapeType.BOX) in COLLISION_TABLE:
        pen_bb, norm_bb, cp_bb = box_box_test(x_a, x_b, q_a, q_b, params_a, params_b)
        penetration = bb_mask.where(pen_bb, penetration)
        normal = bb_mask.unsqueeze(-1).where(norm_bb, normal)
        contact_point = bb_mask.unsqueeze(-1).where(cp_bb, contact_point)
    
    # Sphere-Plane (and Plane-Sphere)
    sp_mask = (shape_type_a == ShapeType.SPHERE) & (shape_type_b == ShapeType.PLANE)
    ps_mask = (shape_type_a == ShapeType.PLANE) & (shape_type_b == ShapeType.SPHERE)
    if (ShapeType.SPHERE, ShapeType.PLANE) in COLLISION_TABLE:
        # Forward case: sphere-plane
        pen_sp, norm_sp, cp_sp = sphere_plane_test(x_a, x_b, q_a, q_b, params_a, params_b)
        penetration = sp_mask.where(pen_sp, penetration)
        normal = sp_mask.unsqueeze(-1).where(norm_sp, normal)
        contact_point = sp_mask.unsqueeze(-1).where(cp_sp, contact_point)
        
        # Reverse case: plane-sphere
        pen_ps, norm_ps, cp_ps = sphere_plane_test(x_b, x_a, q_b, q_a, params_b, params_a)
        norm_ps = -norm_ps  # Flip normal since we swapped
        penetration = ps_mask.where(pen_ps, penetration)
        normal = ps_mask.unsqueeze(-1).where(norm_ps, normal)
        contact_point = ps_mask.unsqueeze(-1).where(cp_ps, contact_point)
    
    # Capsule-Capsule
    cc_mask = (shape_type_a == ShapeType.CAPSULE) & (shape_type_b == ShapeType.CAPSULE)
    if (ShapeType.CAPSULE, ShapeType.CAPSULE) in COLLISION_TABLE:
        pen_cc, norm_cc, cp_cc = capsule_capsule_test(x_a, x_b, q_a, q_b, params_a, params_b)
        penetration = cc_mask.where(pen_cc, penetration)
        normal = cc_mask.unsqueeze(-1).where(norm_cc, normal)
        contact_point = cc_mask.unsqueeze(-1).where(cp_cc, contact_point)
    
    # Capsule-Sphere (and Sphere-Capsule)
    cs_mask = (shape_type_a == ShapeType.CAPSULE) & (shape_type_b == ShapeType.SPHERE)
    sc_mask = (shape_type_a == ShapeType.SPHERE) & (shape_type_b == ShapeType.CAPSULE)
    if (ShapeType.CAPSULE, ShapeType.SPHERE) in COLLISION_TABLE:
        # Forward case: capsule-sphere
        pen_cs, norm_cs, cp_cs = capsule_sphere_test(x_a, x_b, q_a, q_b, params_a, params_b)
        penetration = cs_mask.where(pen_cs, penetration)
        normal = cs_mask.unsqueeze(-1).where(norm_cs, normal)
        contact_point = cs_mask.unsqueeze(-1).where(cp_cs, contact_point)
        
        # Reverse case: sphere-capsule
        pen_sc, norm_sc, cp_sc = capsule_sphere_test(x_b, x_a, q_b, q_a, params_b, params_a)
        norm_sc = -norm_sc  # Flip normal since we swapped
        penetration = sc_mask.where(pen_sc, penetration)
        normal = sc_mask.unsqueeze(-1).where(norm_sc, normal)
        contact_point = sc_mask.unsqueeze(-1).where(cp_sc, contact_point)
    
    # Capsule-Box (and Box-Capsule)
    cb_mask = (shape_type_a == ShapeType.CAPSULE) & (shape_type_b == ShapeType.BOX)
    bc_mask = (shape_type_a == ShapeType.BOX) & (shape_type_b == ShapeType.CAPSULE)
    if (ShapeType.CAPSULE, ShapeType.BOX) in COLLISION_TABLE:
        # Forward case: capsule-box
        pen_cb, norm_cb, cp_cb = capsule_box_test(x_a, x_b, q_a, q_b, params_a, params_b)
        penetration = cb_mask.where(pen_cb, penetration)
        normal = cb_mask.unsqueeze(-1).where(norm_cb, normal)
        contact_point = cb_mask.unsqueeze(-1).where(cp_cb, contact_point)
        
        # Reverse case: box-capsule
        pen_bc, norm_bc, cp_bc = capsule_box_test(x_b, x_a, q_b, q_a, params_b, params_a)
        norm_bc = -norm_bc  # Flip normal since we swapped
        penetration = bc_mask.where(pen_bc, penetration)
        normal = bc_mask.unsqueeze(-1).where(norm_bc, normal)
        contact_point = bc_mask.unsqueeze(-1).where(cp_bc, contact_point)
    
    # Capsule-Plane (and Plane-Capsule)
    cp_mask = (shape_type_a == ShapeType.CAPSULE) & (shape_type_b == ShapeType.PLANE)
    pc_mask = (shape_type_a == ShapeType.PLANE) & (shape_type_b == ShapeType.CAPSULE)
    if (ShapeType.CAPSULE, ShapeType.PLANE) in COLLISION_TABLE:
        # Forward case: capsule-plane
        pen_cp, norm_cp, cp_cp = capsule_plane_test(x_a, x_b, q_a, q_b, params_a, params_b)
        penetration = cp_mask.where(pen_cp, penetration)
        normal = cp_mask.unsqueeze(-1).where(norm_cp, normal)
        contact_point = cp_mask.unsqueeze(-1).where(cp_cp, contact_point)
        
        # Reverse case: plane-capsule
        pen_pc, norm_pc, cp_pc = capsule_plane_test(x_b, x_a, q_b, q_a, params_b, params_a)
        norm_pc = -norm_pc  # Flip normal since we swapped
        penetration = pc_mask.where(pen_pc, penetration)
        normal = pc_mask.unsqueeze(-1).where(norm_pc, normal)
        contact_point = pc_mask.unsqueeze(-1).where(cp_pc, contact_point)
    
    # Apply contact processing
    # Include contacts with zero penetration to handle exact touching
    contact_mask = penetration >= -1e-6  # Small tolerance for numerical stability
    soft_penetration = contact_mask.where(softplus(penetration, beta=10.0), Tensor.zeros_like(penetration))
    
    # Combine with validity mask
    final_mask = contact_mask & valid_mask
    
    # All tensors are already MAX_CONTACTS_PER_STEP in size, so just apply masks
    final_ids_a = final_mask.where(ids_a, -1)
    final_ids_b = final_mask.where(ids_b, -1)
    final_normal = final_mask.unsqueeze(-1).where(normal, Tensor.zeros_like(normal))
    final_penetration = final_mask.where(penetration, Tensor.zeros_like(penetration))
    
    compliance_tensor = Tensor.full((MAX_CONTACTS_PER_STEP,), compliance)
    final_compliance = final_mask.where(compliance_tensor, Tensor.zeros_like(compliance_tensor))
    final_friction = final_mask.where(contact_friction, Tensor.zeros_like(contact_friction))
    
    # Count valid contacts
    num_actual_contacts = final_mask.sum()
    
    return {
        'ids_a': final_ids_a,
        'ids_b': final_ids_b,
        'normal': final_normal,
        'p': final_penetration,
        'compliance': final_compliance,
        'friction': final_friction,
        'contact_count': num_actual_contacts
    }