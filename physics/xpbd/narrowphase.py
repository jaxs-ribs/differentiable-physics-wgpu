from tinygrad import Tensor, dtypes
from physics.types import ShapeType
from physics.math_utils import apply_quaternion_to_vector


def softplus(x: Tensor, beta: float = 10.0) -> Tensor:
    return (1.0 / beta) * ((beta * x).exp() + 1).log()


def sphere_sphere_test(x_a: Tensor, x_b: Tensor, params_a: Tensor, params_b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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


def sphere_plane_test(x_sphere: Tensor, x_plane: Tensor, q_plane: Tensor, 
                      params_sphere: Tensor, params_plane: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    local_normal = Tensor([[0.0, 1.0, 0.0]]).expand(q_plane.shape[0], -1)
    plane_normal = apply_quaternion_to_vector(q_plane, local_normal)
    
    sphere_to_plane = x_sphere - x_plane
    signed_distance = (sphere_to_plane * plane_normal).sum(axis=-1, keepdim=True)
    
    radius = params_sphere[:, 0:1]
    half_thickness = params_plane[:, 1:2]
    
    distance_to_surface = signed_distance - half_thickness
    
    penetration = radius - distance_to_surface
    
    normal = plane_normal
    
    contact_point = x_sphere - normal * radius
    
    return penetration.squeeze(-1), normal, contact_point


def generate_contacts(x: Tensor, q: Tensor, candidate_pairs: Tensor, 
                     shape_type: Tensor, shape_params: Tensor, compliance: float = 0.001) -> dict:
    valid_mask = candidate_pairs[:, 0] != -1
    active_pairs = candidate_pairs
    
    if active_pairs.shape[0] == 0:
        return {
            'ids_a': Tensor.zeros((0,), dtype=dtypes.int32),
            'ids_b': Tensor.zeros((0,), dtype=dtypes.int32),
            'normal': Tensor.zeros((0, 3)),
            'p': Tensor.zeros((0,)),
            'compliance': Tensor.zeros((0,))
        }
    
    ids_a = active_pairs[:, 0]
    ids_b = active_pairs[:, 1]
    
    x_a = x.gather(0, ids_a.unsqueeze(-1).expand(-1, 3))
    x_b = x.gather(0, ids_b.unsqueeze(-1).expand(-1, 3))
    q_a = q.gather(0, ids_a.unsqueeze(-1).expand(-1, 4))
    q_b = q.gather(0, ids_b.unsqueeze(-1).expand(-1, 4))
    
    shape_type_a = shape_type.gather(0, ids_a)
    shape_type_b = shape_type.gather(0, ids_b)
    
    params_a = shape_params.gather(0, ids_a.unsqueeze(-1).expand(-1, 3))
    params_b = shape_params.gather(0, ids_b.unsqueeze(-1).expand(-1, 3))
    
    is_sphere_a = shape_type_a == ShapeType.SPHERE
    is_sphere_b = shape_type_b == ShapeType.SPHERE
    is_box_a = shape_type_a == ShapeType.BOX
    is_box_b = shape_type_b == ShapeType.BOX
    
    is_sphere_sphere = is_sphere_a & is_sphere_b
    
    plane_threshold = 0.1
    is_plane_a = is_box_a & (params_a[:, 1] <= plane_threshold)
    is_plane_b = is_box_b & (params_b[:, 1] <= plane_threshold)
    
    is_sphere_plane_ab = is_sphere_a & is_plane_b
    is_sphere_plane_ba = is_plane_a & is_sphere_b
    
    
    pen_ss, norm_ss, cp_ss = sphere_sphere_test(x_a, x_b, params_a, params_b)
    
    pen_sp_ab, norm_sp_ab, cp_sp_ab = sphere_plane_test(x_a, x_b, q_b, params_a, params_b)
    
    pen_sp_ba, norm_sp_ba, cp_sp_ba = sphere_plane_test(x_b, x_a, q_a, params_b, params_a)
    
    norm_sp_ba = -norm_sp_ba
    
    penetration = pen_ss
    normal = norm_ss
    contact_point = cp_ss
    
    penetration = is_sphere_plane_ab.where(pen_sp_ab, penetration)
    normal = is_sphere_plane_ab.unsqueeze(-1).where(norm_sp_ab, normal)
    contact_point = is_sphere_plane_ab.unsqueeze(-1).where(cp_sp_ab, contact_point)
    
    # Override with plane-sphere results where appropriate
    penetration = is_sphere_plane_ba.where(pen_sp_ba, penetration)
    normal = is_sphere_plane_ba.unsqueeze(-1).where(norm_sp_ba, normal)
    contact_point = is_sphere_plane_ba.unsqueeze(-1).where(cp_sp_ba, contact_point)
    
    contact_mask = penetration > 0
    
    soft_penetration = contact_mask.where(softplus(penetration, beta=10.0), Tensor.zeros_like(penetration))
    
    final_mask = contact_mask & valid_mask
    
    final_ids_a = final_mask.where(ids_a, -1)
    final_ids_b = final_mask.where(ids_b, -1)
    
    final_normal = final_mask.unsqueeze(-1).where(normal, Tensor.zeros_like(normal))
    
    final_physical_penetration = final_mask.where(penetration, Tensor.zeros_like(penetration))
    final_soft_penetration = final_mask.where(soft_penetration, Tensor.zeros_like(soft_penetration))
    
    compliance_tensor = Tensor.full((ids_a.shape[0],), compliance)
    final_compliance = final_mask.where(compliance_tensor, Tensor.zeros_like(compliance_tensor))
    
    return {
        'ids_a': final_ids_a,
        'ids_b': final_ids_b,
        'normal': final_normal,
        'p': final_physical_penetration,
        'p_soft': final_soft_penetration,
        'compliance': final_compliance
    }