from tinygrad import Tensor, dtypes


def solve_velocities(v: Tensor, omega: Tensor, contacts: dict, inv_mass: Tensor, 
                    inv_inertia: Tensor, dt: float, restitution: float = 0.1) -> tuple[Tensor, Tensor]:
    if 'ids_a' not in contacts or contacts['ids_a'].shape[0] == 0:
        return v, omega
    
    ids_a = contacts['ids_a']
    ids_b = contacts['ids_b']
    normals = contacts['normal']
    
    valid_mask = (ids_a != -1) & (ids_b != -1)
    
    v_a = v.gather(0, ids_a.unsqueeze(-1).expand(-1, 3))
    v_b = v.gather(0, ids_b.unsqueeze(-1).expand(-1, 3))
    v_rel = v_a - v_b
    
    v_n = (v_rel * normals).sum(axis=-1)
    
    approaching_mask = (v_n < 0) & valid_mask
    
    delta_v_n = -(1 + restitution) * v_n
    
    inv_mass_a = inv_mass.gather(0, ids_a)
    inv_mass_b = inv_mass.gather(0, ids_b)
    
    eff_inv_mass = inv_mass_a + inv_mass_b
    
    j = approaching_mask.where(delta_v_n / (eff_inv_mass + 1e-8), Tensor.zeros_like(delta_v_n))
    
    delta_v_a = j.unsqueeze(-1) * inv_mass_a.unsqueeze(-1) * normals
    delta_v_b = -j.unsqueeze(-1) * inv_mass_b.unsqueeze(-1) * normals
    
    delta_v_a = approaching_mask.unsqueeze(-1).where(delta_v_a, Tensor.zeros_like(delta_v_a))
    delta_v_b = approaching_mask.unsqueeze(-1).where(delta_v_b, Tensor.zeros_like(delta_v_b))
    
    v_new = apply_velocity_corrections(v, ids_a, ids_b, delta_v_a, delta_v_b, approaching_mask)
    
    omega_new = omega
    
    return v_new, omega_new


def apply_velocity_corrections(v: Tensor, ids_a: Tensor, ids_b: Tensor,
                             delta_v_a: Tensor, delta_v_b: Tensor,
                             valid_mask: Tensor) -> Tensor:
    corrections = Tensor.zeros_like(v)
    
    valid_ids_a = valid_mask.where(ids_a, 0)
    valid_ids_b = valid_mask.where(ids_b, 0)
    
    masked_delta_a = valid_mask.unsqueeze(-1).where(delta_v_a, Tensor.zeros_like(delta_v_a))
    masked_delta_b = valid_mask.unsqueeze(-1).where(delta_v_b, Tensor.zeros_like(delta_v_b))
    
    idx_a_expanded = valid_ids_a.unsqueeze(-1).expand(-1, 3)
    idx_b_expanded = valid_ids_b.unsqueeze(-1).expand(-1, 3)
    
    corrections = corrections.scatter_reduce(0, idx_a_expanded, masked_delta_a, 'sum')
    corrections = corrections.scatter_reduce(0, idx_b_expanded, masked_delta_b, 'sum')
    
    return v + corrections