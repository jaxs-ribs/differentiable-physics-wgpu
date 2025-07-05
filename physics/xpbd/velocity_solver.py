from tinygrad import Tensor, dtypes


def solve_velocities(v: Tensor, omega: Tensor, contacts: dict, inv_mass: Tensor, 
                    inv_inertia: Tensor, dt: float, lambda_acc: Tensor, restitution: float = 0.1) -> tuple[Tensor, Tensor]:
    if 'ids_a' not in contacts:
        return v, omega
    
    ids_a = contacts['ids_a']
    ids_b = contacts['ids_b']
    normals = contacts['normal']
    friction = contacts.get('friction', Tensor.zeros((ids_a.shape[0],)))
    penetrations = contacts.get('p', Tensor.zeros((ids_a.shape[0],)))
    contact_count = contacts.get('contact_count', Tensor.zeros(1))
    
    # Create valid mask based on both invalid IDs and contact count
    contact_indices = Tensor.arange(ids_a.shape[0])
    valid_mask = (ids_a != -1) & (ids_b != -1) & (contact_indices < contact_count)
    
    v_a = v.gather(0, ids_a.unsqueeze(-1).expand(-1, 3))
    v_b = v.gather(0, ids_b.unsqueeze(-1).expand(-1, 3))
    v_rel = v_a - v_b
    
    v_n = (v_rel * normals).sum(axis=-1)
    
    approaching_mask = (v_n < 0) & valid_mask
    contact_mask = (penetrations > 0) & valid_mask
    
    delta_v_n = -(1 + restitution) * v_n
    
    inv_mass_a = inv_mass.gather(0, ids_a)
    inv_mass_b = inv_mass.gather(0, ids_b)
    
    eff_inv_mass = inv_mass_a + inv_mass_b
    
    j_n = approaching_mask.where(delta_v_n / (eff_inv_mass + 1e-8), Tensor.zeros_like(delta_v_n))
    
    v_t = v_rel - v_n.unsqueeze(-1) * normals
    
    v_t_mag = (v_t * v_t).sum(axis=-1).sqrt() + 1e-8
    
    t_dir = v_t / v_t_mag.unsqueeze(-1)
    
    j_t_needed = v_t_mag / (eff_inv_mass + 1e-8)
    
    max_friction_force = friction * lambda_acc.abs()
    
    j_t = j_t_needed.minimum(max_friction_force)
    j_t = valid_mask.where(j_t, Tensor.zeros_like(j_t))
    
    delta_v_a_normal = j_n.unsqueeze(-1) * inv_mass_a.unsqueeze(-1) * normals
    delta_v_b_normal = -j_n.unsqueeze(-1) * inv_mass_b.unsqueeze(-1) * normals
    
    delta_v_a_friction = -j_t.unsqueeze(-1) * inv_mass_a.unsqueeze(-1) * t_dir
    delta_v_b_friction = j_t.unsqueeze(-1) * inv_mass_b.unsqueeze(-1) * t_dir
    
    delta_v_a = delta_v_a_normal + delta_v_a_friction
    delta_v_b = delta_v_b_normal + delta_v_b_friction
    
    apply_mask = approaching_mask | contact_mask
    delta_v_a = apply_mask.unsqueeze(-1).where(delta_v_a, Tensor.zeros_like(delta_v_a))
    delta_v_b = apply_mask.unsqueeze(-1).where(delta_v_b, Tensor.zeros_like(delta_v_b))
    
    v_new = apply_velocity_corrections(v, ids_a, ids_b, delta_v_a, delta_v_b, apply_mask)
    
    omega_new = omega
    
    return v_new, omega_new


def apply_velocity_corrections(v: Tensor, ids_a: Tensor, ids_b: Tensor,
                             delta_v_a: Tensor, delta_v_b: Tensor,
                             valid_mask: Tensor) -> Tensor:
    # JIT-compatible scatter-add implementation using one-hot encoding
    num_bodies = v.shape[0]
    num_contacts = ids_a.shape[0]  # Always MAX_CONTACTS_PER_STEP
    
    # Create one-hot masks
    # mask_a[i, j] = 1 if contact i's first body is body j
    body_indices = Tensor.arange(num_bodies).unsqueeze(0)  # Shape: (1, num_bodies)
    ids_a_expanded = ids_a.unsqueeze(1)  # Shape: (num_contacts, 1)
    ids_b_expanded = ids_b.unsqueeze(1)  # Shape: (num_contacts, 1)
    
    # Create masks and apply valid_mask to zero out invalid contacts
    mask_a = (ids_a_expanded == body_indices) & valid_mask.unsqueeze(1)  # Shape: (num_contacts, num_bodies)
    mask_b = (ids_b_expanded == body_indices) & valid_mask.unsqueeze(1)  # Shape: (num_contacts, num_bodies)
    
    # Convert boolean masks to float for matrix multiplication
    mask_a = mask_a.float()
    mask_b = mask_b.float()
    
    # Perform scatter-add via matrix multiplication
    # mask_a.T @ delta_v_a gives the sum of all delta_v_a for each body
    corrections_a = mask_a.transpose(0, 1).matmul(delta_v_a)  # Shape: (num_bodies, 3)
    corrections_b = mask_b.transpose(0, 1).matmul(delta_v_b)  # Shape: (num_bodies, 3)
    
    # Total corrections
    total_corrections = corrections_a + corrections_b
    
    # Apply corrections
    return v + total_corrections