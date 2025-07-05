from tinygrad import Tensor, dtypes


def solve_constraints(x_pred: Tensor, q_pred: Tensor, contacts: dict, 
                     inv_mass: Tensor, inv_inertia: Tensor, dt: float,
                     iterations: int = 8) -> tuple[Tensor, Tensor]:
    if 'ids_a' not in contacts:
        return x_pred, q_pred
    
    ids_a = contacts['ids_a']
    ids_b = contacts['ids_b']
    normals = contacts['normal']
    penetrations = contacts['p']
    compliance = contacts['compliance']
    contact_count = contacts.get('contact_count', Tensor.zeros(1))
    
    # Create valid mask based on invalid IDs
    # Contacts with ids_a == -1 are invalid
    valid_mask = ids_a != -1
    
    num_contacts = ids_a.shape[0]
    
    lambda_acc = Tensor.zeros((num_contacts,))
    
    x_corrected = x_pred.detach()
    
    for _ in range(iterations):
        x_corrected, lambda_acc = solver_iteration(
            x_corrected, ids_a, ids_b, normals, penetrations, 
            compliance, inv_mass, lambda_acc, dt, valid_mask
        )
    
    return x_corrected, q_pred


def solver_iteration(x: Tensor, ids_a: Tensor, ids_b: Tensor, normals: Tensor, 
                    penetrations: Tensor, compliance: Tensor, inv_mass: Tensor,
                    lambda_acc: Tensor, dt: float, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
    inv_mass_a = inv_mass.gather(0, ids_a)
    inv_mass_b = inv_mass.gather(0, ids_b)
    
    gen_inv_mass = valid_mask.where(inv_mass_a + inv_mass_b, Tensor.ones_like(inv_mass_a))
    
    C = valid_mask.where(penetrations, Tensor.zeros_like(penetrations))
    
    alpha = compliance
    dt_squared = dt * dt
    
    numerator = -(C + alpha * lambda_acc / dt_squared)
    denominator = gen_inv_mass + alpha / dt_squared
    delta_lambda = valid_mask.where(numerator / (denominator + 1e-8), Tensor.zeros_like(numerator))
    
    lambda_new = lambda_acc + delta_lambda
    
    delta_x_a = -delta_lambda.unsqueeze(-1) * inv_mass_a.unsqueeze(-1) * normals
    delta_x_b = delta_lambda.unsqueeze(-1) * inv_mass_b.unsqueeze(-1) * normals
    
    delta_x_a = valid_mask.unsqueeze(-1).where(delta_x_a, Tensor.zeros_like(delta_x_a))
    delta_x_b = valid_mask.unsqueeze(-1).where(delta_x_b, Tensor.zeros_like(delta_x_b))
    
    x_new = apply_position_corrections(x, ids_a, ids_b, delta_x_a, delta_x_b, valid_mask)
    
    return x_new, lambda_new


def apply_position_corrections(x: Tensor, ids_a: Tensor, ids_b: Tensor, 
                              delta_x_a: Tensor, delta_x_b: Tensor,
                              valid_mask: Tensor) -> Tensor:
    # JIT-compatible scatter-add implementation using one-hot encoding
    num_bodies = x.shape[0]
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
    # mask_a.T @ delta_x_a gives the sum of all delta_x_a for each body
    corrections_a = mask_a.transpose(0, 1).matmul(delta_x_a)  # Shape: (num_bodies, 3)
    corrections_b = mask_b.transpose(0, 1).matmul(delta_x_b)  # Shape: (num_bodies, 3)
    
    # Total corrections
    total_corrections = corrections_a + corrections_b
    
    # Apply corrections
    return x + total_corrections