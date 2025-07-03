from tinygrad import Tensor, dtypes


def solve_constraints(x_pred: Tensor, q_pred: Tensor, contacts: dict, 
                     inv_mass: Tensor, inv_inertia: Tensor, dt: float,
                     iterations: int = 8) -> tuple[Tensor, Tensor]:
    if 'ids_a' not in contacts or contacts['ids_a'].shape[0] == 0:
        return x_pred, q_pred
    
    ids_a = contacts['ids_a']
    ids_b = contacts['ids_b']
    normals = contacts['normal']
    penetrations = contacts['p']
    compliance = contacts['compliance']
    
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
    corrections = Tensor.zeros_like(x)
    
    valid_ids_a = valid_mask.where(ids_a, 0)
    valid_ids_b = valid_mask.where(ids_b, 0)
    
    masked_delta_a = valid_mask.unsqueeze(-1).where(delta_x_a, Tensor.zeros_like(delta_x_a))
    masked_delta_b = valid_mask.unsqueeze(-1).where(delta_x_b, Tensor.zeros_like(delta_x_b))
    
    idx_a_expanded = valid_ids_a.unsqueeze(-1).expand(-1, 3)
    idx_b_expanded = valid_ids_b.unsqueeze(-1).expand(-1, 3)
    
    corrections = corrections.scatter_reduce(0, idx_a_expanded, masked_delta_a, 'sum')
    
    corrections = corrections.scatter_reduce(0, idx_b_expanded, masked_delta_b, 'sum')
    
    return x + corrections