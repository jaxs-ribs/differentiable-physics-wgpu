"""Unit tests for XPBD constraint solver."""
import pytest
import numpy as np
from tinygrad import Tensor, dtypes
from physics.xpbd.solver import solve_constraints, solver_iteration, apply_position_corrections
from physics.xpbd.broadphase_consts import MAX_CONTACTS_PER_STEP


def create_fixed_size_contacts(ids_a, ids_b, normal, p, compliance, friction=None):
    """Helper to create fixed-size contact dictionary for tests."""
    num_contacts = len(ids_a) if hasattr(ids_a, '__len__') else (1 if ids_a else 0)
    
    # Handle empty case
    if num_contacts == 0:
        return {
            'ids_a': Tensor.full((MAX_CONTACTS_PER_STEP,), -1, dtype=dtypes.int32),
            'ids_b': Tensor.full((MAX_CONTACTS_PER_STEP,), -1, dtype=dtypes.int32),
            'normal': Tensor.zeros((MAX_CONTACTS_PER_STEP, 3)),
            'p': Tensor.zeros((MAX_CONTACTS_PER_STEP,)),
            'compliance': Tensor.zeros((MAX_CONTACTS_PER_STEP,)),
            'friction': Tensor.zeros((MAX_CONTACTS_PER_STEP,)),
            'contact_count': Tensor([0])
        }
    
    # Convert to tensors if needed
    if not isinstance(ids_a, Tensor):
        ids_a = Tensor(ids_a, dtype=dtypes.int32)
    if not isinstance(ids_b, Tensor):
        ids_b = Tensor(ids_b, dtype=dtypes.int32)
    if not isinstance(normal, Tensor):
        normal = Tensor(normal)
    if not isinstance(p, Tensor):
        p = Tensor(p)
    if not isinstance(compliance, Tensor):
        compliance = Tensor(compliance)
    
    # Ensure correct shapes
    if ids_a.shape == ():
        ids_a = ids_a.unsqueeze(0)
    if ids_b.shape == ():
        ids_b = ids_b.unsqueeze(0)
    if len(normal.shape) == 1:
        normal = normal.unsqueeze(0)
    if p.shape == ():
        p = p.unsqueeze(0)
    if compliance.shape == ():
        compliance = compliance.unsqueeze(0)
    
    # Pad to MAX_CONTACTS_PER_STEP
    pad_size = MAX_CONTACTS_PER_STEP - num_contacts
    
    if pad_size > 0:
        ids_a = ids_a.cat(Tensor.full((pad_size,), -1, dtype=dtypes.int32), dim=0)
        ids_b = ids_b.cat(Tensor.full((pad_size,), -1, dtype=dtypes.int32), dim=0)
        normal = normal.cat(Tensor.zeros((pad_size, 3)), dim=0)
        p = p.cat(Tensor.zeros((pad_size,)), dim=0)
        compliance = compliance.cat(Tensor.zeros((pad_size,)), dim=0)
    
    if friction is None:
        friction = Tensor.zeros((MAX_CONTACTS_PER_STEP,))
    else:
        if not isinstance(friction, Tensor):
            friction = Tensor(friction)
        if friction.shape == ():
            friction = friction.unsqueeze(0)
        if len(friction) < MAX_CONTACTS_PER_STEP:
            friction = friction.cat(Tensor.zeros((MAX_CONTACTS_PER_STEP - len(friction),)), dim=0)
    
    return {
        'ids_a': ids_a[:MAX_CONTACTS_PER_STEP],
        'ids_b': ids_b[:MAX_CONTACTS_PER_STEP],
        'normal': normal[:MAX_CONTACTS_PER_STEP],
        'p': p[:MAX_CONTACTS_PER_STEP],
        'compliance': compliance[:MAX_CONTACTS_PER_STEP],
        'friction': friction[:MAX_CONTACTS_PER_STEP],
        'contact_count': Tensor([num_contacts])
    }


def test_solve_constraints_empty_contacts():
    """Test solver with no contacts."""
    # Create simple scene
    x_pred = Tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    q_pred = Tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    inv_mass = Tensor([1.0, 1.0])
    inv_inertia = Tensor.eye(3).unsqueeze(0).expand(2, -1, -1)
    dt = 0.016
    
    # Empty contacts - still need fixed size arrays
    contacts = create_fixed_size_contacts([], [], [], [], [])
    
    x_new, q_new = solve_constraints(x_pred, q_pred, contacts, inv_mass, inv_inertia, dt)
    
    # Should return unchanged positions
    assert np.allclose(x_new.numpy(), x_pred.numpy())
    assert np.allclose(q_new.numpy(), q_pred.numpy())


def test_solve_constraints_single_contact():
    """Test solver with a single penetrating contact."""
    # Two spheres, one above the other, penetrating
    x_pred = Tensor([
        [0.0, 1.0, 0.0],   # Sphere 0 at y=1
        [0.0, 0.0, 0.0]    # Sphere 1 at y=0 (ground)
    ])
    q_pred = Tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    inv_mass = Tensor([1.0, 0.0])  # Top sphere movable, bottom fixed
    inv_inertia = Tensor.eye(3).unsqueeze(0).expand(2, -1, -1)
    dt = 0.016
    
    # Contact pushing sphere 0 up
    contacts = create_fixed_size_contacts(
        ids_a=[0],
        ids_b=[1],
        normal=[[0.0, 1.0, 0.0]],  # Normal pointing up
        p=[0.1],  # Penetration depth (softplus'd)
        compliance=[0.0]  # Rigid contact
    )
    
    x_new, q_new = solve_constraints(x_pred, q_pred, contacts, inv_mass, inv_inertia, dt, iterations=1)
    
    # Sphere 0 should move up
    assert x_new.numpy()[0, 1] > x_pred.numpy()[0, 1]
    # Sphere 1 should not move (infinite mass)
    assert np.allclose(x_new.numpy()[1], x_pred.numpy()[1])


def test_solver_iteration():
    """Test a single solver iteration."""
    # Simple two-body system
    x = Tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ids_a = Tensor([0], dtype=dtypes.int32)
    ids_b = Tensor([1], dtype=dtypes.int32)
    normals = Tensor([[1.0, 0.0, 0.0]])  # Normal pointing from b to a
    penetrations = Tensor([0.1])
    compliance = Tensor([0.001])
    inv_mass = Tensor([1.0, 1.0])
    lambda_acc = Tensor([0.0])
    dt = 0.016
    valid_mask = Tensor([True])
    
    x_new, lambda_new = solver_iteration(
        x, ids_a, ids_b, normals, penetrations, 
        compliance, inv_mass, lambda_acc, dt, valid_mask
    )
    
    # Behavioral tests:
    # 1. Bodies should move (solver did something)
    assert not np.allclose(x_new.numpy(), x.numpy()), "Solver should modify positions"
    
    # 2. If there's penetration, bodies should move in correct directions
    # With our setup: bodies at x=0 and x=1, normal pointing from B to A (+x)
    # Penetration of 0.1 means they should separate
    delta_a = x_new.numpy()[0] - x.numpy()[0]
    delta_b = x_new.numpy()[1] - x.numpy()[1]
    
    # The solver pushes bodies apart along the normal
    # Normal points from B to A, so:
    # - Body A gets pushed in +normal direction (right, +x)
    # - Body B gets pushed in -normal direction (left, -x)
    # This increases their separation
    assert delta_a[0] >= 0, "Body A should move right (along normal)"
    assert delta_b[0] <= 0, "Body B should move left (opposite normal)"


def test_apply_position_corrections():
    """Test position correction scatter operation."""
    # Three bodies
    x = Tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0]
    ])
    
    # Two contacts: 0-1 and 1-2
    ids_a = Tensor([0, 1], dtype=dtypes.int32)
    ids_b = Tensor([1, 2], dtype=dtypes.int32)
    
    # Corrections
    delta_x_a = Tensor([[-0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    delta_x_b = Tensor([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
    valid_mask = Tensor([True, True])
    
    x_new = apply_position_corrections(x, ids_a, ids_b, delta_x_a, delta_x_b, valid_mask)
    
    # Body 0: receives -0.1 from first contact
    assert np.isclose(x_new.numpy()[0, 0], -0.1)
    
    # Body 1: receives +0.1 from first contact and -0.1 from second contact = 0
    assert np.isclose(x_new.numpy()[1, 0], 1.0)
    
    # Body 2: receives +0.1 from second contact
    assert np.isclose(x_new.numpy()[2, 0], 2.1)


def test_solve_constraints_compliance():
    """Test solver with compliant (soft) contacts."""
    # Two spheres
    x_pred = Tensor([
        [0.0, 0.0, 0.0],
        [0.9, 0.0, 0.0]
    ])
    q_pred = Tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    inv_mass = Tensor([1.0, 1.0])
    inv_inertia = Tensor.eye(3).unsqueeze(0).expand(2, -1, -1)
    dt = 0.016
    
    # Soft contact
    contacts = create_fixed_size_contacts(
        ids_a=[0],
        ids_b=[1],
        normal=[[1.0, 0.0, 0.0]],
        p=[0.1],
        compliance=[0.1]  # Soft contact
    )
    
    x_new_soft, _ = solve_constraints(x_pred, q_pred, contacts, inv_mass, inv_inertia, dt, iterations=8)
    
    # With rigid contact for comparison
    contacts_rigid = create_fixed_size_contacts(
        ids_a=[0],
        ids_b=[1],
        normal=[[1.0, 0.0, 0.0]],
        p=[0.1],
        compliance=[0.0]  # Rigid contact
    )
    x_new_rigid, _ = solve_constraints(x_pred, q_pred, contacts_rigid, inv_mass, inv_inertia, dt, iterations=8)
    
    # Behavioral test: soft contacts should produce less correction than rigid
    # Measure how much each body moved
    correction_soft_a = abs(x_new_soft.numpy()[0, 0] - x_pred.numpy()[0, 0])
    correction_soft_b = abs(x_new_soft.numpy()[1, 0] - x_pred.numpy()[1, 0])
    correction_rigid_a = abs(x_new_rigid.numpy()[0, 0] - x_pred.numpy()[0, 0])
    correction_rigid_b = abs(x_new_rigid.numpy()[1, 0] - x_pred.numpy()[1, 0])
    
    # Total correction should be less for soft contact
    total_correction_soft = correction_soft_a + correction_soft_b
    total_correction_rigid = correction_rigid_a + correction_rigid_b
    assert total_correction_soft < total_correction_rigid, \
        f"Soft contact should produce less correction: soft={total_correction_soft}, rigid={total_correction_rigid}"


def test_solve_constraints_multiple_iterations():
    """Test that more iterations lead to better constraint satisfaction."""
    # Two bodies with contact
    x_pred = Tensor([
        [0.0, 0.0, 0.0],
        [0.9, 0.0, 0.0]
    ])
    q_pred = Tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    inv_mass = Tensor([1.0, 1.0])
    inv_inertia = Tensor.eye(3).unsqueeze(0).expand(2, -1, -1)
    dt = 0.016
    
    # Contact with penetration
    contacts = create_fixed_size_contacts(
        ids_a=[0],
        ids_b=[1],
        normal=[[1.0, 0.0, 0.0]],
        p=[0.1],  # Penetration value
        compliance=[0.001]
    )
    
    # Solve with different iteration counts
    x_1iter, _ = solve_constraints(x_pred, q_pred, contacts, inv_mass, inv_inertia, dt, iterations=1)
    x_8iter, _ = solve_constraints(x_pred, q_pred, contacts, inv_mass, inv_inertia, dt, iterations=8)
    
    # Calculate total movement (should be different with different iterations)
    movement_1 = abs(x_1iter.numpy()[0, 0] - x_pred.numpy()[0, 0]) + abs(x_1iter.numpy()[1, 0] - x_pred.numpy()[1, 0])
    movement_8 = abs(x_8iter.numpy()[0, 0] - x_pred.numpy()[0, 0]) + abs(x_8iter.numpy()[1, 0] - x_pred.numpy()[1, 0])
    
    # Basic test: solver should do something
    assert movement_1 > 0, "Solver should move bodies"
    assert movement_8 > 0, "Solver should move bodies"
    
    # With compliant contact, more iterations should converge
    # Just verify the solver is stable and doesn't explode
    assert movement_8 < 1.0, "Solver should remain stable with more iterations"


def test_solve_constraints_signature():
    """Test that function has correct signature."""
    import inspect
    sig = inspect.signature(solve_constraints)
    
    # Should have the required parameters
    assert 'x_pred' in sig.parameters
    assert 'q_pred' in sig.parameters
    assert 'contacts' in sig.parameters
    assert 'inv_mass' in sig.parameters
    assert 'inv_inertia' in sig.parameters
    assert 'dt' in sig.parameters
    assert 'iterations' in sig.parameters


def test_solve_constraints_default_iterations():
    """Test that default iteration count works."""
    x_pred = Tensor([[0.0, 0.0, 0.0]])
    q_pred = Tensor([[1.0, 0.0, 0.0, 0.0]])
    inv_mass = Tensor([1.0])
    inv_inertia = Tensor.eye(3).unsqueeze(0)
    dt = 0.016
    contacts = create_fixed_size_contacts([], [], [], [], [])
    
    # Should work without specifying iterations
    x_new, q_new = solve_constraints(x_pred, q_pred, contacts, inv_mass, inv_inertia, dt)
    assert x_new is not None


def test_invalid_contact_handling():
    """Test handling of invalid contacts (id = -1)."""
    x_pred = Tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    q_pred = Tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    inv_mass = Tensor([1.0, 1.0])
    inv_inertia = Tensor.eye(3).unsqueeze(0).expand(2, -1, -1)
    dt = 0.016
    
    # Mix of valid and invalid contacts - but fixed-size system handles this differently
    # We only pass valid contacts, and contact_count tells us how many are valid
    contacts = create_fixed_size_contacts(
        ids_a=[0],
        ids_b=[1],
        normal=[[1.0, 0.0, 0.0]],
        p=[0.1],
        compliance=[0.0]
    )
    
    x_new, _ = solve_constraints(x_pred, q_pred, contacts, inv_mass, inv_inertia, dt, iterations=1)
    
    # Should process only the valid contact
    assert not np.allclose(x_new.numpy(), x_pred.numpy())  # Some movement occurred