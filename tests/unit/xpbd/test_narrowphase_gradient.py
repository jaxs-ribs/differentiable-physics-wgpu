"""Test gradient flow through narrowphase collision detection."""
import numpy as np
from tinygrad import Tensor
from physics.types import ShapeType
from physics.xpbd.narrowphase import generate_contacts


def test_gradient_flow_through_unique():
    """Verify that Tensor.unique() preserves gradient flow."""
    # Create positions that require gradients
    x = Tensor(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32), requires_grad=True)
    q = Tensor(np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    
    # Two spheres
    shape_type = Tensor(np.array([ShapeType.SPHERE, ShapeType.SPHERE], dtype=np.int32))
    shape_params = Tensor(np.array([[0.6, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=np.float32))
    friction = Tensor(np.array([0.5, 0.5], dtype=np.float32))
    
    # They should collide
    candidate_pairs = Tensor(np.array([[0, 1]], dtype=np.int32))
    
    # Generate contacts
    contacts = generate_contacts(x, q, candidate_pairs, shape_type, shape_params, friction)
    
    # Create a loss based on penetration
    penetration = contacts['p']
    loss = penetration.sum()
    
    # Check that we can compute gradients
    loss.backward()
    
    # The gradient should push the spheres apart
    assert x.grad is not None, "Gradient should flow through unique()"
    assert x.grad.numpy()[0, 0] < 0, "First sphere should move left"
    assert x.grad.numpy()[1, 0] > 0, "Second sphere should move right"
    
    print("Gradient flow test passed!")
    print(f"Gradients: {x.grad.numpy()}")


def test_gradient_with_mixed_shapes():
    """Test gradient flow with different shape types."""
    # Box and sphere
    x = Tensor(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32), requires_grad=True)
    q = Tensor(np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    
    shape_type = Tensor(np.array([ShapeType.BOX, ShapeType.SPHERE], dtype=np.int32))
    shape_params = Tensor(np.array([[0.6, 0.6, 0.6], [0.5, 0.0, 0.0]], dtype=np.float32))
    friction = Tensor(np.array([0.5, 0.5], dtype=np.float32))
    
    candidate_pairs = Tensor(np.array([[0, 1]], dtype=np.int32))
    
    contacts = generate_contacts(x, q, candidate_pairs, shape_type, shape_params, friction)
    
    loss = contacts['p'].sum()
    loss.backward()
    
    assert x.grad is not None, "Gradient should flow through mixed shape collision"
    print("Mixed shape gradient test passed!")


if __name__ == "__main__":
    test_gradient_flow_through_unique()
    test_gradient_with_mixed_shapes()