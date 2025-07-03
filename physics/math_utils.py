"""Mathematical utility functions for XPBD physics calculations.

This module provides essential mathematical operations for quaternions and vectors
using tinygrad tensors, optimized for GPU/parallel computation.
"""
from tinygrad import Tensor


def quat_mul(q1: Tensor, q2: Tensor) -> Tensor:
    # Extract components
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    # Compute product components
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    # Stack components
    return Tensor.stack(w, x, y, z, dim=-1)


def quat_exp(v: Tensor) -> Tensor:
    # Compute the magnitude of the vector
    v_norm = (v * v).sum(axis=-1, keepdim=True).sqrt()
    
    # Handle zero vectors (no rotation)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    v_norm_safe = v_norm + epsilon
    
    # Compute quaternion components
    half_angle = v_norm / 2.0
    w = half_angle.cos()
    
    # Compute imaginary part: sin(half_angle) * v/||v||
    sin_half = half_angle.sin()
    v_normalized = v / v_norm_safe
    xyz = sin_half * v_normalized
    
    # Concatenate w and xyz components
    return Tensor.cat(w, xyz, dim=-1)


def quat_normalize(q: Tensor) -> Tensor:
    # Compute quaternion magnitude
    q_norm = (q * q).sum(axis=-1, keepdim=True).sqrt()
    
    # Avoid division by zero
    epsilon = 1e-8
    q_norm_safe = q_norm + epsilon
    
    # Normalize
    return q / q_norm_safe


def cross_product(v1: Tensor, v2: Tensor) -> Tensor:
    # Extract components
    x1, y1, z1 = v1[..., 0], v1[..., 1], v1[..., 2]
    x2, y2, z2 = v2[..., 0], v2[..., 1], v2[..., 2]
    
    # Compute cross product components
    x = y1*z2 - z1*y2
    y = z1*x2 - x1*z2
    z = x1*y2 - y1*x2
    
    # Stack components
    return Tensor.stack(x, y, z, dim=-1)


def apply_quaternion_to_vector(q: Tensor, v: Tensor) -> Tensor:
    # Convert vector to pure quaternion
    v_quat = Tensor.cat(Tensor.zeros_like(v[..., :1]), v, dim=-1)
    
    # Compute quaternion conjugate
    q_conj = Tensor.cat(q[..., :1], -q[..., 1:], dim=-1)
    
    # Apply rotation: q ⊗ v ⊗ q*
    result = quat_mul(quat_mul(q, v_quat), q_conj)
    
    # Extract vector part (ignore scalar component)
    return result[..., 1:]