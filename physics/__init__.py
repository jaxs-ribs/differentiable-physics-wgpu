"""Tinygrad-based physics engine with modular architecture."""
from .types import BodySchema, ShapeType, create_body_array
from .integration import integrate
from .broadphase_tensor import differentiable_broadphase
from .narrowphase import narrowphase
from .solver import resolve_collisions
from .engine import TensorPhysicsEngine

__all__ = ['BodySchema', 'ShapeType', 'create_body_array',
           'integrate', 'differentiable_broadphase', 'narrowphase', 'resolve_collisions',
           'TensorPhysicsEngine']