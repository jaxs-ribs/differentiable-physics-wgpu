from enum import IntEnum
import numpy as np
from typing import NamedTuple

class ExecutionMode(IntEnum):
    ORACLE = 0

class ShapeType(IntEnum):
  SPHERE = 0
  BOX = 2
  CAPSULE = 3
  PLANE = 4

class Contact(NamedTuple):
  pair_indices: tuple[int, int]
  normal: np.ndarray
  depth: float
  point: np.ndarray

def create_soa_body_data(positions: list[np.ndarray], velocities: list[np.ndarray], 
                         orientations: list[np.ndarray], angular_vels: list[np.ndarray],
                         masses: list[float], shape_types: list[ShapeType], 
                         shape_params: list[np.ndarray], frictions: list[float]) -> dict[str, np.ndarray]:
    n_bodies = len(positions)
    
    x = np.stack(positions, axis=0)
    v = np.stack(velocities, axis=0)
    q = np.stack(orientations, axis=0)
    omega = np.stack(angular_vels, axis=0)
    
    inv_mass = np.array([1.0 / m if m < 1e7 else 0.0 for m in masses], dtype=np.float32)
    
    inv_inertia = np.zeros((n_bodies, 3, 3), dtype=np.float32)
    for i, (mass, shape_type, params) in enumerate(zip(masses, shape_types, shape_params)):
        if shape_type == ShapeType.SPHERE:
            radius = params[0]
            inertia_val = 2/5 * mass * radius**2
            inertia = np.diag([inertia_val, inertia_val, inertia_val])
        elif shape_type == ShapeType.BOX:
            half_extents = params
            lx, ly, lz = 2*half_extents
            inertia_xx = (1/12) * mass * (ly**2 + lz**2)
            inertia_yy = (1/12) * mass * (lx**2 + lz**2)
            inertia_zz = (1/12) * mass * (lx**2 + ly**2)
            inertia = np.diag([inertia_xx, inertia_yy, inertia_zz])
        elif shape_type == ShapeType.CAPSULE:
            radius = params[0]
            half_height = params[1]
            cylinder_mass = mass * (2 * half_height) / (2 * half_height + 4/3 * np.pi * radius)
            hemisphere_mass = mass - cylinder_mass
            
            cylinder_ixx = cylinder_mass * (3 * radius**2 + 4 * half_height**2) / 12
            cylinder_iyy = cylinder_ixx
            cylinder_izz = cylinder_mass * radius**2 / 2
            
            hemisphere_ixx = hemisphere_mass * (2/5 * radius**2 + 1/2 * half_height**2)
            hemisphere_iyy = hemisphere_ixx
            hemisphere_izz = hemisphere_mass * 2/5 * radius**2
            
            inertia_xx = cylinder_ixx + hemisphere_ixx
            inertia_yy = cylinder_iyy + hemisphere_iyy
            inertia_zz = cylinder_izz + hemisphere_izz
            inertia = np.diag([inertia_xx, inertia_yy, inertia_zz])
        else:
            inertia = np.diag([1.0, 1.0, 1.0])
        
        inv_inertia[i] = np.linalg.inv(inertia)
    
    shape_type_array = np.array([st.value for st in shape_types], dtype=np.int32)
    shape_param_array = np.stack(shape_params, axis=0)
    friction_array = np.array(frictions, dtype=np.float32)
    
    return {
        'x': x,
        'v': v, 
        'q': q,
        'omega': omega,
        'inv_mass': inv_mass,
        'inv_inertia': inv_inertia,
        'shape_type': shape_type_array,
        'shape_params': shape_param_array,
        'friction': friction_array
    }
