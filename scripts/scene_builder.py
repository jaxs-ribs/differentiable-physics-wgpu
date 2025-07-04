import numpy as np
from typing import Optional
from physics.types import ShapeType, create_soa_body_data


class SceneBuilder:
    
    def __init__(self):
        self.positions = []
        self.velocities = []
        self.orientations = []
        self.angular_vels = []
        self.masses = []
        self.shape_types = []
        self.shape_params = []
        self.frictions = []
    
    def add_body(self, position: np.ndarray, mass: float, shape_type: ShapeType, 
                 shape_params: np.ndarray, velocity: Optional[np.ndarray] = None,
                 orientation: Optional[np.ndarray] = None, 
                 angular_vel: Optional[np.ndarray] = None,
                 friction: float = 0.5) -> 'SceneBuilder':
        position = np.array(position, dtype=np.float32)
        if position.shape != (3,):
            raise ValueError("Position must be a 3D vector")
        
        shape_params = np.array(shape_params, dtype=np.float32)
        if shape_params.shape != (3,):
            raise ValueError("Shape params must be a 3D vector")
        
        if velocity is None:
            velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            velocity = np.array(velocity, dtype=np.float32)
            if velocity.shape != (3,):
                raise ValueError("Velocity must be a 3D vector")
        
        if orientation is None:
            orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            orientation = np.array(orientation, dtype=np.float32)
            if orientation.shape != (4,):
                raise ValueError("Orientation must be a 4D quaternion [w, x, y, z]")
            orientation = orientation / np.linalg.norm(orientation)
        
        if angular_vel is None:
            angular_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            angular_vel = np.array(angular_vel, dtype=np.float32)
            if angular_vel.shape != (3,):
                raise ValueError("Angular velocity must be a 3D vector")
        
        if mass <= 0:
            raise ValueError("Mass must be positive")
        
        if not isinstance(shape_type, ShapeType):
            raise ValueError("Shape type must be a ShapeType enum value")
        
        if friction < 0:
            raise ValueError("Friction must be non-negative")
        
        self.positions.append(position)
        self.velocities.append(velocity)
        self.orientations.append(orientation)
        self.angular_vels.append(angular_vel)
        self.masses.append(float(mass))
        self.shape_types.append(shape_type)
        self.shape_params.append(shape_params)
        self.frictions.append(float(friction))
        
        return self
    
    def build(self) -> dict[str, np.ndarray]:
        if not self.positions:
            raise ValueError("Cannot build empty scene. Add at least one body first.")
        
        return create_soa_body_data(
            positions=self.positions,
            velocities=self.velocities,
            orientations=self.orientations,
            angular_vels=self.angular_vels,
            masses=self.masses,
            shape_types=self.shape_types,
            shape_params=self.shape_params,
            frictions=self.frictions
        )
    
    def count(self) -> int:
        return len(self.positions)
    
    def clear(self) -> 'SceneBuilder':
        self.positions.clear()
        self.velocities.clear()
        self.orientations.clear()
        self.angular_vels.clear()
        self.masses.clear()
        self.shape_types.clear()
        self.shape_params.clear()
        self.frictions.clear()
        return self