import numpy as np
from typing import Optional
from physics.types import ShapeType, create_soa_body_data


class SceneBuilder:
    """A utility class for building physics scenes with a clean, human-readable API.
    
    Provides methods to add bodies to a scene and convert to the SoA format
    required by the XPBD physics engine.
    """
    
    def __init__(self):
        """Initialize empty lists for all body properties."""
        self.positions = []
        self.velocities = []
        self.orientations = []
        self.angular_vels = []
        self.masses = []
        self.shape_types = []
        self.shape_params = []
    
    def add_body(self, position: np.ndarray, mass: float, shape_type: ShapeType, 
                 shape_params: np.ndarray, velocity: Optional[np.ndarray] = None,
                 orientation: Optional[np.ndarray] = None, 
                 angular_vel: Optional[np.ndarray] = None) -> 'SceneBuilder':
        """Add a body to the scene.
        
        Args:
            position: 3D position [x, y, z]
            mass: Body mass (use large value like 1e8 for static bodies)
            shape_type: Shape type (SPHERE, BOX, CAPSULE)
            shape_params: Shape-specific parameters (radius for sphere, half-extents for box)
            velocity: 3D velocity [vx, vy, vz] (defaults to zero)
            orientation: Quaternion [w, x, y, z] (defaults to identity)
            angular_vel: 3D angular velocity [wx, wy, wz] (defaults to zero)
        
        Returns:
            Self for method chaining
        """
        # Convert inputs to proper numpy arrays with correct dtypes
        position = np.array(position, dtype=np.float32)
        if position.shape != (3,):
            raise ValueError("Position must be a 3D vector")
        
        shape_params = np.array(shape_params, dtype=np.float32)
        if shape_params.shape != (3,):
            raise ValueError("Shape params must be a 3D vector")
        
        # Apply defaults for optional parameters
        if velocity is None:
            velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            velocity = np.array(velocity, dtype=np.float32)
            if velocity.shape != (3,):
                raise ValueError("Velocity must be a 3D vector")
        
        if orientation is None:
            orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Identity quaternion
        else:
            orientation = np.array(orientation, dtype=np.float32)
            if orientation.shape != (4,):
                raise ValueError("Orientation must be a 4D quaternion [w, x, y, z]")
            # Normalize quaternion
            orientation = orientation / np.linalg.norm(orientation)
        
        if angular_vel is None:
            angular_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            angular_vel = np.array(angular_vel, dtype=np.float32)
            if angular_vel.shape != (3,):
                raise ValueError("Angular velocity must be a 3D vector")
        
        # Validate mass
        if mass <= 0:
            raise ValueError("Mass must be positive")
        
        # Validate shape type
        if not isinstance(shape_type, ShapeType):
            raise ValueError("Shape type must be a ShapeType enum value")
        
        # Add to lists
        self.positions.append(position)
        self.velocities.append(velocity)
        self.orientations.append(orientation)
        self.angular_vels.append(angular_vel)
        self.masses.append(float(mass))
        self.shape_types.append(shape_type)
        self.shape_params.append(shape_params)
        
        return self  # Enable method chaining
    
    def build(self) -> dict[str, np.ndarray]:
        """Build the scene and return SoA data format.
        
        Returns:
            Dictionary containing SoA tensors ready for TensorPhysicsEngine
        """
        if not self.positions:
            raise ValueError("Cannot build empty scene. Add at least one body first.")
        
        return create_soa_body_data(
            positions=self.positions,
            velocities=self.velocities,
            orientations=self.orientations,
            angular_vels=self.angular_vels,
            masses=self.masses,
            shape_types=self.shape_types,
            shape_params=self.shape_params
        )
    
    def count(self) -> int:
        """Return the number of bodies in the scene."""
        return len(self.positions)
    
    def clear(self) -> 'SceneBuilder':
        """Clear all bodies from the scene.
        
        Returns:
            Self for method chaining
        """
        self.positions.clear()
        self.velocities.clear()
        self.orientations.clear()
        self.angular_vels.clear()
        self.masses.clear()
        self.shape_types.clear()
        self.shape_params.clear()
        return self