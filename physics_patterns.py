"""
Physics Pattern Matcher for TinyGrad
Recognizes high-level physics operations and transforms them to CUSTOM ops
"""
import ctypes
import os
from pathlib import Path
from typing import Optional

# Add parent directory to path to import tinygrad
import sys
sys.path.append(str(Path(__file__).parent.parent))

from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher
from tinygrad.dtype import dtypes

# Load the physics library
def load_physics_library():
    """Load the compiled physics shared library"""
    lib_path = Path(__file__).parent
    lib_name = "libphysics.dylib" if sys.platform == "darwin" else "libphysics.so"
    lib_file = lib_path / lib_name
    
    if not lib_file.exists():
        raise RuntimeError(f"Physics library not found at {lib_file}. Run 'make' first.")
    
    lib = ctypes.CDLL(str(lib_file))
    
    # Define function signatures
    # physics_step(float* bodies, int32_t num_bodies, float dt, float* output)
    lib.physics_step.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.physics_step.restype = None
    
    return lib

# Global physics library instance
PHYSICS_LIB = None

def get_physics_lib():
    global PHYSICS_LIB
    if PHYSICS_LIB is None:
        PHYSICS_LIB = load_physics_library()
    return PHYSICS_LIB

# Define custom physics ops that we'll recognize
class PhysicsOps:
    """Marker class for physics operations"""
    PHYSICS_STEP = "PHYSICS_STEP"
    PHYSICS_INTEGRATE = "PHYSICS_INTEGRATE"
    PHYSICS_COLLIDE = "PHYSICS_COLLIDE"

def create_physics_patterns():
    """
    Create pattern matcher for physics operations.
    This will recognize specific patterns and convert them to CUSTOM ops.
    """
    patterns = []
    
    # Pattern: Recognize a physics step operation
    # This is a placeholder - in a real implementation, we'd recognize
    # specific computation patterns that match physics operations
    
    # For now, we'll create a simple pattern that recognizes a special marker
    # In practice, this would recognize patterns like:
    # - Velocity integration: pos = pos + vel * dt
    # - Force application: vel = vel + (force / mass) * dt
    # - Collision detection: distance calculations between bodies
    
    def physics_step_rewriter(ctx, x):
        """
        Rewrite a physics step pattern to a CUSTOM op
        This function is called when the pattern is matched
        """
        # Extract the source tensors from the matched pattern
        bodies_tensor = x.src[0]  # Input bodies tensor
        dt = x.arg  # Time step
        
        # Create a CUSTOM op that calls our C function
        # The format string will be used to generate the C code
        format_str = f"physics_step_wrapper({{0}}, {dt}, {{1}})"
        
        return UOp(Ops.CUSTOM, x.dtype, (bodies_tensor,), format_str)
    
    # Add pattern for physics step marker
    # In a real implementation, this would match actual computation patterns
    patterns.append((
        UPat(Ops.CUSTOM, name="x", arg=PhysicsOps.PHYSICS_STEP),
        physics_step_rewriter
    ))
    
    return PatternMatcher(patterns)

def create_physics_renderer_extensions():
    """
    Create additional patterns for the renderer to handle physics CUSTOM ops
    """
    patterns = []
    
    def render_physics_custom(ctx, x):
        """Render physics CUSTOM ops to actual C function calls"""
        if "physics_step_wrapper" in x.arg:
            # Generate the wrapper function that handles the tensor->array conversion
            ctx["physics_includes"] = """
#include <string.h>
extern void physics_step(float* bodies, int num_bodies, float dt, float* output);

static void physics_step_wrapper(float* input, float dt, float* output) {
    // Assume the first dimension is the number of bodies
    // In a real implementation, we'd pass this information properly
    int num_bodies = 100; // This should come from tensor shape
    physics_step(input, num_bodies, dt, output);
}
"""
            # Return the actual function call
            return x.arg.format(*[ctx[y] for y in x.src])
        return None
    
    patterns.append((
        UPat(Ops.CUSTOM, name="x"),
        render_physics_custom
    ))
    
    return PatternMatcher(patterns)

# Higher-level API for marking physics operations
def physics_step(bodies_tensor, dt: float):
    """
    Mark a tensor operation as a physics step.
    This creates a special UOp that will be recognized by our pattern matcher.
    
    Args:
        bodies_tensor: Tensor containing rigid body data [N, 8]
                      Format: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, mass, radius]
        dt: Time step for integration
    
    Returns:
        UOp marked for physics processing
    """
    # In a real implementation, this would be integrated with Tensor operations
    # For now, we create a marker UOp
    return UOp(Ops.CUSTOM, bodies_tensor.dtype, (bodies_tensor,), PhysicsOps.PHYSICS_STEP)