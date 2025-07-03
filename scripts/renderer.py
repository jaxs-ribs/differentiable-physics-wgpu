import subprocess
import numpy as np
from pathlib import Path


class RendererInvoker:
    def __init__(self):
        self.renderer_path = self._find_renderer()
    
    def _find_renderer(self) -> Path:
        physics_core_dir = Path(__file__).parent.parent
        renderer_path = physics_core_dir / "renderer" / "target" / "release" / "renderer"
        
        if not renderer_path.exists():
            raise RuntimeError(
                f"Renderer executable not found at {renderer_path}\n"
                "Please build the renderer first: cd renderer && cargo build --release"
            )
        
        return renderer_path
    
    def render_video(self, trajectory_path: Path, output_path: Path, 
                    duration: float, fps: int = 30, verbose: bool = False) -> bool:
        # Load and format trajectory for renderer
        trajectory = np.load(trajectory_path)
        formatted_trajectory = self._format_for_renderer(trajectory)
        
        # Save formatted trajectory to temp file
        temp_trajectory_path = trajectory_path.with_suffix('.formatted.npy')
        np.save(temp_trajectory_path, formatted_trajectory)
        
        # Build command
        cmd = [
            str(self.renderer_path),
            str(temp_trajectory_path),
            str(output_path),
            "--duration", str(duration),
            "--fps", str(fps),
            "--mode", "oracle"  # Use oracle backend by default
        ]
        
        if verbose:
            print(f"Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if verbose:
                if result.stdout:
                    print("Renderer output:", result.stdout)
                if result.stderr:
                    print("Renderer errors:", result.stderr)
            
            success = result.returncode == 0
            
            # Clean up temp file
            try:
                temp_trajectory_path.unlink()
            except:
                pass
            
            return success
            
        except subprocess.TimeoutExpired:
            print("Renderer timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"Renderer execution failed: {e}")
            return False
    
    def _format_for_renderer(self, trajectory: np.ndarray) -> np.ndarray:
        frames, bodies, _ = trajectory.shape
        
        # Extract the 18 properties the renderer expects per body
        formatted = np.zeros((frames, bodies, 18), dtype=np.float32)
        
        # Copy relevant properties (positions, velocities, orientation, etc.)
        formatted[:, :, 0:3] = trajectory[:, :, 0:3]    # position
        formatted[:, :, 3:6] = trajectory[:, :, 3:6]    # velocity  
        formatted[:, :, 6:10] = trajectory[:, :, 6:10]  # quaternion
        formatted[:, :, 10:13] = trajectory[:, :, 10:13] # angular velocity
        formatted[:, :, 13] = trajectory[:, :, 13]       # inv_mass
        formatted[:, :, 14] = trajectory[:, :, 23]       # shape_type
        formatted[:, :, 15:18] = trajectory[:, :, 24:27] # shape_params
        
        # Flatten to (frames, bodies * 18)
        return formatted.reshape(frames, -1)