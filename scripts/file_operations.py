import numpy as np
from pathlib import Path
from datetime import datetime


def ensure_directory_exists(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


def generate_timestamped_filename(prefix: str = "simulation", extension: str = "mp4") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{timestamp}.{extension}"


def save_numpy_array(array: np.ndarray, filepath: Path) -> None:
    ensure_directory_exists(filepath.parent)
    np.save(filepath, array)