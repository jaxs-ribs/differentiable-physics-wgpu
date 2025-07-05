from tinygrad import dtypes

# Hash table configuration
HASH_TABLE_SIZE = 8192
MAX_BODIES_PER_CELL = 32
DEFAULT_CELL_SIZE = 2.0

# Domain offset - consider removing or making configurable
BIAS = 1000.0

# Prime numbers for spatial hashing
PRIME_X = 73856093
PRIME_Y = 19349663
PRIME_Z = 83492791

# Neighbor offsets for 3D grid (27 cells including center)
NEIGHBOR_OFFSETS = [
    (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
    (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
    (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
    (0, -1, -1), (0, -1, 0), (0, -1, 1),
    (0, 0, -1), (0, 0, 0), (0, 0, 1),
    (0, 1, -1), (0, 1, 0), (0, 1, 1),
    (1, -1, -1), (1, -1, 0), (1, -1, 1),
    (1, 0, -1), (1, 0, 0), (1, 0, 1),
    (1, 1, -1), (1, 1, 0), (1, 1, 1)
]

# Configuration for adaptive cell sizing
MIN_CELL_SIZE = 0.1
MAX_CELL_SIZE = 10.0
CELL_SIZE_MULTIPLIER = 2.0  # Cell size = largest_aabb * multiplier

# Fixed-size contact buffer to avoid JIT cache invalidation
MAX_CONTACTS_PER_STEP = 64  # Maximum number of contacts in the fixed-size buffer