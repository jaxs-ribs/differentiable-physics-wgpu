import pytest
from tinygrad import Tensor, dtypes
import numpy as np

from physics.xpbd.broadphase import uniform_spatial_hash
from physics.xpbd.broadphase_efficient import (
    efficient_spatial_hash, compute_aabb_from_shape, compute_adaptive_cell_size,
    spatial_hash, build_count_table, compute_prefix_sum, sort_by_hash
)
from physics.xpbd.broadphase_optimized import (
    spatial_hash_optimized, spatial_hash_gpu_friendly, morton_encode_3d,
    compact_duplicate_pairs
)


class TestBroadphaseEfficient:
    def test_aabb_computation(self):
        """Test AABB computation from shapes."""
        x = Tensor([[0, 0, 0], [5, 5, 5], [10, 0, 0]], dtype=dtypes.float32)
        shape_type = Tensor([0, 0, 0], dtype=dtypes.int32)  # All spheres
        shape_params = Tensor([[1.0], [2.0], [0.5]], dtype=dtypes.float32)  # Radii
        
        aabb_min, aabb_max = compute_aabb_from_shape(x, shape_type, shape_params)
        
        expected_min = x.numpy() - shape_params.numpy()
        expected_max = x.numpy() + shape_params.numpy()
        
        np.testing.assert_allclose(aabb_min.numpy(), expected_min, rtol=1e-5)
        np.testing.assert_allclose(aabb_max.numpy(), expected_max, rtol=1e-5)
    
    def test_adaptive_cell_size(self):
        """Test adaptive cell size computation."""
        aabb_min = Tensor([[0, 0, 0], [4, 4, 4]], dtype=dtypes.float32)
        aabb_max = Tensor([[2, 2, 2], [6, 6, 6]], dtype=dtypes.float32)
        
        cell_size = compute_adaptive_cell_size(aabb_min, aabb_max)
        
        # Max AABB size is 2.0, multiplier is 2.0, so cell_size should be 4.0
        assert abs(cell_size - 4.0) < 1e-5
    
    def test_spatial_hash_deterministic(self):
        """Test that spatial hash is deterministic."""
        cell_ids = Tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3]], dtype=dtypes.int32)
        
        hash1 = spatial_hash(cell_ids, 1024)
        hash2 = spatial_hash(cell_ids, 1024)
        
        np.testing.assert_array_equal(hash1.numpy(), hash2.numpy())
        
        # Same cell should produce same hash
        assert hash1[0].numpy() == hash1[2].numpy()
    
    def test_count_table(self):
        """Test count table construction."""
        hash_keys = Tensor([0, 2, 5, 2, 0, 0], dtype=dtypes.int32)
        table_size = 8
        
        counts = build_count_table(hash_keys, table_size)
        
        expected = np.zeros(table_size, dtype=np.int32)
        expected[0] = 3  # Three bodies hash to 0
        expected[2] = 2  # Two bodies hash to 2
        expected[5] = 1  # One body hashes to 5
        
        np.testing.assert_array_equal(counts.numpy(), expected)
    
    def test_prefix_sum(self):
        """Test exclusive prefix sum computation."""
        counts = Tensor([3, 0, 2, 0, 0, 1, 0, 0], dtype=dtypes.int32)
        
        prefix = compute_prefix_sum(counts)
        
        expected = np.array([0, 3, 3, 5, 5, 5, 6, 6], dtype=np.int32)
        np.testing.assert_array_equal(prefix.numpy(), expected)
    
    def test_sort_by_hash(self):
        """Test sorting bodies by hash keys."""
        body_indices = Tensor([0, 1, 2, 3, 4], dtype=dtypes.int32)
        hash_keys = Tensor([5, 2, 8, 2, 5], dtype=dtypes.int32)
        
        sorted_bodies, sorted_hashes = sort_by_hash(body_indices, hash_keys)
        
        # Should be sorted by hash: [2, 2, 5, 5, 8]
        expected_hashes = np.array([2, 2, 5, 5, 8], dtype=np.int32)
        expected_bodies = np.array([1, 3, 0, 4, 2], dtype=np.int32)
        
        np.testing.assert_array_equal(sorted_hashes.numpy(), expected_hashes)
        np.testing.assert_array_equal(sorted_bodies.numpy(), expected_bodies)
    
    def test_efficient_spatial_hash_small(self):
        """Test efficient spatial hash with small example."""
        # Two bodies close together, one far away
        x = Tensor([[0, 0, 0], [0.5, 0, 0], [10, 10, 10]], dtype=dtypes.float32)
        shape_type = Tensor([0, 0, 0], dtype=dtypes.int32)
        shape_params = Tensor([[0.6], [0.6], [0.6]], dtype=dtypes.float32)
        
        pairs = efficient_spatial_hash(x, shape_type, shape_params, cell_size=2.0)
        
        # Bodies 0 and 1 should be paired (close together)
        # Body 2 should not pair with anyone (far away)
        assert pairs.shape[0] >= 1
        pair_set = set(tuple(p) for p in pairs.numpy())
        assert (0, 1) in pair_set or (1, 0) in pair_set
        assert not any((2 in p) for p in pair_set)
    
    def test_efficient_spatial_hash_grid(self):
        """Test with bodies arranged in a grid."""
        # 3x3 grid of bodies
        positions = []
        for i in range(3):
            for j in range(3):
                positions.append([i * 2.0, j * 2.0, 0])
        
        x = Tensor(positions, dtype=dtypes.float32)
        shape_type = Tensor([0] * 9, dtype=dtypes.int32)
        shape_params = Tensor([[0.5]] * 9, dtype=dtypes.float32)
        
        pairs = efficient_spatial_hash(x, shape_type, shape_params, cell_size=2.0)
        
        # Each body should only pair with adjacent neighbors
        # Corner bodies: 2 neighbors, Edge bodies: 3 neighbors, Center: 4 neighbors
        # Total pairs: 4*2/2 + 4*3/2 + 1*4/2 = 4 + 6 + 2 = 12
        assert pairs.shape[0] <= 12 * 2  # Some might be detected from both directions


class TestBroadphaseOptimized:
    def test_morton_encoding(self):
        """Test Morton encoding for spatial locality."""
        x = Tensor([1, 5, 10], dtype=dtypes.int32)
        y = Tensor([2, 6, 11], dtype=dtypes.int32)
        z = Tensor([3, 7, 12], dtype=dtypes.int32)
        
        morton = morton_encode_3d(x, y, z)
        
        # Morton codes should be unique for different coordinates
        morton_np = morton.numpy()
        assert len(np.unique(morton_np)) == len(morton_np)
    
    def test_compact_duplicate_pairs(self):
        """Test duplicate pair removal."""
        # Pairs with duplicates
        pairs = Tensor([[0, 1], [2, 3], [0, 1], [3, 2], [1, 0]], dtype=dtypes.int32)
        
        unique = compact_duplicate_pairs(pairs)
        
        # Should have 3 unique pairs: (0,1), (1,0), (2,3)
        # Note: (3,2) is same as (2,3) if we consider unordered pairs
        assert unique.shape[0] <= 4
    
    def test_spatial_hash_optimized(self):
        """Test optimized spatial hash."""
        # Simple test case
        x = Tensor([[0, 0, 0], [1, 0, 0], [5, 5, 5]], dtype=dtypes.float32)
        shape_type = Tensor([0, 0, 0], dtype=dtypes.int32)
        shape_params = Tensor([[1.0], [1.0], [1.0]], dtype=dtypes.float32)
        
        pairs = spatial_hash_optimized(x, shape_type, shape_params, cell_size=2.0)
        
        # Bodies 0 and 1 should be paired (overlapping AABBs)
        pair_set = set(tuple(p) for p in pairs.numpy())
        assert (0, 1) in pair_set
        
        # Body 2 should not pair with others (too far)
        assert not any((2 in p and p != (2, 2)) for p in pair_set)
    
    def test_gpu_friendly_version(self):
        """Test GPU-friendly spatial hash."""
        # Bodies in same cell
        x = Tensor([[0, 0, 0], [0.5, 0.5, 0.5], [0.8, 0.8, 0.8]], dtype=dtypes.float32)
        shape_type = Tensor([0, 0, 0], dtype=dtypes.int32)
        shape_params = Tensor([[0.3], [0.3], [0.3]], dtype=dtypes.float32)
        
        pairs = spatial_hash_gpu_friendly(x, shape_type, shape_params, cell_size=2.0)
        
        # All bodies in same cell should be paired
        assert pairs.shape[0] == 3  # (0,1), (0,2), (1,2)
    
    def test_performance_scaling(self):
        """Test that optimized version handles more bodies efficiently."""
        # Create 100 randomly distributed bodies
        np.random.seed(42)
        positions = np.random.uniform(-10, 10, (100, 3))
        x = Tensor(positions, dtype=dtypes.float32)
        shape_type = Tensor([0] * 100, dtype=dtypes.int32)
        shape_params = Tensor([[0.5]] * 100, dtype=dtypes.float32)
        
        # Should complete without memory issues
        pairs = spatial_hash_optimized(x, shape_type, shape_params, cell_size=2.0)
        
        # Verify some pairs exist but not all possible pairs
        assert 0 < pairs.shape[0] < 100 * 99 / 2