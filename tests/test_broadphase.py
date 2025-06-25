#!/usr/bin/env python3
import numpy as np
import itertools

def test_uniform_grid_theory():
    """Test uniform grid broad phase algorithm"""
    print("Testing uniform grid broad phase theory...")
    
    # Grid parameters
    cell_size = 2.0  # 2m cells
    grid_size = 10   # 10x10x10 grid
    
    # Test bodies with bounding boxes
    bodies = [
        {"id": 0, "aabb_min": [0, 0, 0], "aabb_max": [1, 1, 1]},      # Cell (0,0,0)
        {"id": 1, "aabb_min": [0.5, 0, 0], "aabb_max": [1.5, 1, 1]},  # Cells (0,0,0) and (1,0,0)
        {"id": 2, "aabb_min": [5, 5, 5], "aabb_max": [6, 6, 6]},      # Cell (2,2,2)
        {"id": 3, "aabb_min": [5.5, 5, 5], "aabb_max": [6.5, 6, 6]},  # Cells (2,2,2) and (3,2,2)
    ]
    
    # Calculate which cells each body occupies
    def get_cells(aabb_min, aabb_max):
        min_cell = [int(aabb_min[i] / cell_size) for i in range(3)]
        max_cell = [int(aabb_max[i] / cell_size) for i in range(3)]
        cells = []
        for x in range(min_cell[0], max_cell[0] + 1):
            for y in range(min_cell[1], max_cell[1] + 1):
                for z in range(min_cell[2], max_cell[2] + 1):
                    cells.append((x, y, z))
        return cells
    
    # Build grid
    grid = {}
    for body in bodies:
        cells = get_cells(body["aabb_min"], body["aabb_max"])
        for cell in cells:
            if cell not in grid:
                grid[cell] = []
            grid[cell].append(body["id"])
    
    print(f"Grid cells occupied: {len(grid)}")
    for cell, ids in grid.items():
        print(f"  Cell {cell}: bodies {ids}")
    
    # Find potential pairs
    potential_pairs = set()
    for cell_bodies in grid.values():
        for i in range(len(cell_bodies)):
            for j in range(i + 1, len(cell_bodies)):
                pair = tuple(sorted([cell_bodies[i], cell_bodies[j]]))
                potential_pairs.add(pair)
    
    print(f"\nPotential collision pairs: {potential_pairs}")
    
    # Calculate efficiency
    total_pairs = len(bodies) * (len(bodies) - 1) // 2
    pruned_pairs = total_pairs - len(potential_pairs)
    prune_rate = pruned_pairs / total_pairs * 100
    
    print(f"\nTotal possible pairs: {total_pairs}")
    print(f"Potential pairs after broad phase: {len(potential_pairs)}")
    print(f"Pairs pruned: {pruned_pairs} ({prune_rate:.1f}%)")
    
    # Verify correct pairs
    expected_pairs = {(0, 1), (2, 3)}  # Bodies that are actually close
    assert potential_pairs == expected_pairs
    print("✓ Correct pairs identified")

def test_grid_efficiency():
    """Test broad phase efficiency with many bodies"""
    print("\nTesting broad phase efficiency...")
    
    # Create a sparse distribution of bodies
    np.random.seed(42)
    num_bodies = 1000
    world_size = 100.0
    body_radius = 0.5
    cell_size = 2.0  # Should be ~2x max body size
    
    bodies = []
    for i in range(num_bodies):
        # Random position
        pos = np.random.uniform(0, world_size, 3)
        bodies.append({
            "id": i,
            "pos": pos,
            "aabb_min": pos - body_radius,
            "aabb_max": pos + body_radius
        })
    
    # Build spatial hash grid
    grid = {}
    for body in bodies:
        min_cell = tuple(int(body["aabb_min"][i] / cell_size) for i in range(3))
        max_cell = tuple(int(body["aabb_max"][i] / cell_size) for i in range(3))
        
        # For small bodies, usually just one cell
        for x in range(min_cell[0], max_cell[0] + 1):
            for y in range(min_cell[1], max_cell[1] + 1):
                for z in range(min_cell[2], max_cell[2] + 1):
                    cell = (x, y, z)
                    if cell not in grid:
                        grid[cell] = []
                    grid[cell].append(body["id"])
    
    # Count potential pairs
    potential_pairs = set()
    for cell_bodies in grid.values():
        for i in range(len(cell_bodies)):
            for j in range(i + 1, len(cell_bodies)):
                pair = tuple(sorted([cell_bodies[i], cell_bodies[j]]))
                potential_pairs.add(pair)
    
    # Calculate efficiency
    total_pairs = num_bodies * (num_bodies - 1) // 2
    pruned_pairs = total_pairs - len(potential_pairs)
    prune_rate = pruned_pairs / total_pairs * 100
    
    print(f"Bodies: {num_bodies}")
    print(f"Grid cells used: {len(grid)}")
    print(f"Total possible pairs: {total_pairs:,}")
    print(f"Potential pairs after broad phase: {len(potential_pairs):,}")
    print(f"Pairs pruned: {pruned_pairs:,} ({prune_rate:.1f}%)")
    
    # For sparse distribution, should prune >90% of pairs
    assert prune_rate > 90.0, f"Pruning rate too low: {prune_rate:.1f}%"
    print("✓ Broad phase achieves >90% pruning")

def test_grid_parameters():
    """Test optimal grid parameters"""
    print("\nTesting grid parameter selection...")
    
    # Rule of thumb: cell size = 2-3x maximum object size
    max_object_radius = 1.0
    optimal_cell_size = 2.5 * max_object_radius
    
    print(f"Max object radius: {max_object_radius} m")
    print(f"Optimal cell size: {optimal_cell_size} m")
    
    # Test with different cell sizes
    test_sizes = [1.0, 2.0, 2.5, 5.0, 10.0]
    
    for cell_size in test_sizes:
        # Simulate efficiency
        # Too small = many cells per object
        # Too large = many objects per cell
        cells_per_object = (2 * max_object_radius / cell_size) ** 3
        avg_objects_per_cell = 10 * (cell_size / optimal_cell_size) ** 3
        
        efficiency = 1.0 / (cells_per_object + avg_objects_per_cell)
        
        print(f"  Cell size {cell_size}: efficiency score {efficiency:.3f}")
    
    print("✓ Grid parameters analyzed")

if __name__ == "__main__":
    print("Running broad phase tests...\n")
    
    test_uniform_grid_theory()
    test_grid_efficiency()
    test_grid_parameters()
    
    print("\n✓ All broad phase tests passed!")