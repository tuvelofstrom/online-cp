# Data Model: KNN Efficiency Optimization

## Overview

The optimized KNN implementation replaces the dense distance matrix with spatial indexing structures to achieve better scalability while maintaining exact conformal prediction guarantees.

## Core Data Structures

### SpatialIndexManager

Manages spatial indexes for efficient nearest neighbor queries.

**Attributes**:

- `indexes`: Dict[label, SpatialIndex] - One spatial index per label class
- `points`: Dict[label, List[np.ndarray]] - Training points grouped by label
- `distance_metric`: str - Distance metric used for indexing
- `k`: int - Number of neighbors for nonconformity computation

**Methods**:

- `add_point(point, label)`: Add a new training point to appropriate index
- `rebuild_indexes()`: Rebuild all spatial indexes after batch updates
- `query_neighbors(point, label, k)`: Find k nearest neighbors for given label hypothesis

### OptimizedConformalKNNClassifier

Enhanced version of ConformalNearestNeighboursClassifier with spatial indexing.

**Inherits from**: ConformalClassifier

**Key Changes**:

- Replaces `self.D` (distance matrix) with `self.index_manager` (SpatialIndexManager)
- Maintains same API for backward compatibility
- Adds `use_spatial_index` parameter to switch implementations

**New Attributes**:

- `index_manager`: SpatialIndexManager instance
- `use_spatial_index`: bool - Whether to use optimized implementation
- `min_points_for_index`: int - Minimum dataset size to use spatial indexing

## Data Flow

### Training Phase

1. Points stored in label-grouped collections
2. Spatial indexes built lazily (on first prediction or when threshold reached)
3. Indexes rebuilt incrementally as new points are added

### Prediction Phase

1. For each label hypothesis:
   - Query spatial index for k nearest same-label neighbors
   - Query spatial index for k nearest different-label neighbors
   - Compute nonconformity score from neighbor distances
2. Calculate p-values and construct prediction set

## Performance Characteristics

### Time Complexity

- **Build**: O(n log n) for spatial index construction
- **Update**: O(log n) amortized for single point addition (with periodic rebuilds)
- **Query**: O(log n) for k-nearest neighbor search
- **Overall Prediction**: O(|Y| * log n) where |Y| is number of classes

### Space Complexity

- **Spatial Index**: O(n) storage
- **Points Storage**: O(n * d) where d is feature dimension
- **Total**: O(n * d) vs O(n²) for distance matrix approach

## Compatibility Layer

### Fallback Mechanisms

- Small datasets (< min_points_for_index): Use original distance matrix approach
- Unsupported distance metrics: Fall back to distance matrix
- Custom distance functions: Use distance matrix with function evaluation

### API Preservation

- All existing methods maintain same signatures
- Internal optimizations transparent to users
- Optional `use_spatial_index` parameter for explicit control

## Validation

### Correctness Checks

- Distance computations must match original implementation exactly
- Nonconformity scores must be identical
- Prediction sets and p-values must be identical

### Performance Benchmarks

- Measure prediction time vs dataset size
- Compare memory usage
- Validate scaling behavior (O(log n) vs O(n²))
