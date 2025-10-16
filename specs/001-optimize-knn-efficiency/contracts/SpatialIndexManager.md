# API Contract: SpatialIndexManager

## Overview

The SpatialIndexManager provides efficient nearest neighbor queries for conformal KNN computations.

## Interface

### Constructor

```python
SpatialIndexManager(distance_metric: str = 'euclidean', k: int = 1)
```

**Parameters**:

- `distance_metric`: Distance metric for neighbor computations ('euclidean', 'manhattan', etc.)
- `k`: Number of neighbors to retrieve

### Methods

#### add_point(point: np.ndarray, label: Any) -> None

Add a training point with its label to the appropriate spatial index.

**Parameters**:

- `point`: Feature vector as numpy array
- `label`: Class label for the point

**Effects**:

- Stores point in label-grouped collection
- Marks indexes as needing rebuild

#### rebuild_indexes() -> None

Rebuild all spatial indexes after batch updates.

**Effects**:

- Constructs new spatial indexes for each label
- O(n log n) operation

#### query_neighbors(point: np.ndarray, label: Any, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]

Find k nearest neighbors of same and different labels.

**Parameters**:

- `point`: Query point
- `label`: Hypothetical label for nonconformity computation
- `k`: Number of neighbors (defaults to instance k)

**Returns**:

- `same_distances`: Distances to k nearest same-label neighbors
- `diff_distances`: Distances to k nearest different-label neighbors

**Preconditions**:

- Indexes must be built (call rebuild_indexes() if needed)

## Implementation Notes

- Uses scipy.spatial.KDTree for euclidean distances
- Falls back to sklearn BallTree for other metrics
- Maintains separate indexes per label for efficient within-class queries
