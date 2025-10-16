# API Contract: OptimizedConformalKNNClassifier

## Overview

Enhanced version of ConformalNearestNeighboursClassifier with spatial indexing for improved performance.

## Interface

### Constructor

```python
OptimizedConformalKNNClassifier(
    k: int = 1,
    label_space: np.ndarray = np.array([-1, 1]),
    distance: str = 'euclidean',
    distance_func: Optional[Callable] = None,
    verbose: int = 0,
    rnd_state: Optional[int] = None,
    n_jobs: Optional[int] = None,
    epsilon: float = 0.1,
    use_spatial_index: bool = True,
    min_points_for_index: int = 100
)
```

**Parameters**:
- `k`: Number of neighbors for nonconformity measure
- `label_space`: Array of possible class labels
- `distance`: Distance metric ('euclidean', 'manhattan', etc.)
- `distance_func`: Custom distance function (disables spatial indexing if provided)
- `verbose`: Verbosity level
- `rnd_state`: Random state for reproducibility
- `n_jobs`: Number of parallel jobs for prediction
- `epsilon`: Significance level for conformal prediction
- `use_spatial_index`: Whether to use optimized spatial indexing
- `min_points_for_index`: Minimum dataset size to enable spatial indexing

### Methods

#### learn_initial_training_set(X: np.ndarray, y: np.ndarray) -> None

Initialize with batch training data.

**Parameters**:
- `X`: Training features (n_samples, n_features)
- `y`: Training labels (n_samples,)

#### learn_one(x: np.ndarray, y: Any, D: Optional[np.ndarray] = None) -> None

Add a single training example.

**Parameters**:
- `x`: Feature vector
- `y`: Class label
- `D`: Precomputed distance matrix (for compatibility)

#### predict(x: np.ndarray, epsilon: Optional[float] = None, return_p_values: bool = False, return_update: bool = False, verbose: int = 0) -> Union[ConformalPredictionSet, Tuple]

Generate prediction set for test instance.

**Parameters**:
- `x`: Test feature vector
- `epsilon`: Significance level (overrides instance epsilon)
- `return_p_values`: Whether to return p-values
- `return_update`: Whether to return updated distance matrix
- `verbose`: Verbosity level

**Returns**:
- Prediction set, optionally with p-values and updated matrix

## Compatibility

### Backward Compatibility
- Same method signatures as ConformalNearestNeighboursClassifier
- Identical prediction outputs for same inputs
- Graceful fallback to original implementation when spatial indexing unavailable

### Performance Modes
- **Spatial Index Mode**: O(log n) queries for large datasets
- **Fallback Mode**: Original O(n²) implementation for small datasets or unsupported metrics

## Implementation Requirements

- Exact distance computations matching original implementation
- Identical nonconformity scores and p-values
- Thread-safe for parallel prediction across label hypotheses
