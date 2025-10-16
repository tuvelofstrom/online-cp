# Quickstart: KNN Efficiency Optimization

## Overview

This guide shows how to use the optimized KNN implementation for faster conformal prediction on larger datasets.

## Installation

The optimized implementation is included in the online-cp library. No additional dependencies required beyond scipy and numpy.

```python
pip install online-cp
```

## Basic Usage

### Import and Initialize

```python
from online_cp import ConformalNearestNeighboursClassifier

# Use optimized version (default for large datasets)
cp = ConformalNearestNeighboursClassifier(k=3, use_spatial_index=True)
```

### Training

```python
import numpy as np

# Generate sample data
X_train = np.random.randn(1000, 10)
y_train = np.random.choice([-1, 1], 1000)

# Train the model
cp.learn_initial_training_set(X_train, y_train)
```

### Prediction

```python
# Make predictions
X_test = np.random.randn(5, 10)

for x in X_test:
    prediction_set = cp.predict(x, epsilon=0.1)
    print(f"Prediction set: {prediction_set}")
```

## Performance Comparison

### Benchmarking Script

```python
import time
import numpy as np
from online_cp import ConformalNearestNeighboursClassifier

# Generate test data
sizes = [100, 500, 1000, 2000]
results = []

for n in sizes:
    X = np.random.randn(n, 5)
    y = np.random.choice([-1, 1], n)

    # Test optimized version
    cp_opt = ConformalNearestNeighboursClassifier(use_spatial_index=True)
    cp_opt.learn_initial_training_set(X, y)

    start = time.time()
    pred_opt = cp_opt.predict(X[0])
    time_opt = time.time() - start

    # Test original version
    cp_orig = ConformalNearestNeighboursClassifier(use_spatial_index=False)
    cp_orig.learn_initial_training_set(X, y)

    start = time.time()
    pred_orig = cp_orig.predict(X[0])
    time_orig = time.time() - start

    results.append({
        'size': n,
        'optimized': time_opt,
        'original': time_orig,
        'speedup': time_orig / time_opt
    })

    print(f"Size {n}: {time_orig:.4f}s → {time_opt:.4f}s ({results[-1]['speedup']:.1f}x speedup)")
```

### Expected Performance

- **Small datasets** (< 100 points): Similar performance
- **Medium datasets** (100-1000 points): 2-5x speedup
- **Large datasets** (> 1000 points): 10-50x speedup

## Configuration Options

### Distance Metrics

```python
# Euclidean (default, fastest)
cp = ConformalNearestNeighboursClassifier(distance='euclidean')

# Manhattan distance
cp = ConformalNearestNeighboursClassifier(distance='manhattan')

# Custom distance function (falls back to original implementation)
def custom_distance(a, b):
    return np.sum(np.abs(a - b))

cp = ConformalNearestNeighboursClassifier(distance_func=custom_distance)
```

### Tuning Parameters

```python
cp = ConformalNearestNeighboursClassifier(
    k=5,                           # Number of neighbors
    min_points_for_index=50,       # When to switch to spatial indexing
    epsilon=0.05,                  # Significance level
    n_jobs=4                       # Parallel prediction jobs
)
```

## Troubleshooting

### Common Issues

1. **Slow performance on small datasets**
   - The optimization activates at `min_points_for_index` (default 100)
   - For smaller datasets, use `use_spatial_index=False`

2. **Memory errors**
   - Ensure scipy is installed: `pip install scipy`
   - For very large datasets, consider batch processing

3. **Different results**
   - Check that `rnd_state` is set for reproducible comparisons
   - Verify distance metrics match between implementations

### Validation

```python
# Ensure identical results
cp_opt = ConformalNearestNeighboursClassifier(use_spatial_index=True, rnd_state=42)
cp_orig = ConformalNearestNeighboursClassifier(use_spatial_index=False, rnd_state=42)

# Train both
cp_opt.learn_initial_training_set(X, y)
cp_orig.learn_initial_training_set(X, y)

# Compare predictions
pred_opt = cp_opt.predict(x)
pred_orig = cp_orig.predict(x)

assert np.array_equal(pred_opt.elements, pred_orig.elements), "Predictions differ!"
```

## Advanced Usage

### Online Learning

```python
cp = ConformalNearestNeighboursClassifier()

# Sequential learning
for x, y in zip(X_stream, y_stream):
    # Predict before observing label
    pred = cp.predict(x)

    # Then learn the true label
    cp.learn_one(x, y)
```

### Parallel Prediction

```python
# Use multiple cores for prediction
cp = ConformalNearestNeighboursClassifier(n_jobs=-1)  # Use all cores

# Predictions are automatically parallelized across label hypotheses
pred = cp.predict(x)
```
