# Research: KNN Efficiency Optimization

## Research Tasks

### Task 1: Spatial Indexing Options for Python

**Query**: What are the best spatial indexing libraries for nearest neighbor queries in Python that support exact distance computations?

**Findings**:

- **scipy.spatial.KDTree**: Built-in, supports exact k-nearest neighbor queries with euclidean and other metrics
- **sklearn.neighbors.BallTree**: More flexible distance metrics, better for high dimensions
- **pynndescent**: Approximate NN, not suitable for exact CP requirements
- **faiss**: High performance but approximate, GPU-accelerated

**Decision**: Use scipy.spatial.KDTree for euclidean distance (most common), with sklearn BallTree as fallback for other metrics
**Rationale**: Maintains exact computations required for conformal guarantees, good performance for typical use cases
**Alternatives considered**: Approximate methods (too slow convergence for CP), custom implementations (higher maintenance)

### Task 2: Online KD-Tree Updates

**Query**: How to handle incremental updates to spatial indexes in online learning scenarios?

**Findings**:

- KD-trees don't support efficient single-point insertions/deletions
- Common approaches: Periodic rebuilds, maintain separate structures per class
- For KNN classification, can maintain one KD-tree per label class
- Rebuilding cost is O(n log n) which is acceptable for online settings

**Decision**: Maintain separate KD-trees for each label class, rebuild when new points are added
**Rationale**: Allows efficient queries within classes, rebuild cost is manageable for online learning
**Alternatives considered**: Single tree with class filtering (slower queries), incremental tree libraries (limited Python options)

### Task 3: Memory and Performance Trade-offs

**Query**: What are the memory implications and performance characteristics of spatial indexing vs distance matrix?

**Findings**:

- Distance matrix: O(n²) memory, O(n²) build, O(n) queries (but with sorting overhead)
- KD-tree: O(n) memory, O(n log n) build, O(log n) queries
- Break-even point around n=1000 for typical dimensions
- Memory usage: KD-tree much more efficient for large n

**Decision**: Use spatial indexing for n > 100, fall back to distance matrix for small datasets
**Rationale**: Optimizes for the target use case of larger datasets while maintaining performance for small ones
**Alternatives considered**: Always use spatial indexing (slower for very small n), hybrid approaches (added complexity)

### Task 4: Distance Metric Compatibility

**Query**: Which distance metrics are supported by spatial indexes and how to handle unsupported ones?

**Findings**:

- KDTree: euclidean, manhattan, chebyshev, minkowski
- BallTree: All sklearn metrics including cosine, hamming, etc.
- Custom metrics require fallback to distance matrix approach

**Decision**: Detect metric type and choose appropriate index, fall back to distance matrix for custom/unsupported metrics
**Rationale**: Maximizes performance while maintaining full compatibility
**Alternatives considered**: Force euclidean-only (breaks existing API), implement custom spatial indexes (high complexity)

## Integration Patterns

### Pattern 1: Dual Implementation Strategy

Maintain both optimized and original implementations with a flag to switch between them for:

- Performance comparison during development
- Gradual rollout and validation
- Fallback for edge cases

### Pattern 2: Lazy Index Building

Build spatial indexes only when needed (first prediction), rebuild only when dataset changes significantly.

### Pattern 3: Caching Strategy

Cache computed nonconformity scores and neighbor distances to avoid recomputation across different label hypotheses.
