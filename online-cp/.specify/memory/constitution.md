# online-cp Constitution

## Core Principles

### I. Strict Adherence to Conformal Prediction Theory

All implementations in online-cp strictly follow the mathematical foundations of conformal prediction as established in the literature. Key guarantees include coverage at (1 - ε) under exchangeability, distribution-free validity, and finite-sample guarantees. No approximations or shortcuts are allowed that compromise these theoretical properties.

### II. Fully Online Learning

The library supports sequential, online learning paradigms with incremental updates using learn_one() methods. No batch retraining is required, and memory usage remains bounded for streaming data. Predictions can be made without full historical data access.

### III. Fully Transductive Prediction

All prediction methods are transductive: prediction sets are generated for specific test instances before observing their labels. Each prediction set is instance-specific and tailored to the particular object, avoiding assumptions about future data distribution.

### IV. Comprehensive Evaluation Framework

Provides rigorous tools for evaluating conformal predictors including efficiency metrics (observed excess, observed fuzziness, error rates, widths), exchangeability testing via conformal test martingales, and cumulative performance monitoring over prediction sequences.

### V. Modularity and Extensibility

Supports various underlying algorithms (ridge regression, nearest neighbors, kernels) in an algorithm-agnostic manner. Allows customization of nonconformity measures, kernels, and parameters. Designed for research and experimentation with conformal prediction methods.

## Implementation Commitments

- **Correctness**: Algorithms implement theoretical guarantees without approximation
- **Efficiency**: Optimized for online settings with minimal computational overhead
- **Reproducibility**: Comprehensive testing and documentation for reliable research use
- **Open Source**: Freely available to promote accessibility and collaboration

## References

Vovk, V., Gammerman, A., & Shafer, G. (2022). Algorithmic Learning in a Random World (2nd ed.). Springer Nature.

## Governance

Constitution supersedes all other practices. Amendments require documentation, approval, and migration plan. All PRs/reviews must verify compliance. Complexity must be justified.

**Version**: 1.0.0 | **Ratified**: 2025-10-16 | **Last Amended**: 2025-10-16
