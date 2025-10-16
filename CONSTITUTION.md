# Constitution of online-cp

## Purpose

The `online-cp` library provides a comprehensive implementation of Online Conformal Prediction (CP) for uncertainty quantification in machine learning. It enables users to generate prediction sets with guaranteed coverage properties for both regression and classification tasks, while operating in a fully online and transductive manner.

## Core Principles

### 1. Strict Adherence to Conformal Prediction Theory

All implementations in `online-cp` strictly follow the mathematical foundations of conformal prediction as established in the literature (Vovk et al., 2022). Key guarantees include:

- **Coverage Guarantee**: Prediction sets achieve (1 - ε) coverage under the exchangeability assumption, where ε is the user-specified significance level.
- **Distribution-Free**: No assumptions about the underlying data distribution beyond exchangeability.
- **Finite-Sample Validity**: Guarantees hold for any finite sample size.

### 2. Fully Online Learning

The library supports sequential, online learning paradigms:

- **Incremental Updates**: Models update with each new example using `learn_one()` methods.
- **No Batch Requirements**: Predictions can be made without requiring full retraining on historical data.
- **Memory Efficiency**: Designed to handle streaming data with bounded memory usage.

### 3. Fully Transductive Prediction

All prediction methods are transductive in nature:

- **Prediction Before Labeling**: Generate prediction sets for new objects before observing their true labels.
- **Instance-Specific**: Each prediction set is tailored to the specific test instance.
- **Non-Parametric**: Avoids assumptions about future data distribution.

### 4. Comprehensive Evaluation Framework

The library provides tools for rigorous evaluation of conformal predictors:

- **Efficiency Metrics**: Tracks observed excess (OE), observed fuzziness (OF), error rates, and prediction set widths.
- **Exchangeability Testing**: Implements conformal test martingales to detect violations of the exchangeability assumption.
- **Cumulative Monitoring**: Supports tracking performance metrics over time series of predictions.

### 5. Modularity and Extensibility

- **Algorithm Agnostic**: Supports various underlying prediction algorithms (ridge regression, nearest neighbors, kernel methods).
- **Configurable Components**: Allows customization of nonconformity measures, kernels, and regularization parameters.
- **Research-Oriented**: Designed to facilitate experimentation and extension of conformal prediction methods.

## Implementation Commitments

- **Correctness**: All algorithms implement the theoretical guarantees of conformal prediction without approximation.
- **Efficiency**: Optimized for online settings with minimal computational overhead.
- **Reproducibility**: Includes comprehensive testing and documentation for reliable research use.
- **Open Source**: Freely available under appropriate licensing to promote accessibility and collaboration.

## References

Vovk, V., Gammerman, A., & Shafer, G. (2022). Algorithmic Learning in a Random World (2nd ed.). Springer Nature.