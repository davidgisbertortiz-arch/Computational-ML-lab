# Module 01: Numerical Toolbox

**Status**: ✅ Complete

Build a strong foundation in numerical methods that directly underpin machine learning training algorithms. This module emphasizes the geometry, conditioning, and stability considerations essential for robust ML implementations.

---

## 📓 Interactive Notebooks

**Educational notebooks** (recommended order):

1. **[Conditioning and Scaling](notebooks/01_conditioning_and_scaling.ipynb)**
   - What is condition number and why it matters
   - Visual intuition: elliptical loss contours
   - Feature scaling reduces κ → faster convergence
   - Learning rate stability limits

2. **[Gradient Descent Dynamics](notebooks/02_gradient_descent_dynamics.ipynb)**
   - Compare GD vs Momentum vs Adam
   - Experiments on quadratics and logistic regression
   - Learning rate sensitivity analysis
   - When to use each optimizer

3. **[Ridge as Numerical Stabilizer](notebooks/03_ridge_as_numerical_stabilizer.ipynb)**
   - Ridge from two angles: regularization + stabilization
   - Coefficient stability under noise
   - Regularization path visualization
   - Choosing λ via cross-validation

4. **[SVD/PCA Geometry](notebooks/04_svd_pca_geometry.ipynb)**
   - PCA as geometric projection
   - Explained variance and reconstruction
   - Implementation from scratch vs sklearn
   - Dimensionality reduction trade-offs

Each notebook includes:
- ✅ Intuitive explanations with 2-5 bullet points
- ✅ Minimal math derivations (only what's needed)
- ✅ From-scratch implementation + library comparison
- ✅ Experiments with plots
- ✅ 3-6 exercises with solutions
- ✅ Reports saved to `reports/`

---

## 📚 What You'll Learn

- Numerical conditioning and its impact on gradient-based optimization
- Why scaling and preconditioning matter for convergence
- Gradient descent, momentum, and adaptive methods (Adam) from scratch
- SVD/PCA as geometric transformations
- Bias-variance tradeoff from a numerical stability perspective
- Closed-form vs iterative solutions: when to use each

---

## 🎯 Learning Objectives

By completing this module, you will:

1. Understand condition numbers and their impact on optimization convergence
2. Implement gradient descent, momentum, and Adam optimizers from scratch
3. Perform PCA via SVD and interpret explained variance
4. Diagnose ill-conditioned problems and apply ridge regularization
5. Compare closed-form vs gradient-based solutions for linear regression
6. Visualize convergence behavior across different problem geometries

---

## 📖 Theory

### Conditioning and Stability

**Condition Number**: $\kappa(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}$

The condition number measures how sensitive a system is to input perturbations:
- $\kappa \approx 1$: Well-conditioned (round objective, fast convergence)
- $\kappa \gg 1$: Ill-conditioned (elongated objective, slow convergence)

**Why it matters for ML**:
- Poorly scaled features → high condition number
- High condition number → slow gradient descent convergence
- Solution: Feature scaling, preconditioning, or adaptive learning rates

### Gradient-Based Optimization

#### Gradient Descent (GD)
$$\theta_{t+1} = \theta_t - \alpha \nabla f(\theta_t)$$

- Simplest first-order method
- Convergence rate depends on $\kappa$: $O(\kappa \log(1/\epsilon))$ iterations
- Struggles with ill-conditioned objectives

#### Momentum
$$v_{t+1} = \beta v_t + \nabla f(\theta_t)$$
$$\theta_{t+1} = \theta_t - \alpha v_{t+1}$$

- Accumulates velocity from past gradients
- Accelerates convergence in consistent directions
- Dampens oscillations in narrow valleys

#### Adam (Adaptive Moment Estimation)
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla f(\theta_t)$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) [\nabla f(\theta_t)]^2$$
$$\hat{m}_t = m_t / (1-\beta_1^t), \quad \hat{v}_t = v_t / (1-\beta_2^t)$$
$$\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

- Adapts learning rate per parameter
- Uses first moment (mean) and second moment (variance)
- Less sensitive to initial learning rate and conditioning

### SVD and PCA

**Singular Value Decomposition**: $X = U \Sigma V^T$

- $U$: left singular vectors (data space basis)
- $\Sigma$: singular values (scaling factors)
- $V$: right singular vectors (feature space basis)

**PCA via SVD**:
1. Center data: $\tilde{X} = X - \bar{X}$
2. Compute SVD: $\tilde{X} = U \Sigma V^T$
3. Principal components: columns of $V$
4. Projection: $Z = XV$ (scores)
5. Reconstruction: $\hat{X} = ZV^T + \bar{X}$

**Explained Variance**: 
$$\text{Var explained by PC}_k = \frac{\sigma_k^2}{\sum_i \sigma_i^2}$$

**Geometry**: PCA finds orthogonal axes aligned with maximum variance directions.

### Bias-Variance from Numerical Perspective

**Bias**: Error from model assumptions
- Underfitting: high bias (too simple model)
- Numerical perspective: approximation error

**Variance**: Sensitivity to training data
- Overfitting: high variance (too complex model)
- Numerical perspective: condition-number dependent

**Ridge Regularization**: $\min_\theta \|X\theta - y\|^2 + \lambda \|\theta\|^2$
- Adds $\lambda I$ to $X^TX$, improving condition number
- Trades bias for reduced variance
- Solution: $\theta = (X^TX + \lambda I)^{-1} X^T y$

---

## 🛠️ Implementation Plan

### Part 1: Optimizers from Scratch ✅
- [x] Gradient Descent with line search
- [x] Momentum optimizer
- [x] Adam optimizer
- [x] Convergence tracking and logging

### Part 2: Linear Algebra ✅
- [x] PCA via SVD
- [x] Explained variance calculation
- [x] Reconstruction error
- [x] Ill-conditioning demo
- [x] Ridge stabilization

### Part 3: Toy Problems ✅
- [x] Quadratic bowls with varying condition numbers
- [x] Linear regression: closed-form vs GD
- [x] Convergence visualization

### Part 4: Experiments ✅
- [x] Optimizer comparison on well/ill-conditioned problems
- [x] PCA on synthetic data
- [x] Ridge regularization effects

---

## 🧪 Experiments & Metrics

### Experiment 1: Optimizer Convergence vs Conditioning

**Goal**: Demonstrate how condition number affects convergence

**Setup**:
- Quadratic objective: $f(\theta) = \frac{1}{2}\theta^T A \theta$
- Vary condition number: $\kappa = 1, 10, 100, 1000$
- Compare GD, Momentum, Adam

**Metrics**:
- Iterations to convergence ($\|f(\theta) - f^*\| < \epsilon$)
- Convergence rate (linear fit on log-loss vs iteration)

**Expected Results**:
- GD struggles with high $\kappa$
- Momentum accelerates convergence
- Adam handles all $\kappa$ robustly

### Experiment 2: PCA on Synthetic Data

**Goal**: Visualize explained variance and reconstruction

**Setup**:
- Generate correlated Gaussian data
- Apply PCA retaining top $k$ components
- Reconstruct and compute error

**Metrics**:
- Explained variance ratio
- Reconstruction error: $\|X - \hat{X}\|_F^2$

**Visualization**:
- Scree plot (explained variance vs component)
- Original vs reconstructed data

### Experiment 3: Ridge vs OLS

**Goal**: Show bias-variance tradeoff with regularization

**Setup**:
- Ill-conditioned design matrix
- Compare OLS vs Ridge ($\lambda = 0.01, 0.1, 1.0$)

**Metrics**:
- Condition number before/after regularization
- Test MSE
- Parameter norm $\|\theta\|$

---

## 📂 Module Structure

```
modules/01_numerical_toolbox/
├── README.md                          # This file
├── src/
│   ├── __init__.py
│   ├── optimizers_from_scratch.py     # GD, Momentum, Adam
│   ├── linear_algebra.py              # PCA, SVD utilities
│   └── toy_problems.py                # Quadratic bowls, LinReg
├── tests/
│   ├── test_optimizers.py             # Optimizer correctness
│   ├── test_linear_algebra.py         # PCA/SVD tests
│   └── test_toy_problems.py           # Integration tests
├── configs/
│   ├── optim_demo.yaml                # Optimizer experiment config
│   └── pca_demo.yaml                  # PCA experiment config
├── notebooks/
│   ├── 01_optimizer_convergence.ipynb # Interactive convergence viz
│   └── 02_pca_explained_variance.ipynb # PCA exploration
└── reports/                           # Generated plots and results
    ├── figures/
    │   ├── convergence_comparison.png
    │   ├── pca_scree_plot.png
    │   └── ridge_effect.png
    └── metrics.json
```

---

## 🚀 Quick Start

### Run Optimizer Demo

```bash
python -m modules.01_numerical_toolbox.src.optimizers_from_scratch run-optim-demo --seed 42
```

Produces:
- Convergence curves for GD, Momentum, Adam
- Condition number comparison
- Saved to `reports/figures/convergence_comparison.png`

### Run PCA Demo

```bash
python -m modules.01_numerical_toolbox.src.linear_algebra run-pca-demo --seed 42
```

Produces:
- Scree plot (explained variance)
- Original vs reconstructed data
- Saved to `reports/figures/pca_*.png`

### Run Tests

```bash
pytest modules/01_numerical_toolbox/tests/ -v
```

---

## ⚠️ Failure Modes

### 1. **Optimizer Divergence**
**Symptom**: Loss increases instead of decreasing  
**Cause**: Learning rate too large  
**Fix**: Reduce learning rate or use line search

### 2. **Slow Convergence**
**Symptom**: Many iterations with little progress  
**Cause**: Ill-conditioned problem  
**Fix**: Scale features, use momentum/Adam, or precondition

### 3. **Numerical Instability in PCA**
**Symptom**: Negative eigenvalues or NaN  
**Cause**: Near-singular covariance matrix  
**Fix**: Add small ridge regularization ($\lambda I$)

### 4. **Ridge Regularization Too Strong**
**Symptom**: Underfitting, high bias  
**Cause**: $\lambda$ too large  
**Fix**: Cross-validation to tune $\lambda$

---

## ✅ Definition of Done

- [x] README with theory and usage
- [x] Optimizers: GD, Momentum, Adam with convergence tracking
- [x] Linear algebra: PCA via SVD, ridge regularization
- [x] Toy problems: quadratic bowls, linear regression
- [x] Tests with >85% coverage
- [x] Demo CLIs producing plots
- [x] Notebooks with visualizations
- [x] All tests passing
- [x] Code formatted and linted

---

## 🔗 Resources

- **Numerical Optimization**: Nocedal & Wright (2006)
- **Matrix Computations**: Golub & Van Loan (2013)
- **Deep Learning Book**: Goodfellow et al. (2016), Ch. 2-4
- **Convex Optimization**: Boyd & Vandenberghe (2004)

---

## 🎓 Key Takeaways

1. **Conditioning matters**: $\kappa \gg 1$ slows convergence dramatically
2. **Scaling is crucial**: Always normalize features before training
3. **Adaptive methods help**: Adam handles diverse landscapes better than GD
4. **PCA = variance maximization**: Geometric interpretation aids intuition
5. **Regularization = better conditioning**: Ridge improves numerical stability
6. **Closed-form when possible**: Faster and exact for small, well-conditioned problems

---

## 🚀 Next Steps

After completing this module:
1. Run all experiments and examine convergence plots
2. Experiment with different condition numbers
3. Apply PCA to real datasets (e.g., MNIST)
4. Move to [Module 02: Statistical Inference & UQ](../02_stat_inference_uq/README.md)
