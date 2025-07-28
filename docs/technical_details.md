# Technical Details

## Quantile Regression Forests

The imputation stage uses Quantile Regression Forests (QRF), an extension of random forests that estimates conditional quantiles rather than means. This approach:

- Captures non-linear relationships between demographics and income
- Preserves distributional characteristics of tax variables
- Allows sampling from the full conditional distribution
- Maintains correlations between imputed variables

### Implementation

```python
from quantile_forest import RandomForestQuantileRegressor

# Configure QRF model
qrf = RandomForestQuantileRegressor(
    n_estimators=100,
    random_state=0,
    min_samples_leaf=1
)

# Train on PUF subsample
qrf.fit(X_train, y_train)

# Sample from conditional distribution
quantile = np.random.beta(1, 1)  # Uniform over [0,1]
prediction = qrf.predict(X_test, quantiles=[quantile])
```

## Reweighting Algorithm

The calibration stage uses PyTorch to optimize household weights through gradient descent:

### Objective Function

Minimize mean squared relative error across all targets:

```
L(w) = (1/m) * Σ((exp(w)ᵀ * M_i - t_i) / t_i)²
```

Where:
- `w`: Log-transformed household weights
- `M`: Loss matrix (households × targets)
- `t`: Target values vector

### Optimization Details

- **Optimizer**: Adam with learning rate 0.1
- **Iterations**: 5,000 or until convergence
- **Regularization**: 5% dropout during training
- **Initialization**: Log of original CPS weights

## Loss Matrix Construction

The loss matrix captures each household's contribution to calibration targets:

### Target Categories

1. **IRS Income Components**
   - By AGI bracket and filing status
   - 11 income types × multiple brackets
   - Total: 5,300+ targets

2. **Demographics**
   - Single-year age populations
   - State populations
   - Healthcare spending by age

3. **Program Participation**
   - SNAP, Social Security, SSI
   - Unemployment compensation
   - Income tax revenue

4. **Tax Expenditures**
   - Simulated revenue effects of repeal
   - SALT, charitable, mortgage interest, medical

### Matrix Assembly

```python
def build_loss_matrix(dataset, year):
    matrix = []
    targets = []
    names = []
    
    # Add each target type
    add_soi_targets(matrix, targets, names, year)
    add_census_targets(matrix, targets, names, year)
    add_cbo_targets(matrix, targets, names, year)
    add_jct_targets(matrix, targets, names, year)
    
    return np.column_stack(matrix), np.array(targets), names
```

## Performance Optimization

### Computational Efficiency
- Sparse matrix operations where applicable
- Batch processing for microsimulation
- Caching of intermediate results
- Parallelization of independent calculations

### Memory Management
- Chunked reading of large datasets
- Efficient dtype selection
- Garbage collection between stages
- HDF5 compression for storage

## Validation Framework

### Cross-Validation
- 5-fold validation on calibration targets
- Hold-out testing for imputation quality
- Stability testing across random seeds

### Metrics
- Absolute relative error by target
- Distributional comparisons (Gini, percentiles)
- Program participation rates
- Revenue projections under reforms

## Reproducibility

All results can be regenerated:

```bash
# Generate datasets
make data

# Run validation
make paper-results

# Build documentation
make documentation
```

Random seeds are fixed throughout:
- QRF training: seed 0
- Worker EAD assignment: seed 0
- Student EAD assignment: seed 1
- Family correlation: seed 100