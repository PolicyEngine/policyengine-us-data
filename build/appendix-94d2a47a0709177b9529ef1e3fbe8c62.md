# Appendix

## Appendix A: Implementation Code

### A.1 Quantile Regression Forest Implementation

The following code demonstrates the implementation of Quantile Regression Forests for variable imputation:

```python
from quantile_forest import RandomForestQuantileRegressor

qrf = RandomForestQuantileRegressor(
    n_estimators=100,
    min_samples_leaf=1,
    random_state=0
)
```

### A.2 PyTorch Optimization for Reweighting

The reweighting optimization uses PyTorch for gradient-based optimization:

```python
import torch

# Initialize with log of original weights
log_weights = torch.log(original_weights)
log_weights.requires_grad = True

# Adam optimizer
optimizer = torch.optim.Adam([log_weights], lr=0.1)

# Optimization loop
for iteration in range(5000):
    weights = torch.exp(log_weights)
    achieved = weights @ loss_matrix
    relative_errors = (achieved - targets) / targets
    loss = torch.mean(relative_errors ** 2)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Appendix B: Tables

### Table A1: Complete List of Imputed Variables

[TO BE GENERATED - Complete list of 72 imputed variables from PUF organized by category]