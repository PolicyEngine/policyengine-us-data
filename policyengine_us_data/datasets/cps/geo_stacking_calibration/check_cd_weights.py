import numpy as np

w = np.load('w_cd_20250911_102023.npy')
print(f'Weight array shape: {w.shape}')
print(f'Non-zero weights: {np.sum(w != 0)}')
print(f'Total weights: {len(w)}')
print(f'Sparsity: {100*np.sum(w != 0)/len(w):.2f}%')