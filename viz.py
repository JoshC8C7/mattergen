import numpy as np
import matplotlib.pyplot as plt

# Batch sizes and repeated memory measurements
x_data = np.array([1, 8, 16, 32, 64, 128, 256])
y_measurements = np.array([
    [1055, 1105, 1075, 1075, 1105],
    [1399, 1283, 1345, 1299, 1359],
    [1917, 1823, 1849, 1829, 1931],
    [2831, 2771, 2307, 2819, 2395],
    [4465, 5243, 15013, 4641, 5481],
    [9495, 9345, 8765, 9513, 10285],
    [23735, 16537, 17175, 18685, 20483]
])

# Calculate mean and std
y_mean = np.mean(y_measurements, axis=1)
y_std = np.std(y_measurements, axis=1)

# Normalize x for stable fitting
x_mean = np.mean(x_data)
x_norm = x_data - x_mean
poly_coeffs = np.polyfit(x_norm, y_mean, deg=2)
poly = np.poly1d(poly_coeffs)

# Evaluation function
def poly_fit(x):
    return poly(x - x_mean)

# Solve for batch size from memory
def solve_batch_size(y_target):
    coeffs = poly_coeffs.copy()
    coeffs[-1] -= y_target
    roots = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)].real
    x_candidates = real_roots + x_mean
    return x_candidates[x_candidates > 0][1] if any(x_candidates > 0) else None

# GPU targets
y_targets = [24564, 81920]
labels = ["RTX4090", "H100"]
x_targets = [solve_batch_size(y) for y in y_targets]
print(x_targets)

# Clip x range at 2^11 = 2048
x_max = min(2048, max(x_targets) * 1.2)
x_dense = np.linspace(1, x_max, 500)
y_dense = poly_fit(x_dense)

# Plot
plt.figure()
plt.plot(x_dense, y_dense, label="Quadratic Fit", zorder=1)
plt.errorbar(x_data, y_mean, yerr=y_std, fmt='o', color='black', capsize=5, label="Measured Data", zorder=2)

for x_val, y_val, label in zip(x_targets, y_targets, labels):
    if x_val < 2048:
        plt.plot([1, x_val], [y_val, y_val], color='red', linestyle='--', zorder=0)
        plt.plot([x_val, x_val], [0, y_val], color='red', linestyle='--', zorder=0)
        plt.text(x_val, y_val + 2000, f"{label}: batch ≈ {x_val:.0f}", color='red', ha='center', fontsize=9)

plt.xscale('log', base=2)
plt.xlabel("Batch Size (log₂ scale)")
plt.ylabel("Memory Usage (MiB)")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.title("Quadratic Fit Clipped at 2¹¹ (2048)", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig("batchsize_vs_memory_clipped.png", dpi=500)
plt.close()
