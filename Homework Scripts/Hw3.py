import numpy as np

# Define matrix A
A = np.array([
    [1, 2, 0],
    [-1, 1, 2],
    [0, -2, 1],
    [2, 0, 1]
], dtype=float)

np.random.seed(170)

vectors = np.random.randn(3, 20000)

N_values = [200, 2000, 20000]
p_values = [1, 2, np.inf]

for p in p_values:
    # Normalize each vector to have p-norm 1
    norms_p = np.linalg.norm(vectors, ord=p, axis=0)
    normalized_vectors = vectors / norms_p

    # Matrix-vector multiplication for all samples at once
    Ax = A @ normalized_vectors

    # Compute the p-norm of the resulting Ax vectors
    Ax_norms = np.linalg.norm(Ax, ord=p, axis=0)

    # Get library value for comparison
    lib_val = np.linalg.norm(A, ord=p)

    p_label = "inf" if p == np.inf else str(p)
    print(f"\nEstimates for p = {p_label}:")

    for N in N_values:
        max_val = np.max(Ax_norms[:N])
        print(f"\nMax over first {N:<6} samples : {max_val:.6f}")

    print(f"Library ||A||_{p_label:<14} : {lib_val:.6f}")

# Frobenius norm
fro_norm = np.linalg.norm(A, ord='fro')
print(f"\nLibrary ||A||_F: {fro_norm:.6f}\n")

# Verify exact calculations match the library 2-norm
exact_val = np.sqrt((15 + np.sqrt(13)) / 2)
lib_2_norm = np.linalg.norm(A, ord=2)

print(f"\nA^T @ A: {A.transpose() @ A:}")
print(f"\nCalculated sqrt((15+sqrt(13))/2): {exact_val:.6f}")
print(f"\nLibrary ||A||_2: {lib_2_norm:.6f}")