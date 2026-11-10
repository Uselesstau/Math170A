import numpy as np

A = np.zeros((12, 8))
for i in range(12):
    t_i = i/11
    for j in range(8):
        A[i, j] = t_i ** j

condition_number = np.linalg.cond(A, 2)

# Classical Gram-Schmidt
def classical_gram_schmidt(M):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])  # uses original column A[:, j]
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v, 2)
        Q[:, j] = v / R[j, j]
    return Q, R

# Modified Gram-Schmidt
def modified_gram_schmidt(M):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = A.copy().astype(float)
    for i in range(n):
        R[i, i] = np.linalg.norm(V[:, i], 2)
        Q[:, i] = V[:, i] / R[i, i]
        for j in range(i + 1, n):
            R[i, j] = np.dot(Q[:, i], V[:, j])  # uses updated vector V[:, j]
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]
    return Q, R


def evaluate(Q, R, A):
    res_A = np.linalg.norm(A - Q @ R, 'fro')
    res_orth = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]), 'fro')

    QTQ = np.abs(Q.T @ Q)
    np.fill_diagonal(QTQ, 0.0)
    max_offdiag = np.max(QTQ)

    return res_A, res_orth, max_offdiag


Q_cgs, R_cgs = classical_gram_schmidt(A)
Q_mgs, R_mgs = modified_gram_schmidt(A)

# Metrics
res_cgs, orth_cgs, max_cgs = evaluate(Q_cgs, R_cgs, A)
res_mgs, orth_mgs, max_mgs = evaluate(Q_mgs, R_mgs, A)

print(f"Condition number kappa_2(A) = {condition_number:.6e}\n")
print(f"Classical Gram-Schmidt:")
print(f"||A - QR||_F = {res_cgs:.6e}")
print(f"||Q^T Q - I||_F = {orth_cgs:.6e}")
print(f"max_{{i != j}} |q_i^T q_j| = {max_cgs:.6e}\n")
print(f"Modified Gram-Schmidt:")
print(f"||A - QR||_F = {res_mgs:.6e}")
print(f"||Q^T Q - I||_F = {orth_mgs:.6e}")
print(f"max_{{i != j}} |q_i^T q_j| = {max_mgs:.6e}")
