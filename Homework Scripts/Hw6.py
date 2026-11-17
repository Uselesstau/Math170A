import numpy as np

T = np.array([[2.0, 1.0, 0.0],
              [1.0, 2.0, 1.0],
              [0.0, 1.0, 2.0]])

q = np.array([1.0, 0.0, 0.0])

results = {}

for k in range(13):
    theta = float(q.T @ T @ q)
    r = float(np.linalg.norm(T @ q - theta * q))

    if k in [0, 1, 2, 4, 8, 12]:
        results[k] = (theta, r)

    z = T @ q
    q = z / np.linalg.norm(z)

print(f"{'k':<4} {'theta_k':<16} {'r_k':<16}")

for k, (theta, r) in results.items():
    print(f"{k:<4} {theta:<16.8f} {r:<16.8e}")