import numpy as np

eta = (10 ** -2, 10 ** -6, 10 ** -10)

print(f"\neta | condition number | relative b error | relative forward error | perturbation value | residual")
for n in eta:

    # Initialize
    A_eta = np.array([[1, 1],
                      [1, 1 + n]])

    x_star = np.array([[1],
                 [1]])

    b_eta = A_eta @ x_star

    delta_b = np.array([[0],
                        [10 ** -8]])

    # Solve
    x = np.linalg.solve(A_eta, b_eta)
    x_tilda = np.linalg.solve(A_eta, b_eta + delta_b)

    # Values

    def two_norm_mat(M):
        lambda_max = max(abs(np.linalg.eigvals(M.T @ M)))
        return lambda_max ** 0.5

    def two_norm_vec(v):
        return (v.T @ v)[0][0] ** 0.5

    condition_num = two_norm_mat(A_eta) * two_norm_mat(np.linalg.inv(A_eta))
    relative_b_error = two_norm_vec(delta_b) / two_norm_vec(b_eta)
    relative_forward_error = two_norm_vec(x_tilda - x) / two_norm_vec(x)
    perturbation_value = relative_forward_error / (condition_num * relative_b_error)
    residual = two_norm_vec(A_eta @ x_tilda - (b_eta + delta_b))

    print(f"{n} | {condition_num} | {relative_b_error} | {relative_forward_error} |  {perturbation_value} |  {residual}")

