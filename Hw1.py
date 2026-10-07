import numpy as np

def linear_solve(A, b, use_partial_pivot=False):

    # Finds the partial_pivot row and makes the pivot
    def partial_pivot(matrix, column):
        pivot = matrix[column:, column]
        for k in range(column):
            pivot = np.insert(pivot, 0, 0)
        partial_pivot_value, partial_pivot_index = max(pivot), (pivot == max(pivot)).argmax()
        r_1 = matrix[column, :].copy()
        r_pivot = matrix[partial_pivot_index, :]
        matrix[column] = r_pivot
        matrix[partial_pivot_index] = r_1
        return matrix

    try:
        augment = np.concatenate((A, b), axis=1)
    except ValueError:
        print(f"Matrix {A} is not compatible with {b}")
        return ""

    # Constants
    rows, cols = augment.shape

    # Creates the Upper-triangle matrix
    for j in range(cols):
        if j > rows - 1:
            break
        if use_partial_pivot:
            augment = partial_pivot(augment, j)
        for i in range(rows):
            if i <= j or augment[i, j] == 0:
                continue
            l_ij = augment[i, j] / augment[j, j]
            r_i = augment[i, :] - l_ij * augment[j, :]
            augment[i] = r_i

    # Back substitution
    with np.errstate(divide='raise', invalid='raise'):
        x = np.zeros((rows, 1), float)
        for i in range(rows - 1, -1, -1):
            sum = 0
            for j in range(i + 1, rows):
                sum += augment[i, j] * x[j, 0]
            try:
                x[i, 0] = (augment[i, cols - 1] - sum) / augment[i, i]
            except (ZeroDivisionError, FloatingPointError):
                print(f"The Upper Matrix: \n{augment} \nHas a zero pivot at row {i+1}")
                return ""

        return x


# Part d
A_test = np.array(([0, 2, 1],
                   [1, -1, 2],
                   [2, 1, -1]), float)
b_test = np.array(([3],
                   [-3],
                   [5]))

X = linear_solve(A_test, b_test, use_partial_pivot=True)
print(f"x=\n{X} \nAx-b=\n{A_test @ X - b_test}")


# Part e
A_e = np.array(([10 ** -20, 1],
                [1, 1]), float)
b_e = np.array(([1],
                [2]))

X_nopivot = linear_solve(A_e, b_e)
X_pivot = linear_solve(A_e, b_e, use_partial_pivot=True)

print(f"x with pivot:\n{X_pivot} \nx without pivot:\n{X_nopivot}")
