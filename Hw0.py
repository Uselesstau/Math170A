import numpy as np

# Number 1
def my_matvec(A, x):
    b = []
    for i in range(len(A)):
        b.append([])
        value = 0
        for j in range(len(A[i])):
            value += A[i][j] * x[j]
        b[i].append(value)
    return b

M = np.array([[2, -1, 0], [1, 3, 2], [-2, 0, 1], [4, 1, -1]])
V = np.array([[1], [2], [-1]])

print(my_matvec(M, V))
print(M @ V)

# Number 2
def my_matmat(A, B):
    b = []
    for i in range(len(A)):
        b.append([])
        for k in range(len(B[0])):
            value = 0
            for j in range(len(A[i])):
                value += A[i][j] * B[j][k]
            b[i].append(value)
    return b

a = np.array([[1, 2, -1], [0, 3, 4]])
b = np.array([[2, 1], [-1, 0], [3, -2]])

print(my_matvec(a, b))
print(a @ b)