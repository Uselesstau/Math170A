import numpy as np
import time

seed = 1
np.random.seed(seed)

def timing_computations(sizes=[100,200,400], trails=5):

    def generate_square_matrix(n):
        mat = np.zeros((n,n))
        for i in range(n):
           for j in range(n):
               mat[i,j] = np.random.random() * np.random.randint(-1, 1) * np.random.randint(1, 10000)
        return mat

    def generate_vector(n):
        vec = np.zeros(n)
        for i in range(n):
            vec[i] = np.random.random() * np.random.randint(-1, 1) * np.random.randint(1, 10000)
        return vec

    def ab_x_operation(a, b, x):
        start_time = time.perf_counter()
        (a @ b) @ x
        end_time = time.perf_counter()
        delta_time = end_time - start_time
        return delta_time

    def a_bx_operation(a, b, x):
        start_time = time.perf_counter()
        a @ (b @ x)
        end_time = time.perf_counter()
        delta_time = end_time - start_time
        return delta_time

    for i in range(len(sizes)):
        times_ab_x = []
        times_a_bx = []
        for j in range(trails):
            A = generate_square_matrix(sizes[i])
            B = generate_square_matrix(sizes[i])
            x = generate_vector(sizes[i])

            times_ab_x.append(ab_x_operation(A,B,x))
            times_a_bx.append(a_bx_operation(A,B,x))

        median_ab_x = np.median(times_ab_x)
        median_a_bx = np.median(times_a_bx)

        ratio = median_ab_x/median_a_bx - 1

        print(f"\n\n\nFor Trail n={sizes[i]}, Median AB(x) = {median_ab_x}, Median A(Bx) = {median_a_bx}, Median ratio = {ratio}")
        print(f"\nTime chart for AB(x) = {times_ab_x} \nTime chart for A(Bx) = {times_a_bx}")

timing_computations()