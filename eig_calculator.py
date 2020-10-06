import numpy as np

MAX_ITERATIONS = 1000
ERROR = 0.001
EPSILON = 1e-2


##
# Adaptacion del algoritmo de descomposicion QR con Gram-Schmidt
# -- https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
# -- https://en.wikipedia.org/wiki/QR_decomposition
# -- https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/GramSchmidt.pdf 
##
def gram_schmidt(matrix):
    m, n = matrix.shape
    
    if n != m:
        raise Exception("The matrix must be square to obtain eigenvalues!")

    Q = np.zeros(shape=(m, n))
    R = np.zeros(shape=(n, n))

    for k in range(0, n):
        R[k, k] = np.linalg.norm(matrix[0:m, k])
        Q[0:m, k] = matrix[0:m, k] / R[k, k]

        for j in range(k + 1, n):
            R[k, j] = np.dot(np.transpose(Q[0:m, k]), matrix[0:m, j])
            matrix[0:m, j] = matrix[0:m, j] - np.dot(Q[0:m, k], R[k, j])

    return Q, R



def compare_eig(old_R, new_R):
    for i in range(0, old_R.shape[0]):
        if abs(old_R[i][i] - new_R[i][i]) > EPSILON:
            # print(abs(old_R[i][i] - new_R[i][i]))
            return False

    return True




"""
        TESTEO
"""

# A = np.array([[12,-51,4],[6,167,-68],[-4,24,-41]])
# val,vec = gram_schmidt(A)
# print("VALUES")
# print(val)
# print("EIG_VALUES")
# print(np.linalg.eig(A)[0])
# print("VECTOR")
# print(vec)
# print("\n")
# print("EIG_VEC")
# print(np.linalg.eig(A)[1])

