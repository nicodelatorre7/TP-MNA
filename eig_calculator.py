import numpy as np

MAX_ITERATIONS = 1000
ERROR = 0.001



## Implementacion de Householder QR -> https://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf
#  Compute the QR decomposition of an m-by-n matrix A using
#  Householder transformations.
##
def qr_decomposition(A):
    
    #shape me dice de cuanto por cuanto es la matriz -> m = filas ; n = columnas (m deberia ser = a n siempre)
    m, n = np.shape(A)
    Q = np.eye(m)
    R = np.copy(A)
    for i in range(n):
        
        norm = np.linalg.norm(R[i:m, i])
        u1 = R[i, i] + np.sign(R[i, i]) * norm
        v = R[i:m, i].reshape((-1, 1)) / u1
        v[0] = 1
        tau = np.sign(R[i, i]) * u1 / norm

        # Ahorro la multiplicacion de matrices: solo necesito restar una columna de cada matriz
        R[i:m, :] = R[i:m, :] - (tau * v) * np.dot(v.reshape((1, -1)), R[i:m, :])
        Q[:, i:n] = Q[:, i:n] - (Q[:, i:m].dot(v)).dot(tau * v.transpose())

    return Q, R


# https://en.wikipedia.org/wiki/QR_algorithm 
def eig_calculator(a):
    q, r = qr_decomposition(a)
    qcomp = q

    i = 0
    b = 1

    old_val = np.zeros(r.shape[0])
    new_val = np.ones(r.shape[0])

    while b > ERROR and i < MAX_ITERATIONS:

        old_val = new_val
        a = np.matmul(q.transpose(), a)
        a = np.matmul(a, q)
        q, r = qr_decomposition(a)
        new_val = np.diag(a)
        qcomp = np.matmul(qcomp, q)
        i+=1
        b = max(abs(new_val-old_val))


    # para la normalizacion
    for i in range(0, qcomp.shape[0]):
        qcomp[:, i] = qcomp[:, i] / np.linalg.norm(qcomp[:, i])

    #ordeno autovectores de acuerdo al peso de los autovalores
    a = np.diag(a)

    #argsort es ascendente por default, entonces le pongo - para hacerlo descendente.
    sort = np.argsort(- np.absolute(a))

    eVal = a[sort]
    eVec = qcomp[:, sort]

    #cada col de eVec tiene un autovalor asociado en la misma columna de eVal
    return eVal, eVec


"""
        TESTEO
"""

# A = np.random.rand(5,5)*1000
# val,vec = eig_calculator(A)
# print(val)
# print(np.linalg.eig(A)[0])
# print(vec)
# print("\n")
# print(np.linalg.eig(A)[1])

