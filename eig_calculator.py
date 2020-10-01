import numpy as np
from sympy import *



def eig_calculator(matrix):
    #shape me dice de cuanto por cuanto es la matriz -> m = filas ; n = columnas (m deberia ser = a n siempre)
    m, n = np.shape(matrix)
    if n != m:
        raise Exception("The matrix must be square to obtain eigenvalues!")

    I = np.eye(n)
    x = Symbol('x')

    aux = Matrix(matrix - I*x)
    eq1 = aux.det()
    eig_values = solve(eq1,x)

    #redondeo a 4 decimales
    i=0
    while i < n:
        eig_values[i] = round(eig_values[i],4)
        i+=1


    """
    ##
    # HASTA ACA ANDA 
    # LO DE ABAJO NO, SOLO CALCULA AUTOVALORES,
    # Y EN AUTOVECTORES TIRA CUALQUIER MIERDA#
    ##
    """

    # adaptacion de: https://www.youtube.com/watch?v=ssfMqFycXOU a python
    eig_vectors = np.zeros((n,n))
    # print(eig_vectors)
    # i = 0
    # while i < n:
    #     M = Matrix( matrix - eig_values[i]* I)
    #     M_red = M.rref()
        # print("M_RED: ")
        # print(M_red)
        # print("M_RED[i]: ")
        # print(M_red[0][i])
        # res.append(-M_red[n-1][i])
        # eig_vectors[i] = -M_red[n-1][i]
    #     print(-M_red[1][i])
        # i+=1

    # print(eig_vectors)

    return eig_values,eig_vectors


def main():
    x = np.array([[1,2,3],[4,5,6],[7,4,9]])
    #x = np.array([[0.9, 0.01 , 0.09],[0.09, 0.9 , 0.01],[0.09,0.01,0.9]] )
    # print("....X....")
    # print(x)
    eVal, eVec = eig_calculator(x)
    print("....A-Valores....")
    print(eVal)
    print("....A-Vectores....")
    print(eVec)

main()
