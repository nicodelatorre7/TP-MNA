from os import listdir
from os.path import join, isdir
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn import svm
import eig_calculator as eig


faces_path = 'caras-db'

people_count = 3
images_count = 9
img_area = 112 * 92

matrix_d = people_count*images_count
# heuristics
max_eigenfaces = 100

def get_training_images():
    # arreglo con imagenes
    images = np.zeros([people_count*images_count, img_area])
    people = np.zeros([people_count*images_count,1])

    # completamos el arreglo
    print("reading images... ")
    im_num = 0
    for k in range(people_count):
        i = k + 1
        for k2 in range(images_count):
            i2 = k2 +1
            img = plt.imread('./'+faces_path+'/s{}/{}'.format(i,i2)+'.pgm')/255.0
            images[im_num,:] = np.reshape(img,[1,img_area])
            people[im_num,0] = i
            im_num += 1

    return images, people

def kpca_training():
    images, people = get_training_images()

    average_face = np.mean(images, 0)
    for i in range(images.shape[0]):
        images[i, :] -= average_face

    print("calculating cov_matrix... ")
    T = np.transpose(images)
    n, m = T.shape
    L = get_cov_matrix(images)

    #A = np.array([[60., 30., 20.], [30., 20., 15.], [20., 15., 12.]])
    last_R = np.zeros(T.shape)
    eigen = False
    eigen_L = 1

    print("post calculate matrix... ")

    for i in range(1000):
        Q, R = eig.gram_schmidt(L)
        L = np.dot(R, Q)
        eigen_L = np.dot(eigen_L, Q)
        eigen = eig.compare_eig(last_R, R)
        last_R = R

    eigen_C = np.dot(T, eigen_L)

    for i in range(m):
        eigen_C[:,i] /= np.linalg.norm(eigen_C[:,i])

    # for col in range(alphas.shape[1]):
    #     alphas[:,col] = alphas[:,col]/np.sqrt(lambdas[col])

    a = eigen_C[:,0:max_eigenfaces]
    return eigen_C[:,0:max_eigenfaces]

def classify_svm(eigenfaces, input):

    # A partir de las eigenfaces y una imagen de entrada, determinar a qué persona pertenece la imagen de entrada
    train_images, people = get_training_images()

    test_image = np.zeros([1,img_area])
    input = input/255.0
    test_image[0,:] = np.reshape(input,[1,img_area])

    # calculamos la "cara media" y la restamos del arreglo de imagenes.
    average_face = np.mean(train_images, 0)
    for i in range(test_image.shape[0]):
        test_image[i, :] -= average_face

    train_images = np.dot(train_images,eigenfaces)
    test_image = np.dot(test_image,eigenfaces)

    people = np.asarray(people).ravel()

    clf = svm.LinearSVC()
    clf.fit(train_images, people)

    return clf.predict(test_image)

def get_cov_matrix(images):
    K = np.tanh(np.dot(images, np.transpose(images)))

    # Calculamos autovectores y autovalores y para eso necesitamos la matriz
    # a partir de K'
    oneN = np.ones((matrix_d,matrix_d))/matrix_d # matriz 1_N
    K_ = K - np.dot(oneN,K) - np.dot(K, oneN) + np.dot(np.dot(oneN,K),oneN)

    return K_

    # calculo de autovalores y autovectores a partir de K_
    # eig_vals, eig_vecs = np.linalg.eig(K_)

    # #Proyeccion
    # projection = []
    # alphas = eig_vecs
    # lambdas = eig_vals

    # #Los autovalores vienen en orden descendente. Lo cambio
    # lambdas = np.flipud(lambdas)
    # alphas  = np.fliplr(alphas)

    # for col in range(alphas.shape[1]):
    #     alphas[:,col] = alphas[:,col]/np.sqrt(lambdas[col])
    #     # alphas[:,col] = alphas[:,col]/lambdas[col]

    # return np.dot(np.transpose(K_),alphas)




