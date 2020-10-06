from os import listdir
from os.path import join, isdir
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
#from sklearn import svm
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

    # completamos el arreglo
    print("reading images... ")
    im_num = 0
    for k in range(people_count):
        i = k + 1
        for k2 in range(images_count):
            i2 = k2 +1
            img = plt.imread('./'+faces_path+'/s{}/{}'.format(i,i2)+'.pgm')
            images[im_num,:] = np.reshape(img,[1,img_area])
            im_num += 1

    # calculamos la "cara media" y la restamos del arreglo de imagenes.
    average_face = np.mean(images, 0)
    for i in range(images.shape[0]):
        images[i, :] -= average_face

    return images

def kpca_training():
    images = get_training_images()

    print("calculating cov_matrix... ")
    # matriz de covarianza
    A = np.transpose(images)
    n, m = A.shape
    cov_mat = get_cov_matrix(images)

    print("calculating eigenfaces")

    images_matrix = np.asmatrix(images)
    eigenvalues, eigenfaces = np.linalg.eig(cov_mat)
    # eigenvalues, eigenfaces = ec.eig_calculator(cov_mat)

    for i in range(m):
        eigenfaces[:,i] /= np.linalg.norm(eigenfaces[:,i])

    return eigenfaces[:,0:max_eigenfaces]

def classify_svm(eigenfaces, input):
    # A partir de las eigenfaces y una imagen de entrada, determinar a qué persona pertenece la imagen de entrada

    # (train_images, train_labels)
    # (test_image, test_label)

    train_images = get_training_images()

    test_image = input
    test_image = np.reshape(test_image,[1,img_area])

    train_images      = np.dot(eigenfaces,train_images)
    test_image   = np.dot(eigenfaces[0],test_image)

    model = keras.Sequential([
    #keras.layers.Flatten(input_shape=(112, 92)),
    keras.layers.Dense(128, activation='relu'),  # 128 nodos de aprendizaje
    keras.layers.Dense(people_count)             # people_count possible labels
    ])

    print("compiling classification module... ")
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    train_labels = [0, 0, 0, 0, 0, 0,0,0,0, 1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,2]
    train_labels = np.asarray(train_labels)


    print("feeding training images into neural net... ")
    model.fit(train_images, train_labels, epochs=10)

    probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])


    #test_image = (np.expand_dims(test_image,0))
    predictions = probability_model.predict(test_image)

    return predictions[0]


def get_cov_matrix(images):
    K = np.tanh(np.dot(images, np.transpose(images)))

    # Calculamos autovectores y autovalores y para eso necesitamos la matriz
    # a partir de K'
    oneN = np.ones((matrix_d,matrix_d))/matrix_d # matriz 1_N
    K_ = K - np.dot(oneN,K) - np.dot(K, oneN) + np.dot(np.dot(oneN,K),oneN)

    # calculo de autovalores y autovectores a partir de K_
    eig_vals, eig_vecs = np.linalg.eig(K_)

    #Proyeccion
    projection = []
    alphas = eig_vecs
    lambdas = eig_vals

    #Los autovalores vienen en orden descendente. Lo cambio
    lambdas = np.flipud(lambdas)
    alphas  = np.fliplr(alphas)

    for col in range(alphas.shape[1]):
        # alphas[:,col] = alphas[:,col]/np.sqrt(lambdas[col])
        alphas[:,col] = alphas[:,col]/lambdas[col]

    return np.dot(np.transpose(K_),alphas)
# kpca_training()





