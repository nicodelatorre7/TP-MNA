from os import listdir
from os.path import join, isdir
from pathlib import Path
import numpy as numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
#from sklearn import svm
import eig_calculator as eig


faces_path = 'caras-db'

people_count = 25
images_count = 9
img_area = 112 * 92

matrix_d = people_count*images_count

eigenfaces =

def get_training_images():
    # arreglo con imagenes
    images = numpy.zeros([people_count*images_count, img_area])

    # completamos el arreglo
    im_num = 0
    for k in range(people_count):
        i = k + 1
        for k2 in range(images_count):
            i2 = k2 +1
            img = plt.imread('./'+faces_path+'/s{}/{}'.format(i,i2)+'.pgm')
            images[im_num,:] = numpy.reshape(img,[1,img_area])
            im_num += 1

    return images

def kpca_training():
    images = get_training_images()

    # calculamos la "cara media" y la restamos del arreglo de imagenes.
    average_face = numpy.mean(images, 0)
    for i in range(images.shape[0]):
        images[i, :] -= average_face

    # matriz de covarianza
    T = numpy.transpose(images)

    # dependien si es CPA o KPCA es una matriz de covarianza u otra
    cov_mat = get_cov_matrix(images)


    # CALCULAR AUTOVECTORES DE LA MATRIZ


def classify(eigenfaces, input):
    # A partir de las eigenfaces y una imagen de entrada, determinar a qué persona pertenece la imagen de entrada

    # (train_images, train_labels)
    # (test_image, test_label)

    train_images = get_training_images()

    test_image = input

    model = keras.Sequential([
    #keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),  # 128 nodos de aprendizaje
    keras.layers.Dense(people_count)             # people_count possible labels
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])


    test_image = (numpy.expand_dims(test_image,0))
    predictions = probability_model.predict(test_image)

    return numpy.argmax(predictions[0])


def get_cov_matrix(images):
    K = np.tanh(np.dot(images, np.transpose(images)))

    # Calculamos autovectores y autovalores y para eso necesitamos la matriz
    # a partir de K'
    oneN = np.ones((matrix_d,matrix_d)) # matriz 1_N
    K_ = K - np.dot(oneN,K) - np.dot(K, oneN) + np.dot(np.dot(oneN,K),oneN)

    # calculo de autovalores y autovectores a partir de K_
    eig_vals, eig_vecs = eig.eig_calculator(K_)

    #Proyeccion
    projection = []
    alphas = eig_vecs
    lamdas = eig_vals

kpca_training()





