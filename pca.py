from os import listdir
from os.path import join, isdir
from pathlib import Path
import numpy as numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import eig_calculator as ec
from sklearn import svm


faces_path = 'caras-db'

people_count = 3
images_count = 9
img_area = 112 * 92

# heuristics
max_eigenfaces = 100

def get_training_images():
    # arreglo con imagenes
    images = numpy.zeros([people_count*images_count, img_area])

    # completamos el arreglo
    print("reading images... ")
    im_num = 0
    for k in range(people_count):
        i = k + 1
        for k2 in range(images_count):
            i2 = k2 +1
            img = plt.imread('./'+faces_path+'/s{}/{}'.format(i,i2)+'.pgm')
            images[im_num,:] = numpy.reshape(img,[1,img_area])
            im_num += 1

    # calculamos la "cara media" y la restamos del arreglo de imagenes.
    average_face = numpy.mean(images, 0)
    for i in range(images.shape[0]):
        images[i, :] -= average_face

    return images

def pca_training():
    images = get_training_images()

    print("calculating cov_matrix... ")
    # matriz de covarianza
    A = numpy.transpose(images)
    n, m = A.shape
    cov_mat = numpy.dot(images, A)

    print("calculating eigenfaces")
    
    images_matrix = numpy.asmatrix(images)
    # eigenvalues, eigenfaces = numpy.linalg.eig(cov_mat)
    # eigenvalues, eigenfaces = ec.eig_calculator(cov_mat)
    
    for i in range(m):
        eigenfaces[:,i] /= numpy.linalg.norm(eigenfaces[:,i])


    return eigenfaces[:,0:max_eigenfaces]

    



def classify(eigenfaces, input):
    # A partir de las eigenfaces y una imagen de entrada, determinar a qué persona pertenece la imagen de entrada

    # (train_images, train_labels) 
    # (test_image, test_label)

    train_images = get_training_images()

    test_image = input
    test_image = numpy.reshape(test_image,[1,img_area])

    train_images      = numpy.dot(train_images,eigenfaces)
    test_image   = numpy.dot(test_image,eigenfaces)

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
    train_labels = numpy.asarray(train_labels)


    print("feeding training images into neural net... ")
    model.fit(train_images, train_labels, epochs=10)

    probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
    

    #test_image = (numpy.expand_dims(test_image,0))
    predictions = probability_model.predict(test_image)

    return predictions[0]









