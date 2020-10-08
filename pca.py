from os import listdir
from os.path import join, isdir
from pathlib import Path
import numpy as numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from eig_calculator import gram_schmidt, compare_eig
from sklearn import svm


faces_path = 'caras-db'

people_count = 31
images_count = 7
img_area = 112 * 92

# heuristics
max_eigenfaces = 20

def get_training_images():
    # arreglo con imagenes
    images = numpy.zeros([people_count*images_count, img_area])
    people = numpy.zeros([people_count*images_count,1])

    # completamos el arreglo
    print("\n[ TRAINING ]")
    print("* Reading images... ")
    im_num = 0
    for k in range(people_count):
        i = k + 1
        for k2 in range(images_count):
            i2 = k2 +1
            img = plt.imread('./'+faces_path+'/s{}/{}'.format(i,i2)+'.pgm')/255.0
            images[im_num,:] = numpy.reshape(img,[1,img_area])
            people[im_num,0] = i
            im_num += 1

    return images, people

def pca_training():
    images, people = get_training_images()

    # calculamos la "cara media" y la restamos del arreglo de imagenes.
    average_face = numpy.mean(images, 0)
    for i in range(images.shape[0]):
        images[i, :] -= average_face

    print("* Calculating covariance matrix... ")
    T = numpy.transpose(images)
    n, m = T.shape
    L = numpy.dot(images, T)

    #A = np.array([[60., 30., 20.], [30., 20., 15.], [20., 15., 12.]])
    last_R = numpy.zeros(T.shape)
    eigen = False
    eigen_L = 1

    print('* Getting eigenvalues and eigenvectors..')
    while not eigen:
        Q, R = gram_schmidt(L)
        L = numpy.dot(R, Q)
        eigen_L = numpy.dot(eigen_L, Q)
        eigen = compare_eig(last_R, R)
        last_R = R

    eigen_C = numpy.dot(T, eigen_L)

    for i in range(m):
        eigen_C[:,i] /= numpy.linalg.norm(eigen_C[:,i])

    a = eigen_C[:,0:max_eigenfaces]
    return a

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


    predictions = probability_model.predict(test_image)

    return predictions[0]

def classify_svm(eigenfaces, input):
    # A partir de las eigenfaces y una imagen de entrada, determinar a qué persona pertenece la imagen de entrada

    train_images, people = get_training_images()

    test_image = numpy.zeros([1,img_area])
    input = input/255.0
    test_image[0,:] = numpy.reshape(input,[1,img_area])

    # calculamos la "cara media" y la restamos del arreglo de imagenes.
    average_face = numpy.mean(train_images, 0)
    for i in range(test_image.shape[0]):
        test_image[i, :] -= average_face


    train_images      = numpy.dot(train_images,eigenfaces)
    test_image   = numpy.dot(test_image,eigenfaces)

    people = numpy.asarray(people).ravel()

    clf = svm.LinearSVC()
    clf.fit(train_images, people)


    return clf.predict(test_image)





