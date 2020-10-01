from os import listdir
from os.path import join, isdir
from pathlib import Path
import numpy as numpy
import matplotlib.pyplot as plt
#from sklearn import svm


faces_path = 'caras-db'

people_count = 25
images_count = 10
img_area = 112 * 92



def pca_training():
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
    

    # calculamos la "cara media" y la restamos del arreglo de imagenes.
    average_face = numpy.mean(images, 0)
    for i in range(images.shape[0]):
        images[i, :] -= average_face

    # matriz de covarianza
    T = numpy.transpose(images)
    cov_mat = numpy.dot(images, T)


    # CALCULAR AUTOVECTORES DE LA MATRIZ


pca_training()
