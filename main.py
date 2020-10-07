import pca as pca
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np



testing_num = 30
img_area = 112 * 92
images = np.zeros([testing_num, 112, 92])

print("reading testing images... ")
for k in range(testing_num):
    images[k, :, :] = mpimg.imread('./input_face/face_person_{0}.pgm'.format(k+1))
    # print(images[k,:, :])


eigenfaces = pca.pca_training()

print("there are {0} eigenfaces".format(eigenfaces.shape))


results = np.zeros(testing_num)
for k in range(0, testing_num):
    results[k] = pca.classify_svm(eigenfaces, images[k,:,:])

for k in range(0, testing_num):
    print( "image {0} belongs to the {1} person.".format(k+1,results[k]))


