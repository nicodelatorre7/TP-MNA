import kpca
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

eigenfaces = kpca.kpca_training()

img_0 = mpimg.imread('./input_face/face_person_0.pgm')
img_1 = mpimg.imread('./input_face/face_person_1.pgm')
img_2 = mpimg.imread('./input_face/face_person_2.pgm')


print("there are {0} eigenfaces".format(eigenfaces.shape))

img_00 = kpca.classify_svm(eigenfaces, img_0)
img_10 = kpca.classify_svm(eigenfaces, img_1)
img_20 = kpca.classify_svm(eigenfaces, img_2)


print( "image 0 belongs to the {0} person.".format(img_00))
print( "image 1 belongs to the {0} person.".format(img_10))
print( "image 2 belongs to the {0} person.".format(img_20))

img_01 = kpca.classify_svm(eigenfaces, img_0)
img_11 = kpca.classify_svm(eigenfaces, img_1)
img_21 = kpca.classify_svm(eigenfaces, img_2)


print( "image 0 belongs to the {0} person.".format(img_01))
print( "image 1 belongs to the {0} person.".format(img_11))
print( "image 2 belongs to the {0} person.".format(img_21))