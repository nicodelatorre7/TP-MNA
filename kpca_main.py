import kpca
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

testing_num = 31
img_area = 112 * 92
images = np.zeros([testing_num, 112, 92])
print('######################### KPCA #########################\n')
print("* Reading testing images... ")
for k in range(testing_num):
    images[k, :, :] = mpimg.imread('./input_face/face_person_{0}.pgm'.format(k+1))
    # print(images[k,:, :])


eigenfaces = kpca.kpca_training()

print("* There are {0} eigenfaces".format(eigenfaces.shape))

print('\n [ TESTING ]')
results = np.zeros(testing_num)
for k in range(0, testing_num):
    results[k] = kpca.classify_svm(eigenfaces, images[k,:,:])

counter = 0
for k in range(0, testing_num):
    if k+1 == results[k]:
        counter +=1 
    print( "-> Image {0} belongs to the {1} person.".format(k+1,results[k]))

print('\n Finished Processing Images')
print('* Images scanned: {0}'.format(testing_num))
print('* Images area:')
print('\t - Height: {0}'.format(112))
print('\t - Width: {0}'.format(92))
print("* Successful rate: {0}%".format(100*counter/testing_num))



