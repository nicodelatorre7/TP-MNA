
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('./test_image.pgm')

print(img)

fig, ax = plt.subplots() 
ax.imshow(img) 
ax.axis('off') 
  
plt.title('testing image handling',  
                                     fontweight ="bold") 
plt.show() 