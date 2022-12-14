import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('./beer/stella_0.jpg')
surf = cv.xfeatures2d.SURF_create(10000) #Hessian threshold, best 300-500
kp, des = surf.detectAndCompute(img, None)
img2 = cv.drawKeypoints(img, kp, None, (255,0,0) ,4)
plt.imshow(img2), plt.show()