import cv2 as cv
import matplotlib.pyplot as plt

HESSIAN_THRESHOLD = 300#200
FRAME_RESIZE_FACTOR = 2#3
TEMPLATE_RESIZE_FACTOR = 1#2
FLANN_DISTANCE_SENS = 0.75#0.75
FLANN_MATCH_COUNT_SENS = 20 #11
COMPARE_SENS = 9#9

beerimgpath = './beer/laces_0.jpg'
beerimg = cv.imread(beerimgpath)

beerframeimgpath = './beerframe/hltipa_0.jpg'
beerframeimg = cv.imread(beerframeimgpath)

surf = cv.xfeatures2d.SURF_create(HESSIAN_THRESHOLD)
kp1, des_beer = surf.detectAndCompute(beerimg,None)
kp2, des_beerframe = surf.detectAndCompute(beerframeimg,None)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des_beer,des_beerframe,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(beerimg,kp1,beerframeimg,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()