import numpy as np
import cv2 as cv

SHOW_CAMERA = False
HESSIAN_THRESHOLD = 300#200
FRAME_RESIZE_FACTOR = 2#3
TEMPLATE_RESIZE_FACTOR = 1#2
FLANN_DISTANCE_SENS = 0.75#0.75
FLANN_MATCH_COUNT_SENS = 20 #11
COMPARE_SENS = 9#9

IMGPATH_HLTIPA = './beer/hltipa_0.jpg'
IMGPATH_LACES = './beer/laces_0.jpg'
IMGPATH_STELLA = './beer/stella_0.jpg'

def learnbeer():
    print("Learning beer...")

    print("Learned .")

def qualifying_matches(matches):
    good = []
    for m,n in matches:
        if m.distance < FLANN_DISTANCE_SENS*n.distance:
            good.append([m])
    return good

def main():
    #init
    template_hltipa = cv.imread(IMGPATH_HLTIPA)
    template_laces = cv.imread(IMGPATH_LACES)
    template_stella = cv.imread(IMGPATH_STELLA)

    surf = cv.xfeatures2d.SURF_create(HESSIAN_THRESHOLD)
    kp_hltipa, des_hltipa = surf.detectAndCompute(template_hltipa,None)
    kp_laces, des_laces = surf.detectAndCompute(template_laces,None)
    kp_stella, des_stella = surf.detectAndCompute(template_stella,None)

    bf = cv.BFMatcher()

    cap = cv.VideoCapture(0, cv.CAP_V4L)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    #enter loop
    while True:
        e1 = cv.getTickCount()

        #get frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        #surf the camera frame
        kp_frame, des_frame = surf.detectAndCompute(frame,None)
        #match against known 
        matches_hltipa = bf.knnMatch(des_hltipa,des_frame,k=2)
        matches_laces = bf.knnMatch(des_laces,des_frame,k=2)
        matches_stella = bf.knnMatch(des_stella,des_frame,k=2)
        good_hltipa = qualifying_matches(matches_hltipa)
        good_laces = qualifying_matches(matches_laces)
        good_stella = qualifying_matches(matches_stella)
        #determine best match
        print("-------------------------------")
        print(f"hltipa:{len(good_hltipa)}")
        print(f"laces:{len(good_laces)}")
        print(f"stella:{len(good_stella)}")

        if SHOW_CAMERA:
            cv.imshow('frame', frame)

        e2 = cv.getTickCount()
        time = (e2 - e1)/ cv.getTickFrequency()
        print(f"{1/time} fps")

        if cv.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    sys.exit()

if __name__ == "__main__":
    main()