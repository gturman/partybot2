import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0, cv.CAP_V4L)
picnum = 0
beername = 'laces'
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    e1 = cv.getTickCount()

    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', frame)

    e2 = cv.getTickCount()
    time = (e2 - e1)/ cv.getTickFrequency()
    print(f'{1/time} fps')

    if cv.waitKey(1) == ord('s'):
        print('saving photo')
        cv.imwrite(f'./beer/{beername}_{picnum}.jpg', frame)
        picnum = picnum + 1
    elif cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()