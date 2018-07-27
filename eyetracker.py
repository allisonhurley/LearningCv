import numpy as np
import cv2
import dlib # dlib for accurate face detection
import imutils # helper functions from pyimagesearch.com


cap = cv2.VideoCapture(0) 

# Face detector
detector = dlib.get_frontal_face_detector()

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2

    # Top left drawing
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right drawing
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left drawing
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right drawing
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    
count = 0

while True:
    if count % 3 != 0:
        # read frames from live web cam stream
        (grabbed, frame) = cap.read()

        # resize the frames to be smaller and switch to gray scale
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Make copies of the frame for transparency processing
        overlay = frame.copy()
        output = frame.copy()

        # set transparency value
        alpha  = 0.5

        # detect faces in the gray scale frame
        face_rects = detector(gray, 0)

        # loop over the face detections
        for i, d in enumerate(face_rects):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()

            # draw a fancy border around the faces
            draw_border(overlay, (x1, y1), (x2, y2), (162, 255, 0), 2, 10, 10)

        # make semi-transparent bounding box
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        # show the frame
        cv2.imshow("Face Detection", output)
        #key = cv2.waitKey(1) & 0xFF
        
    count +=1
    # press q to break out of the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

'''
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()