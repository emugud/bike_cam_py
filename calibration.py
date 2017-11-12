import glob
import os
import numpy as np
import random as rand
from pathlib import Path
import cv2

os.chdir(str(Path.home())+'/Dropbox/bike cam/calibration')
os.chdir('/media/der_emu/bigBaby/bikevid')





video_file_name = 'calib_a80.mp4'
cap = cv2.VideoCapture(video_file_name)



def count_frames_manual(video):
    # initialize the total number of frames read
    total = 0

    # loop over the frames of the video
    while True:
        # grab the current frame
        (grabbed, frame) = video.read()

        # check to see if we have reached the end of the
        # video
        if not grabbed:
            break

        # increment the total number of frames read
        total += 1

    # return the total number of frames in the video file
    return total



# we get video

cap = cv2.VideoCapture(video_file_name)
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
n_frames = count_frames_manual(cap)
cap.release()
cap = cv2.VideoCapture(video_file_name)

def get_random_frame(video_file_name, n_frames, fps, n_th):
    available = list(range(n_frames))
    available = [item for index, item in enumerate(available) if (index + 1) % n_th != 0]
    #print(len(available))
    while len(available) > 0:
        x = available[rand.randint(0, len(available)-1)]
        yield x
        available.remove(x)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


for frame in get_random_frame(video_file_name,n_frames, fps, 100):
    cap.set(cv2.CAP_PROP_POS_MSEC, frame / fps * 1000)
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        cv2.imshow('img',img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
