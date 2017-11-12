import os
import numpy as np
import random as rand
from pathlib import Path
import cv2
#os.chdir(str(Path.home())+'/Dropbox/bike cam/calibration')
#os.chdir('/media/der_emu/bigBaby/bikevid')
os.chdir(r'/home/eg/Videos')
video_file_name = 'vie_01_cutted.mp4'


# define what we are interested in
roi_x = 360
roi_y = 500
#t_start = 4 # 4min
#t_end = 2 # 2min until end

# get some video stats
cap = cv2.VideoCapture(video_file_name)

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( n_frames )

fps = cap.get(cv2.CAP_PROP_FPS)
print("video has: " + str(fps) + " fps")

# we calc the frames we want to keep
#n_0 = fps*t_start*60
#n_1 = n_frames - (fps*t_end*60)


#we get the shape of the cutted video
ret, frame = cap.read()
roi = frame[roi_x:, roi_y:]

print(roi.shape)

cap.release()
cv2.destroyAllWindows()

# we write the new video
#fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fourcc = cv2.CV_FOURCC('m', 'p', '4', 'v')
#out = cv2.VideoWriter(r'/home/eg/Videos/out.avi', fourcc,20.0,roi.shape[0:2])
out = cv2.VideoWriter(r'/home/eg/Videos/out.avi', fourcc,30.0,roi.shape[:-1][::-1])

cap = cv2.VideoCapture(video_file_name)

#n = 1
#started = False
while True:
    #n += 1
    ret, frame = cap.read()
    if ret:
        roi = frame[roi_x:, roi_y:]
    else:
        break
    #if n > n_0:
    #    if not started:
    #        print("started writing")
    #        started = True
    #    out.write(roi)
        out.write(roi)
        cv2.imshow('frame',roi)
    #if n > n_1:
    #    break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()

#for _ in range(n-1):
#    x = rand.randint(0, n-1)
#    if
#    print(str(x))


