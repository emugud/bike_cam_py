    # import cv2
    # import sys
    # sys.path.append("/home/eg/Dropbox/bike_cam/bike_cam_py/models")
    # import keras_pose_estimator as pose
    #
    # sys.path.append("/home/eg/Dropbox/bike_cam/bike_cam_py/local_modules")
    # from config_reader import config_reader
    #
    # from scipy.ndimage.filters import gaussian_filter
    # import matplotlib
    # import pylab as plt
    # import numpy as np
    # import util
    # import math
    #
    # # load the keras model
    # model = pose.init_keras_pose_estimator()
    #
    # # load the config file
    # config_file = "/home/eg/Dropbox/bike_cam/bike_cam_py/models/keras_pose_estimator/config"
    # param, model_params = config_reader(config_file)
    #
    # # load sample video
    # video_file_name=r"/home/eg/Dropbox/bike_cam/test/vie_01_snip1.avi"
    # cap = cv2.VideoCapture(video_file_name)
    #
    # # read video shape
    # vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #
    # # the model is applied at different scales to the image -> avg over predictions
    # multiplier = [x * model_params['boxsize'] / vid_height for x in param['scale_search']]
    # multiplier = multiplier
    #
    # #
    # #  find connection in the specified sequence, center 29 is in the position 15
    # limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
    #            [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
    #            [1,16], [16,18], [3,17], [6,18]]
    #
    # # the middle joints heatmap correpondence
    # mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
    #           [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
    #           [55,56], [37,38], [45,46]]
    #
    # mid_num = 10
    #
    #
    # colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
    #           [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
    #           [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    # cmap = matplotlib.cm.get_cmap('hsv')
    #
    # while(cap.isOpened()):
    #     # get the next frame
    #     ret, frame = cap.read()
    #
    #     # init PAF and heatmap averages
    #     heatmap_avg = None
    #     paf_avg = None
    #
    #     #cv2.imshow('frame', frame)
    #     #if cv2.waitKey(25) & 0xFF == ord('q'):
    #     #    break
    #
    #     #loop over all scales
    #     for m in range(len(multiplier)):
    #
    #         # modify the current frame
    #         # required shape (1, width, height, channels)
    #         scale = multiplier[m]
    #         frame_resized = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    #         frame_padded, pad = util.padRightDownCorner(frame_resized, model_params['stride'], model_params['padValue'])
    #         frame_in = np.transpose(np.float32(frame_padded[:, :, :, np.newaxis]), (3, 0, 1, 2)) / 256 - 0.5
    #
    #         #cv2.imshow('frame_re '+str(m), frame_resized)
    #         #if cv2.waitKey(25) & 0xFF == ord('q'):
    #         #    break
    #
    #         #predict
    #         output_blobs = model.predict(frame_in)
    #
    #          # extract outputs, resize, and remove padding
    #         heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
    #         heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
    #         heatmap = heatmap[:frame_padded.shape[0] - pad[2], :frame_padded.shape[1] - pad[3], :]
    #         heatmap = cv2.resize(heatmap, (vid_width, vid_height), interpolation=cv2.INTER_CUBIC)
    #
    #         #cv2.imshow('heat shoulders'+str(m), heatmap[:,:,2] + heatmap[:,:,5])
    #         #if cv2.waitKey(25) & 0xFF == ord('q'):
    #         #    break
    #
    #
    #         paf = np.squeeze(output_blobs[0]) # output 0 is PAFs
    #         paf = cv2.resize(paf, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
    #         paf = paf[:frame_padded.shape[0]-pad[2], :frame_padded.shape[1]-pad[3], :]
    #         paf = cv2.resize(paf, (vid_width, vid_height), interpolation=cv2.INTER_CUBIC)
    #
    #         if(heatmap_avg is None):
    #             heatmap_avg = heatmap / len(multiplier)
    #         else:
    #             heatmap_avg = heatmap_avg + heatmap / len(multiplier)
    #
    #         if(paf_avg is None):
    #             paf_avg = paf / len(multiplier)
    #         else:
    #             paf_avg = paf_avg + paf / len(multiplier)
    #
    #     dst = cv2.addWeighted(heatmap_avg[:,:,2] + heatmap_avg[:,:,5], 0.5, frame, 0.5, 0)
    #     cv2.imshow('heat avg shoulders', dst)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    #     # update the prediction only for every second frame
    #     while (cap.get(cv2.CAP_PROP_POS_FRAMES) % 10 != 0):
    #             # ge the previous point cloud #not implemented yet
    #         heat_rgb =
    #         dst = cv2.addWeighted(heat_rgb, 0.5, frame, 0.5, 0)
    #
    #         cv2.imshow('heat avg shoulders', dst)
    #
    #         if cv2.waitKey(25) & 0xFF == ord('q'):
    #             break
    #
    #         ret, frame = cap.read()
    #
    #
    # cap.release()


import numpy as np
import os
import cv2
#import tensorflow as tf



# import cv2
# import numpy as np
# import os
# import sys
# sys.path.append('../')
# import tensorflow as tf
# from tf_models.models.research.object_detection.utils import label_map_util
# from tf_models.models.research.object_detection.utils import visualization_utils as vis_util
#
# # load sample video
# video_file_name=r"/mnt/427149F311EAC541/MEGA/bike_cam/data/test/vie_01_snip1.avi"
# cap = cv2.VideoCapture(video_file_name)
#
# ret, input = cap.read()
# input = input[0:360,0:512]
# #a-b
# #| |
# #c-d
#
# b = np.array([215, 93], np.float32)
# a = np.array([120, 98], np.float32)
# d = np.array([460, 185], np.float32)
# c = np.array([290, 210], np.float32)
#
# pts = np.r_[a,b,c,d]
# #pts = np.array([[460, 185], [290, 210], [120, 98], [215, 93]], np.float32)
# #pts = np.array([[460, 185], [290, 210], [120, 98], [215, 93]], np.float32)
# #pts = pts.reshape((-1, 1, 2))
#
# pts = pts.reshape((4,2))
#
# pts2 = np.r_[b+(0.5*(b-d)),a+(0.5*(a-c)),d,c]
# pts2 = pts2.reshape((4,2))
#
# #a-b
# #| |
# #c-d
# a0=np.array([0,0], np.float32)
# b0=np.array([0,512], np.float32)
# c0=np.array([360,0], np.float32)
# d0=np.array([360,512], np.float32)
#
# pts_out1 = np.r_[a0,b0,c0,d0]
# pts_out1 = pts_out1.reshape((4,2))
#
# print(pts)
# print(pts_out1)
#
# M1 = cv2.getPerspectiveTransform(pts, pts_out1)
#
# dst = cv2.warpPerspective(input,M1,(512,360))
#
# pts = pts.reshape((-1, 1, 2)).astype(int)
# input = cv2.polylines(input, [pts], True, (0, 255, 0))
# pts2 = pts2.reshape((-1, 1, 2)).astype(int)
# input = cv2.polylines(input, [pts2], True, (0, 0, 255))
#
# font = cv2.FONT_HERSHEY_SIMPLEX
# text = 'abcd'
# for l in range(4):
#     input = cv2.putText(input,text[l],tuple(pts[l][0]), font, 1,(255,255,255),2,cv2.LINE_AA)
#
# cv2.imshow('object detection', cv2.resize(dst, (0,0), fx=2, fy=2) )
# cv2.imshow('object detection2', cv2.resize(input, (0,0), fx=2, fy=2) )
#
# print(pts)

# load sample video
# vie_01_snip1
#video_file_name=r"/mnt/427149F311EAC541/MEGA/bike_cam/data/test/tres_ciclistas.mov"
#
#
#cap = cv2.VideoCapture(video_file_name)
#
#

#while True:
    # load tf model
#
# import numpy as np
# import os
# import cv2
# import tensorflow as tf
# from cycl_detection import tf_detector
# from local_modules import video_handler
#
# PATH_TO_PROCESS_DIR = r'/mnt/427149F311EAC541/MEGA/bike_cam/tf_learning_data/'
# PATH_TO_CKPT = os.path.join(PATH_TO_PROCESS_DIR, 'ssdv1/frozen_inference_graph.pb')
# PATH_TO_LABELS = r"/mnt/427149F311EAC541/pipeline/label_map.pbtxt"
#
# #tfd = tf_detector.TFDetector(PATH_TO_CKPT,PATH_TO_LABELS,1)
# vh = video_handler.VideoHandler('tres_ciclistas')
#
# ret, frame = vh.get_next()
# #frame = vh.get_current_tracking_zone()
# image = cv2.warpPerspective(frame, vh.M,(vh.maxWidth, vh.maxHeight))
# cv2.imshow('object detection',image)
# if cv2.waitKey(25000) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()
import numpy as np
import os
import cv2
#import tensorflow as tf
#from cycl_detection import tf_detector
from local_modules import video_handler

# PATH_TO_PROCESS_DIR = r'/mnt/427149F311EAC541/MEGA/bike_cam/tf_learning_data/'
# PATH_TO_CKPT = os.path.join(PATH_TO_PROCESS_DIR, 'ssdv1/frozen_inference_graph.pb')
# PATH_TO_LABELS = r"/mnt/427149F311EAC541/pipeline/label_map.pbtxt"
#
#
#
# #tfd = tf_detector.TFDetector(PATH_TO_CKPT,PATH_TO_LABELS,1)
# vh = video_handler.VideoHandler('tres_ciclistas')

# while True:
#     # we go to the next frame
#     ret = vh.next()
#     if not ret:
#         break
#     # and pic the detection_zone
#     det_frame = vh.get_current_detection_zone(True)
#     # and put this into the detector
#     tfd.detect(det_frame)
#     # we update the history of detections in the Detector
#     tfd.update_history()
#
# all_blobs = tfd.get_all_blobs()
# all_blobs_xy = vh.project_detection_to_xy_plane(all_blobs)
#
# import pickle
# pickle.dump( all_blobs, open( "/mnt/427149F311EAC541/MEGA/bike_cam/data/test/blobs.p", "wb" ) )
#
# image = np.zeros(vh.xy_plane_shape, dtype = np.uint8)
#
# for k in range(all_blobs_xy.shape[1]):
#     image = cv2.circle(image, tuple(all_blobs_xy[:,k].astype(int)), 5, (255), -1)
#
# small = cv2.resize(image, (0,0), fx=0.33, fy=0.33)
# cv2.imshow('obj3', small)
#
# if cv2.waitKey(250000) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()
#
# tfd.close()

PATH_TO_PROCESS_DIR = r'/mnt/427149F311EAC541/MEGA/bike_cam/tf_learning_data/'
#PATH_TO_CKPT = os.path.join(PATH_TO_PROCESS_DIR, 'ssdv1/frozen_inference_graph.pb')
#PATH_TO_LABELS = r"/mnt/427149F311EAC541/pipeline/label_map.pbtxt"



#tfd = tf_detector.TFDetector(PATH_TO_CKPT,PATH_TO_LABELS,1)
vh = video_handler.VideoHandler('tres_ciclistas',folder = r"/home/eg/MEGAsync/bike_cam/data/test/")

import pickle
from matplotlib import pyplot as plt
all_blobs = pickle.load(open( "/home/eg/MEGAsync/bike_cam/data/test/blobs.p", "rb" ) )
#  tfd = tf_detector.TFDetector(PATH_TO_CKPT,PATH_TO_LABELS,1)
#all_blobs = pickle.load(open( "/mnt/427149F311EAC541/MEGA/bike_cam/data/test/blobs.p", "rb" ) )
#vh = video_handler.VideoHandler('tres_ciclistas')

ret, whole_frame = vh.get_next()
tracking_frame = vh.get_current_tracking_zone().copy()
detection_frame = vh.get_current_detection_zone().copy()
xy_bird_frame = cv2.warpPerspective(whole_frame, vh.M,(1750,3000))

all_blobs_tracking = vh.mv_detection_to_tracking(all_blobs)
all_blobs_whole = vh.mv_detection_to_input(all_blobs)

prj_src =  np.array(vh.video_meta['floor_proj_src'], dtype=np.float32)
prj_dst =  np.array(vh.video_meta['floor_proj_dst'], dtype=np.float32)

P = cv2.getAffineTransform(prj_src, prj_dst)
all_blobs_whole_prj = cv2.transform(all_blobs_whole.astype(np.float64).reshape(1,-1,2),P)
all_blobs_whole_prj = np.squeeze(all_blobs_whole_prj)

#all_blobs_xy = vh.project_detection_to_xy_plane(all_blobs)
all_blobs_xy = cv2.perspectiveTransform(all_blobs_whole_prj.astype(np.float64).reshape(1,-1,2),vh.M)
all_blobs_xy = np.squeeze(all_blobs_xy)

# we test all visulaizations by moving the blobs to all the sub screens
for k in range(all_blobs.shape[0]):
    detection_frame = cv2.circle(detection_frame, tuple(all_blobs[k].astype(int)), 5, (0, 0, 255), -1)
    tracking_frame = cv2.circle(tracking_frame, tuple(all_blobs_tracking[k].astype(int)), 5, (0, 0, 255), -1)
    whole_frame = cv2.circle(whole_frame, tuple(all_blobs_whole[k].astype(int)), 5, (0, 0, 255), -1)
    xy_bird_frame = cv2.circle(xy_bird_frame, tuple(all_blobs_xy[k].astype(int)), 15, (0, 0, 255), -1)

#print(all_blobs_xy)

plt.imshow(xy_bird_frame)
plt.show()

plt.imshow(whole_frame)
plt.show()

pass

