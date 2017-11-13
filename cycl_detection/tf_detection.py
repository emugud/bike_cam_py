import cv2
import numpy as np
import os
import sys
import pandas as pd
sys.path.append('../')

import tensorflow as tf
from tf_models.models.research.object_detection.utils import label_map_util
from tf_models.models.research.object_detection.utils import visualization_utils as vis_util

# PYTHONPATH:
# /home/eg/Dropbox/bike_cam/bike_cam_py/tf_models/models/research:/home/eg/Dropbox/bike_cam/bike_cam_py/tf_models/models/research/slim


# load sample video
# vie_01_snip1
video_file_name=r"/mnt/427149F311EAC541/MEGA/bike_cam/data/test/ber_01_snip.avi"

# load tf model
PATH_TO_PROCESS_DIR = r'/mnt/427149F311EAC541/MEGA/bike_cam/tf_learning_data/'
PATH_TO_CKPT = os.path.join(PATH_TO_PROCESS_DIR,'ssdv1/frozen_inference_graph.pb')
PATH_TO_LABELS = r"/mnt/427149F311EAC541/pipeline/label_map.pbtxt"
NUM_CLASSES = 1

cap = cv2.VideoCapture(video_file_name)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

n_sec = 20
n_fps = 30
n_array = n_sec*n_fps
plt_thrshld = 0.3
perform_heq = True
all_boxes = np.zeros((n_array,100,4),dtype=np.float32)
all_scores = np.zeros((n_array,100),dtype=np.float32)
all_mids = np.zeros((n_array,100,2),dtype=np.float32)
all_corners = np.zeros((n_array,100),dtype=np.bool)
all_keep = np.zeros((n_array,100),dtype=np.bool)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        i = 0
        while True:
            ret, input = cap.read()
            if not ret:
                break
            # ROI for VIE 01: input = input[0:360,0:512]
            #input = input[300:980, 500:1260]
            input = input[300:1080, 500:1920]

            if perform_heq:
                img_yuv = cv2.cvtColor(input, cv2.COLOR_BGR2YUV)
                # equalize the histogram of the Y channel
                img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                # convert the YUV image back to RGB format
                img_yuv = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            else:
                img_yuv=input
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(img_yuv, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            input2 = input.copy()

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                input,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=plt_thrshld)

            cv2.imshow('object detection', input)


            #ymin, xmin, ymax, xmax = box
            box_s = np.squeeze(boxes)

            # this vector stores the midpoints of all detections in the form [[[m1x,m1y],[m2x,m2y],...]]
            mids = np.vstack(((box_s[:, 3] + box_s[:, 1])/2,
                              (box_s[:, 2] + box_s[:, 0])/2)).T.reshape(1,-1,2)

            all_mids = np.append(all_mids, mids, axis=0)
            all_mids = all_mids[1:]

            #  corner is a boolean vector specifying if the box is in a corner
            corner = np.logical_or(np.min(box_s, axis=1) < 0.001, np.max(box_s, axis=1) > 0.999)
            all_corners = np.append(all_corners, corner.reshape(1, -1), axis=0)
            all_corners = all_corners[1:]

            all_boxes = np.append(all_boxes, boxes, axis=0)
            all_boxes = all_boxes[1:]

            all_scores = np.append(all_scores, scores, axis=0)
            all_scores = all_scores[1:]

            keep = np.logical_and(np.squeeze(scores) > plt_thrshld,np.logical_not(corner))
            all_keep = np.append(all_keep, keep.reshape(1, -1), axis=0)
            all_keep = all_keep[1:]


            pts = (mids[keep.reshape(1, -1)]*np.r_[input.shape[1],input.shape[0]]).astype(int)
            all_pts =  (all_mids[all_keep]*np.r_[input.shape[1],input.shape[0]]).astype(int)



            if i == (n_array-1):
                pass

            for k in range(all_pts.shape[0]):
                input2 = cv2.circle(input2, tuple(all_pts[k]), 3, (0, 0, 255), -1)

            #for k in range(all_pts.shape[0])

            cv2.imshow('obj2', input2)
            cv2.imshow('obj3', img_yuv)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            i += 1
            i = i % n_array
#print(all_boxes)


