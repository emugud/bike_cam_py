import cv2
import sys
sys.path.append("/home/eg/Dropbox/bike_cam/bike_cam_py/models")
import keras_pose_estimator as pose

sys.path.append("/home/eg/Dropbox/bike_cam/bike_cam_py/local_modules")
from config_reader import config_reader

from scipy.ndimage.filters import gaussian_filter
import matplotlib
import pylab as plt
import numpy as np
import util
import math

# load the keras model
model = pose.init_keras_pose_estimator()

# load the config file
config_file = "/home/eg/Dropbox/bike_cam/bike_cam_py/models/keras_pose_estimator/config"
param, model_params = config_reader(config_file)

# load sample video
video_file_name=r"/home/eg/Videos/vie_01_snip1.avi"
cap = cv2.VideoCapture(video_file_name)

# read video shape
vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# the model is applied at different scales to the image -> avg over predictions
multiplier = [x * model_params['boxsize'] / vid_height for x in param['scale_search']]

#
#  find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
           [1,16], [16,18], [3,17], [6,18]]

# the middle joints heatmap correpondence
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
          [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
          [55,56], [37,38], [45,46]]

mid_num = 10


colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
cmap = matplotlib.cm.get_cmap('hsv')

while(cap.isOpened()):
    # get the next frame
    ret, frame = cap.read()

    # update the prediction only for every second frame
    #if(cap.get(cv2.CAP_PROP_POS_FRAMES) % 2 == 0):
    #    # ge the previous point cloud #not implemented yet
    #    cv2.imshow('frame', frame)
    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break

    # init PAF and heatmap averages
    heatmap_avg = None
    paf_avg = None

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    #loop over all scales
    for m in range(len(multiplier)):

        # modify the current frame
        # required shape (1, width, height, channels)
        scale = multiplier[m]
        frame_resized = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        frame_padded, pad = util.padRightDownCorner(frame_resized, model_params['stride'], model_params['padValue'])
        frame_in = np.transpose(np.float32(frame_padded[:, :, :, np.newaxis]), (3, 0, 1, 2)) / 256 - 0.5

        #cv2.imshow('frame_re '+str(m), frame_resized)
        #if cv2.waitKey(25) & 0xFF == ord('q'):
        #    break

        #predict
        output_blobs = model.predict(frame_in)

         # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:frame_padded.shape[0] - pad[2], :frame_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (vid_width, vid_height), interpolation=cv2.INTER_CUBIC)

        #cv2.imshow('heat shoulders'+str(m), heatmap[:,:,2] + heatmap[:,:,5])
        #if cv2.waitKey(25) & 0xFF == ord('q'):
        #    break


        paf = np.squeeze(output_blobs[0]) # output 0 is PAFs
        paf = cv2.resize(paf, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
        paf = paf[:frame_padded.shape[0]-pad[2], :frame_padded.shape[1]-pad[3], :]
        paf = cv2.resize(paf, (vid_width, vid_height), interpolation=cv2.INTER_CUBIC)

        if(heatmap_avg is None):
            heatmap_avg = heatmap / len(multiplier)
        else:
            heatmap_avg = heatmap_avg + heatmap / len(multiplier)

        if(paf_avg is None):
            paf_avg = paf / len(multiplier)
        else:
            paf_avg = paf_avg + paf / len(multiplier)

    cv2.imshow('heat avg shoulders', heatmap_avg[:,:,2] + heatmap_avg[:,:,5])
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break



    all_peaks = []
    peak_counter = 0

    for part in range(19 - 1):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > param['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]

        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    import math

    connection_all = []
    special_k = []


    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                      for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                      for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * vid_height / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append(
                            [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    print("found = 2")
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    for i in range(18):
        rgba = np.array(cmap(1 - i / 18. - 1. / 36))
        rgba[0:3] *= 255
        for j in range(len(all_peaks[i])):
            cv2.circle(frame, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    #cv2.imshow('final', frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break


