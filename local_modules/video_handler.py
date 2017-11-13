import cv2
import os
import pickle
import numpy as np


def hist_equalize(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to RGB format
    img_yuv = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_yuv


class VideoHandler():

    def __init__(self,video_name, folder = r"/mnt/427149F311EAC541/MEGA/bike_cam/data/test/"):
        self.video_meta = pickle.load(open(os.path.join(folder,"video_meta.p"), "rb"))[video_name]
        self.video_link= os.path.join(folder,self.video_meta['filename'])
        self.cap = cv2.VideoCapture(self.video_link)

        # compute the perspective transform matrix
        self.src = np.array(self.video_meta['rectangle_image'], dtype=np.float32)
        self.dst = np.array(self.video_meta['rectangle_tracking'], dtype=np.float32)
        self.xy_plane_size = self.video_meta['xy_plane_size']
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)

        # we calculate the transformation matrix for
        # projecting the points on the road
        prj_src = np.array(self.video_meta['floor_proj_src'], dtype=np.float32)
        prj_dst = np.array(self.video_meta['floor_proj_dst'], dtype=np.float32)
        self.P = cv2.getAffineTransform(prj_src, prj_dst)

    def get_birds_eye_view(self):
        frame_as_bird = cv2.warpPerspective(self.frame, self.M, self.xy_plane_size)
        return frame_as_bird

    def get_next(self,hist_eq = False):
        self.ret, self.frame = self.cap.read()
        self.frame_hist_eq = hist_equalize(self.frame)
        if hist_eq:
            return self.ret, self.frame_hist_eq
        else:
            return self.ret, self.frame

    def next(self):
        self.ret, self.frame = self.cap.read()
        if self.ret:
            self.frame_hist_eq = hist_equalize(self.frame)
        return self.ret

    def get_current_detection_zone(self,hist_eq = True):
        if hist_eq:
            return self.frame_hist_eq[
                                    self.video_meta['detection_zone'][0]:self.video_meta['detection_zone'][1],
                                    self.video_meta['detection_zone'][2]:self.video_meta['detection_zone'][3]].copy()
        else:
            return self.frame[self.video_meta['detection_zone'][0]:self.video_meta['detection_zone'][1],
                                    self.video_meta['detection_zone'][2]:self.video_meta['detection_zone'][3]].copy()

    def get_current_tracking_zone(self, hist_eq = False):
        if hist_eq:
            return self.frame_hist_eq[
                                    self.video_meta['tracking_zone'][0]:self.video_meta['tracking_zone'][1],
                                    self.video_meta['tracking_zone'][2]:self.video_meta['tracking_zone'][3]].copy()
        else:
            return self.frame[self.video_meta['tracking_zone'][0]:self.video_meta['tracking_zone'][1],
                                    self.video_meta['tracking_zone'][2]:self.video_meta['tracking_zone'][3]].copy()

    def mv_detection_to_tracking(self, blobs):
        tz = np.array(self.video_meta['tracking_zone'])
        dz = np.array(self.video_meta['detection_zone'])
        tz = np.hstack((tz[2:], tz[:2]))
        dz = np.hstack((dz[2:], dz[:2]))
        return (blobs + (dz - tz)[0::2])

    def mv_detection_to_input(self, blobs):
        dz = np.array(self.video_meta['detection_zone'])
        dz = np.hstack((dz[2:], dz[:2]))
        return (blobs + (dz)[0::2])

    def project_on_road(self, blobs):
        blobs_0 = cv2.transform(blobs.astype(np.float64).reshape(1, -1, 2), self.P)
        blobs_0 = np.squeeze(blobs_0)
        return blobs_0


    def project_detection_to_xy_plane(self,blobs):
        blobs_xy = self.mv_detection_to_input(blobs)
        blobs_xy = self.project_on_road(blobs_xy)
        if len(blobs_xy.shape) > 0:
            blobs_xy = cv2.perspectiveTransform(blobs_xy.astype(np.float64).reshape(1, -1, 2), self.M)
            blobs_xy = np.squeeze(blobs_xy).reshape(-1,2)

        return blobs_xy


if __name__=='__main__':
    t = VideoHandler('tres_ciclistas')
    while True:
        ret = t.next()
        image = t.get_current_tracking_zone()
        #print(image.shape)
        cv2.imshow('obj2', image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
