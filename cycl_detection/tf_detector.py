import numpy as np
import sys
sys.path.append('../')
import tensorflow as tf
from tf_models.models.research.object_detection.utils import label_map_util
from tf_models.models.research.object_detection.utils import visualization_utils as vis_util


class TFDetector():
    def __init__(self,PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES, n_array = 300, plt_thrshld = 0.3):
        self.n_array = n_array
        self.plt_thrshld = plt_thrshld

        self.all_boxes = np.zeros((n_array, 100, 4), dtype=np.float32)
        self.all_scores = np.zeros((n_array, 100), dtype=np.float32)
        self.all_mids = np.zeros((n_array, 100, 2), dtype=np.float32)
        self.all_corners = np.zeros((n_array, 100), dtype=np.bool)
        self.all_keep = np.zeros((n_array, 100), dtype=np.bool)

        self.PATH_TO_CKPT = PATH_TO_CKPT
        self.PATH_TO_LABELS = PATH_TO_LABELS
        self.NUM_CLASSES = NUM_CLASSES

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            self.od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as self.fid:
                self.serialized_graph = self.fid.read()
                self.od_graph_def.ParseFromString(self.serialized_graph)
                tf.import_graph_def(self.od_graph_def, name='')

        self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
                                                                    max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        self.sess = tf.Session(graph=self.detection_graph)

        # Variables that will store detections
        self.boxes = None
        self.scores = None
        self.classes = None

        #  corner is a boolean vector specifying if the box is in a corner
        self.corner = None

        # this vector stores the midpoints of all detections in the form [[[m1x,m1y],[m2x,m2y],...]]
        self.mids = None

        # this vector is a mask over the detection points to save which detections are not boxes and
        # have a score larger than the set threshold
        self.keep = None

        # this variable will store the size of the image the last detection was run on.
        # it is required when getting the blobs
        self.image_shape = None

    def close(self):
        self.sess.close()
        return None


    def detect(self,image):
        self.image_shape = image.shape
        # Definite input and output Tensors for detection_graph
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        (self.boxes, self.scores, self.classes, num) = self.sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        # ymin, xmin, ymax, xmax = box
        _box_s = np.squeeze(self.boxes)

        # this vector stores the midpoints of all detections in the form [[[m1x,m1y],[m2x,m2y],...]]
        self.mids = np.vstack(((_box_s[:, 3] + _box_s[:, 1]) / 2,
                               (_box_s[:, 2] + _box_s[:, 0]) / 2)).T.reshape(1, -1, 2)

        self.corner = np.logical_or(np.min(_box_s, axis=1) < 0.001, np.max(_box_s, axis=1) > 0.999)

        self.keep = np.logical_and(np.squeeze(self.scores) > self.plt_thrshld, np.logical_not(self.corner))

        return num


    def update_history(self):
        self.all_mids = np.append(self.all_mids, self.mids, axis=0)
        self.all_mids = self.all_mids[1:]

        #  corner is a boolean vector specifying if the box is in a corner
        self.all_corners = np.append(self.all_corners, self.corner.reshape(1, -1), axis=0)
        self.all_corners = self.all_corners[1:]

        self.all_boxes = np.append(self.all_boxes, self.boxes, axis=0)
        self.all_boxes = self.all_boxes[1:]

        self.all_scores = np.append(self.all_scores, self.scores, axis=0)
        self.all_scores = self.all_scores[1:]

        self.all_keep = np.append(self.all_keep, self.keep.reshape(1, -1), axis=0)
        self.all_keep = self.all_keep[1:]

    def visualize_detections_on_image_(self,image):
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(self.boxes),
            np.squeeze(self.classes).astype(np.int32),
            np.squeeze(self.scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=self.plt_thrshld)
        return image

    def get_current_blobs(self):
        return (self.mids[self.keep.reshape(1, -1)] * np.r_[self.image_shape[1], self.image_shape[0]]).astype(int)

    def get_all_blobs(self):
        return (self.all_mids[self.all_keep] * np.r_[self.image_shape[1], self.image_shape[0]]).astype(int)
