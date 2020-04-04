import datetime
import os
import tarfile
import urllib
import numpy as np
import cv2
from copy import copy

from object_detection.utils import visualization_utils as vis_util
# from stonesoup.feeder.filter import MetadataValueFilter

from stonesoup.reader.video import VideoClipReader  # , FFmpegVideoStreamReader
from stonesoup.detector.tensorflow import TensorFlowBoxObjectDetector


def detection_to_bbox(state_vector):
    x_min, y_min, width, height = (state_vector[0, 0],
                                   state_vector[1, 0],
                                   state_vector[2, 0],
                                   state_vector[3, 0])
    return np.array([y_min, x_min, y_min + height, x_min + width])


def draw_detections(frame, detections, category_index, score_threshold=0.5):
    if len(detections):
        boxes = np.array([detection_to_bbox(detection.state_vector)
                          for detection in detections])
        classes = np.array([detection.metadata["class"]["id"]
                            for detection in detections])
        scores = np.array([detection.metadata["score"]
                           for detection in detections])
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            boxes,
            classes,
            scores,
            category_index,
            min_score_thresh=score_threshold,
            use_normalized_coordinates=False,
            line_thickness=1,
            max_boxes_to_draw=200,
            # skip_scores=True,
            # skip_labels=True
        )
    return frame


#####################################################################################
# This is an optional step to download:                                             #
#   1) The TensorFlow object detection model                                        #
#   2) The label map file, mapping object ids to names for visualisation            #
# ###################################################################################

# What model to download.
# Models can bee found here:
# github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# Download Model
if not os.path.exists(os.path.join(os.getcwd(), MODEL_FILE)):
    print("Downloading model")
    opener = urllib.request.URLopener()
    opener.retrieve(MODELS_DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

# List of the strings that is used to add correct label for each box.
LABEL_FILE = 'mscoco_label_map.pbtxt'
LABELS_DOWNLOAD_BASE = \
    'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
PATH_TO_LABELS = LABEL_FILE
# Download labels
if not os.path.exists(os.path.join(os.getcwd(), LABEL_FILE)):
    print("Downloading label")
    urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILE, PATH_TO_LABELS)
######################################################################################


# VIDEO CLIP READER ##################################################################
def arrange_bgr(image):
    return image[:, :, [2, 1, 0]]  # Rearrange RGB to BGR


VIDEO_PATH = r'.\sample.mp4'
start_time = datetime.timedelta(minutes=0, seconds=0)
end_time = None  # datetime.timedelta(minutes=3, seconds=20)
video_reader = VideoClipReader(VIDEO_PATH, start_time, end_time)
video_reader.clip = video_reader.clip.fl_image(arrange_bgr)
run_async = False
######################################################################################


# STREAM READER ######################################################################
# STREAM_URL = 'rtsp://192.168.55.10:554/1/h264minor'
# # The following options ensure real-time streaming
# in_opts = {'threads': 1, 'fflags': 'nobuffer'}
# out_opts = {'format': 'rawvideo', 'pix_fmt': 'bgr24'}
# video_reader = FFmpegVideoStreamReader(STREAM_URL, input_opts=in_opts, output_opts=out_opts)
# run_async = True
######################################################################################

detector = TensorFlowBoxObjectDetector(video_reader, PATH_TO_CKPT,
                                       PATH_TO_LABELS, run_async=run_async)
category_index = detector.category_index
# detector = MetadataValueFilter(detector, 'class', lambda x: x['name'] == 'car')
for timestamp, detections in detector:

    print(timestamp)
    # print(detections)
    print("----------------")
    # Extract frame from reader
    frame = video_reader.frame
    pixels = copy(frame.pixels)
    pixels = draw_detections(pixels, detections,
                             category_index,
                             score_threshold=0.2)

    # Display image
    cv2.imshow('VIDEO', pixels)
    cv2.waitKey(1)
