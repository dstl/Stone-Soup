# -*- coding: utf-8 -*-

try:
    import tensorflow as tf
    from object_detection.utils import label_map_util as lm_util
except ImportError as error:
    raise ImportError(
        "Usage of the TensorFlow detectors requires that TensorFlow and the research module of "
        "the TensorFlow Model Garden are installed. A quick guide on how  to set these up can be "
        "found here: "
        "https://tensorflow2objectdetectioninstallation.readthedocs.io/en/latest/ ")\
        from error

import threading
import numpy as np
from copy import copy
from pathlib import Path

from .base import Detector
from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..types.detection import Detection


class TensorFlowBoxObjectDetector(Detector):
    """TensorFlowBoxObjectDetector

    A box object detector that generates detections of objects, in the form of bounding boxes, 
    from image/video frames using a `TensorFlow object detection model 
    <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md>`_.
    
    The detections generated by the box detector have the form of bounding boxes that capture 
    the area of the frame where an object is detected. Each bounding box is represented by a 
    vector of the form ``[x, y, w, h]``, where ``x, y`` denote the relative coordinates of the 
    top-left corner, while ``w, h`` denote the relative width and height of the bounding box. 
    
    Additionally, each detection carries the following meta-data fields:

    - ``raw_box``: The raw bounding box, as generated by TensorFlow.

    - ``class``: A dict with keys ``id`` and ``name`` relating to the id and name of the 
    detection class.

    - ``score``: A float in the range ``(0, 1]`` indicating the detector's confidence.
    
    Important
    ---------
    Use of this class requires that TensorFlow and the research module of the TensorFlow Model 
    Garden are installed. A quick guide on how to set these up can be found 
    `here <https://tensorflow2objectdetectioninstallation.readthedocs.io/en/latest/>`_. 
    
    """  # noqa

    model_path = Property(
        Path,
        doc="Path to frozen detection graph (``*.pb`` file). This is the "
            "actual model that is used for the object detection")

    labels_path = Property(
        Path,
        doc="Path to label map (``*.pbtxt`` file). This is the "
            "file that contains mapping of object/class ids to "
            "meaningful names")

    run_async = Property(
        bool,
        doc="If set to ``True``, the detector will digest frames from the "
            "reader asynchronously and only perform detection on the last "
            "frame digested. This is suitable when the detector is applied to "
            "readers generating a live feed "
            "(e.g. :class:`~.FFmpegVideoStreamReader`), "
            "where real-time processing is paramount. Defaults to ``False``",
        default=False)

    session_config = Property(
        tf.compat.v1.ConfigProto,
        doc="A `ConfigProto <https://www.tensorflow.org/code/tensorflow/core/protobuf/config"
            ".proto>`_ protocol buffer with configuration options for the TensorFlow session. "
            "Defaults to ``None``",
        default=None
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialise TF graph and category index
        self._graph = tf.Graph()
        with self._graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.GFile(str(self.model_path), 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self._sess = tf.compat.v1.Session(graph=self._graph, config=self.session_config)
        self.category_index = lm_util.create_category_index_from_labelmap(
            self.labels_path,
            use_display_name=True)

        # Variables used in async mode
        if self.run_async:
            self._buffer = None
            # Initialise frame capture thread
            self._capture_thread = threading.Thread(target=self._capture)
            self._capture_thread.daemon = True
            self._thread_lock = threading.Lock()
            self._capture_thread.start()

    @BufferedGenerator.generator_method
    def detections_gen(self):
        """Returns a generator of detections for each frame.

        Yields
        ------
        : :class:`datetime.datetime`
            Datetime of current time step
        : set of :class:`~.Detection`
            Detections generated in the time step. The detection state
            vector is of the form ``(x, y, w, h)``, where ``x, y`` denote
            the relative coordinates of the top-left corner of the bounding
            box containing the object, while ``w, h`` denote the relative
            width and height of the bounding box. Additionally, each detection
            carries the following meta-data fields:

            - ``raw_box``: The raw bounding box, as generated by TensorFlow.

            - ``class``: A dict with keys ``id`` and ``name`` relating to the \
            id and name of the detection class, as specified in the label map.

            - ``score``: A float in the range ``(0, 1]`` indicating the \
            detector's confidence
        """

        if self.run_async:
            yield from self._detections_gen_async()
        else:
            yield from self._detections_gen()

    def _capture(self):
        for timestamp, frame in self.sensor:
            self._thread_lock.acquire()
            self._buffer = frame
            self._thread_lock.release()

    def _detections_gen(self):
        for timestamp, frame in self.sensor:
            detections = self._get_detections_from_frame(frame)
            yield timestamp, detections

    def _detections_gen_async(self):
        while self._capture_thread.is_alive():
            if self._buffer is not None:
                self._thread_lock.acquire()
                frame = copy(self._buffer)
                self._buffer = None
                self._thread_lock.release()

                detections = self._get_detections_from_frame(frame)

                yield frame.timestamp, detections

    def _get_detections_from_frame(self, frame):
        # Expand dimensions since the model expects images
        # to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(frame.pixels, axis=0)
        # Extract image tensor
        image_tensor = self._graph.get_tensor_by_name(
            'image_tensor:0')
        # Extract detection boxes
        boxes = self._graph.get_tensor_by_name(
            'detection_boxes:0')
        # Extract detection scores
        scores = self._graph.get_tensor_by_name(
            'detection_scores:0')
        # Extract detection classes
        classes = self._graph.get_tensor_by_name(
            'detection_classes:0')
        # Extract number of detections
        num_detections = self._graph.get_tensor_by_name(
            'num_detections:0')

        # Perform detection
        (boxes, scores, classes, num_detections) = self._sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        num_detections = int(num_detections)
        classes = classes[0, :num_detections].astype(np.int32)
        boxes = boxes[0, :num_detections]
        scores = scores[0, :num_detections]

        # Empty detection set
        frame_height, frame_width, _ = frame.pixels.shape
        detections = set()
        for box, class_, score in zip(boxes, classes, scores):
            metadata = {
                "raw_box": box,
                "class": self.category_index[class_],
                "score": score
             }
            # Transform box to be in format (x, y, w, h)
            box_xy = np.array([[box[1]*frame_width],
                               [box[0]*frame_height],
                               [(box[3] - box[1])*frame_width],
                               [(box[2] - box[0])*frame_height]])
            detection = Detection(
                state_vector=box_xy,
                timestamp=frame.timestamp,
                metadata=metadata)
            detections.add(detection)

        return detections
