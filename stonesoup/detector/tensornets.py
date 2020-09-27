# -*- coding: utf-8 -*-
import enum

import numpy as np
try:
    import tensorflow as tf

    try:
        _tf_version = int(tf.__version__.split('.')[0])
    except TypeError:
        # Occurs with Sphinx due to mock imports
        _tf_version = 1
    if _tf_version > 1:
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()

    import tensornets
    from tensornets import datasets
except ImportError as error:
    raise ImportError(
        "Usage of 'stonesoup.detector.tensornets' requires that the optional"
        "package dependencies 'tensorflow' and 'tensornets' are installed.") \
        from error

from ._video import _VideoAsyncDetector
from ..base import Property
from ..types.detection import Detection


class Networks(enum.Enum):
    """TensorNet pre-trained networks, supported by :class:`TensorNetsObjectDetector`

    See TensorNets documentation for more information on these networks.
    """
    def __repr__(self):
        # Suppress value, as not important
        return '<%s.%s>' % (self.__class__.__name__, self.name)

    def _generate_next_value_(name, start, count, last_values):
        # Use name to grab function from tensornets
        return getattr(tensornets, name)

    YOLOv2VOC = enum.auto()  #: YOLOv2 trained against PASCAL VOC dataset
    YOLOv2COCO = enum.auto()  #: YOLOv2 tranined against COCO dataset
    TinyYOLOv2VOC = enum.auto()  #: TinyYOLOv2 trained against PASCAL VOC dataset
    TinyYOLOv2COCO = enum.auto()  #: TinyYOLOv2 trained against COCO dataset
    YOLOv3VOC = enum.auto()  #: YOLOv3 trained against PASCAL VOC dataset
    YOLOv3COCO = enum.auto()  #: YOLOv3 tranined against COCO dataset


class TensorNetsObjectDetector(_VideoAsyncDetector):
    """TensorNets Object Detection class

    This uses pre-trained networks from TensorNets for object detection in video
    frames. Supported networks are listed in :class:`Networks`.
    """
    net: Networks = Property(doc="TensorNet network to use for object detection")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = Networks(self.net)  # Ensure Networks enum
        self._inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
        self._model = self.net.value(self._inputs)
        self._session = tf.Session()

        self._session.run(self._model.pretrained())

    @property
    def class_names(self):
        if 'VOC' in self.net.name:
            class_names = datasets.voc.classnames
        elif 'COCO' in self.net.name:
            class_names = datasets.coco.classnames
        else:
            raise NotImplementedError("Unsupported network {!r}".format(self.net))
        return class_names

    def _run(self, image):
        if 'YOLOv2' in self.net.name:
            fetches = self._model
        elif 'YOLOv3' in self.net.name:
            fetches = self._model.preds
        else:
            raise NotImplementedError("Unsupported network {!r}".format(self.net))
        return self._session.run(fetches, {self._inputs: self._model.preprocess(image)})

    def _get_detections_from_frame(self, frame):
        image_np_expanded = np.expand_dims(frame.pixels, axis=0)
        preds = self._run(image_np_expanded)
        boxes = self._model.get_boxes(preds, frame.pixels.shape[:2])

        detections = set()
        for class_id, (class_name, class_boxes) in enumerate(zip(self.class_names, boxes)):
            for box in class_boxes:
                metadata = {
                    "raw_box": box,
                    "class": {'name': class_name, 'id': class_id},
                    "class_name": class_name,
                    "class_id": class_id,
                    "score": box[-1],
                }
                # Transform box to be in format (x, y, w, h)
                detection = Detection(
                    [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                    timestamp=frame.timestamp,
                    metadata=metadata)
                detections.add(detection)

        return detections
