Detectors
=========

.. automodule:: stonesoup.detector
    :no-members:

.. automodule:: stonesoup.detector.base
    :show-inheritance:

Integrating other object-detection frameworks
---------------------------------------------

Stone Soup currently provides a TensorFlow object detector, but detectors based on other machine
learning frameworks can use the same integration pattern. For a video/image detector, the key
contract is to consume a frame and return Stone Soup :class:`~stonesoup.types.detection.Detection`
objects carrying the frame timestamp.

The existing :class:`~stonesoup.detector.tensorflow.TensorFlowBoxObjectDetector` is a useful
reference implementation. Its framework-specific work is isolated in
``_get_detections_from_frame(frame)``: it reads ``frame.pixels``, invokes the model, converts the
model outputs into Stone Soup state vectors and metadata, and returns a set of detections. A
PyTorch-backed detector can follow the same pattern by replacing the TensorFlow inference and
output conversion with the corresponding PyTorch model calls.

For reusable external integrations, prefer implementing the public :class:`~stonesoup.detector.base.Detector`
interface. The video helper used internally by the TensorFlow detector is private and may change
between releases.

TensorFlow
----------
.. automodule:: stonesoup.detector.tensorflow
    :show-inheritance:

