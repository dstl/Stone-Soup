# -*- coding: utf-8 -*
import pytest

try:
    from stonesoup.detector.tensorflow import TensorFlowBoxObjectDetector
except ImportError:
    # Catch optional dependencies import error
    pytest.skip(
        "Skipping due to missing optional dependencies. Usage of the TensorFlow detectors "
        "requires that TensorFlow and the TensorFlow Object Detection API are installed. A quick "
        "guide on how to set these up can be found here: "
        "https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html",
        allow_module_level=True
    )


def test_tensorflow_box_object_detector():

    # Expect Type error
    with pytest.raises(TypeError):
        TensorFlowBoxObjectDetector()

    # TODO: Add more tests
