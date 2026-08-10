import pytest

try:
    from stonesoup.detector.ultralytics import UltralyticsBoxObjectDetector
except ImportError:
    # Catch optional dependencies import error
    pytest.skip(
        "Skipping due to missing optional dependencies. Usage of the Ultralytics detectors "
        "requires that the optional package dependency 'ultralytics' is installed. This can be "
        "achieved by running 'python -m pip install stonesoup[ultralytics]'.",
        allow_module_level=True
    )


def test_ultralytics_box_object_detector():

    # Expect Type error
    with pytest.raises(TypeError):
        UltralyticsBoxObjectDetector()

    # TODO: Add more tests
