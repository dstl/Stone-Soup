import numpy as np
import pytest
import datetime
from stonesoup.types.sensordata import ImageFrame


def test_image_frame():

    # Expect Type error
    with pytest.raises(TypeError):
        ImageFrame()

    image_np = np.random.random((15, 15))
    timestamp = datetime.datetime.now()

    # No timestamp
    frame = ImageFrame(image_np)
    assert np.array_equal(frame.pixels, image_np)
    assert frame.timestamp is None

    # With timestamp
    frame2 = ImageFrame(image_np, timestamp)
    assert np.array_equal(frame.pixels, frame2.pixels)
    assert frame2.timestamp == timestamp
