import numpy as np
from datetime import datetime

from .base import Type
from ..base import Property


class SensorData(Type):
    """Sensor Data type"""


class ImageFrame(SensorData):
    """ Image Frame type used to represent a simple image/video frame """

    pixels: np.ndarray = Property(
        doc="An array of shape (w,h,x) containing the individual pixel  values, where w:width, "
            "h:height and x may vary depending on the color format")
    timestamp: datetime = Property(doc="An optional timestamp", default=None)

    def __bool__(self):
        return len(self.pixels) > 0
