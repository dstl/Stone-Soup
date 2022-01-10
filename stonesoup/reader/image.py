import datetime
import matplotlib.image as mpimg

from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..types.sensordata import ImageFrame
from .base import FrameReader
from .file import FileReader


class SingleImageFileReader(FileReader, FrameReader):
    """ImageFileReader
    A reader that reads a single image file from a given directory.
    """
    timestamp: datetime.datetime = Property(
        doc="Timestamp given to the returned frame",
        default=None)

    @BufferedGenerator.generator_method
    def frames_gen(self):
        img = mpimg.imread(self.path)*255
        frame = ImageFrame(img, self.timestamp)
        yield self.timestamp, frame
