from datetime import datetime, timedelta


import numpy as np

from aisutils.decoder import NMEADecoder
from aisutils.parser import NMEAParser, MSSISParser


from stonesoup.reader import DetectionReader, TextFileReader
from stonesoup.types import Detection


class NMEAReader(DetectionReader):
	bbox = Property(
        (float, float, float, float),
        default=None,
        doc="left, bottom, right, top")
    start_time = Property(
        datetime,
        doc="Time to begin extracting NMEA messages.")
    end_time = Property(
        datetime,
        doc="Time to finish extracting NMEA messages.")
    number_steps = Property(
        int, default=10, doc="Number of time steps to run for")