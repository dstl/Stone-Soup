from datetime import datetime


import numpy as np

from aisutils.decoder import NMEADecoder
from aisutils.parser import NMEAParser, MSSISParser


from stonesoup.reader import DetectionReader, TextFileReader
from stonesoup.types import Detection