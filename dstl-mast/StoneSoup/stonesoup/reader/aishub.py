import json
from datetime import datetime

import numpy as np

from .base import DetectionReader
from .file import TextFileReader
from ..types.detection import Detection
from stonesoup.buffered_generator import BufferedGenerator


class JSON_AISDetectionReader(DetectionReader, TextFileReader):
    """A simple detection reader for JSON files of AIS (maritime
    transponder) detections.

    This particular JSON AIS reader is written to read the JSON format
    used by files downloaded from 'www.aishub.net' (must be a contributor
    to their AIS database to download their files).


    JSON format::

        [{"ERROR": "false"},
        [{"NAME": "MARTCILINO", "MMSI": 205466990, "LONGITUDE": 3078401,
        "TIME": "1516233686", "LATITUDE": 31215032, ...},
        {"NAME": "MARTCILINO", "MMSI": 205466990, "LONGITUDE": 3064592,
        "TIME": "1516234275", "LATITUDE": 31227463, ...}]]

    Notes
    -----
    * TIME is in Linux Epoch format
    * LONGITUDE and LATITUDE are (long/lat degrees)*(600,000)
    * MMSI is unique ship identifier
    * The AIS detection attributes for lattitude, longitude, and timestamp are
      saved as the attributes of a 'Detection'; the other attributes are saved
      as the dictionary 'metadata' attribute of a 'Detection'.
    """

    # path - inherited from 'TextFileReader'->'FileReader'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @BufferedGenerator.generator_method
    def detections_gen(self):

        # read data from the JSON file
        with self.path.open(encoding=self.encoding) as json_file:
            file_data = json.load(json_file)
            file_data = file_data[1]

            # sort the imported AIS Detection list by time of detection
            file_data.sort(key=lambda x: x['TIME'])

            for record in file_data:

                # extract latitude and longitude values from JSON record
                lat_value = float(record['LATITUDE'])/600000
                lon_value = float(record['LONGITUDE'])/600000

                # extract timestamp values from JSON record
                time_value = datetime.utcfromtimestamp(float(
                        record['TIME']))

                # delete lat, lon, timestamp from JSON record;
                # the rest is metadata
                del record['LATITUDE']
                del record['LONGITUDE']
                del record['TIME']

                # form Detection object from JSON record
                detect = Detection(
                            np.array([[lon_value], [lat_value]],
                                     dtype=np.float32),
                            time_value, metadata=record)
                yield time_value, {detect}
