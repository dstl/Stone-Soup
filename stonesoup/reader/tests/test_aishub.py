import datetime

import numpy as np

from stonesoup.reader.aishub import JSON_AISDetectionReader


def test_aishub(tmpdir):

    # create test JSON file
    json_filename = tmpdir.join("test.json")
    json_file = open(json_filename, 'w')
    json_file.write("""[{"ERROR": "false"},
            [{"NAME": "DETTIFOSS", "MMSI": 304159000, "LONGITUDE": 15000000,
                     "TIME": "1527689580", "LATITUDE": 30000000},
             {"NAME": "DETTIFOSS", "MMSI": 304159000, "LONGITUDE": 15600000,
                     "TIME": "1527689640", "LATITUDE": 30600000}]]""")
    json_file.close()

    # read the JSON file with a "JSON_AISDetectionReader()"
    JSON_reader = JSON_AISDetectionReader(path=json_filename)
    detections = [
        detection
        for _, detections in JSON_reader.detections_gen()
        for detection in detections]

    # verify that all of the AIS records from the JSON file were read correctly
    for n, detection in enumerate(detections):

        # verify the LATITUDE and LONGITUDE values
        assert np.array_equal(
                    detection.state_vector, np.array([[25 + n], [50 + n]]))

        # verify the TIME values
        assert detection.timestamp.hour == 14
        assert detection.timestamp.minute == 13 + n
        assert detection.timestamp.date() == datetime.date(2018, 5, 30)

        # verify the metadata attributes
        assert detection.metadata['NAME'] == 'DETTIFOSS'
        assert detection.metadata['MMSI'] == 304159000
