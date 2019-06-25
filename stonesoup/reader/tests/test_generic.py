# -*- coding: utf-8 -*-
import datetime
from textwrap import dedent

import numpy as np

from ..generic import CSVDetectionReader


def test_csv(tmpdir):
    csv_filename = tmpdir.join("test.csv")
    with csv_filename.open('w') as csv_file:
        csv_file.write(dedent("""\
            x,y,z,identifier,t
            10,20,30,22018332,2018-01-01T14:00:00Z
            11,21,31,22018332,2018-01-01T14:01:00Z
            12,22,32,22018332,2018-01-01T14:02:00Z
            """))

    # run test with:
    #   - 'metadata_fields' for 'CSVDetectionReader' == default
    #   - copy all metadata items
    csv_reader = CSVDetectionReader(csv_filename.strpath, ["x", "y"], "t")
    detections = [
        detection
        for _, detections in csv_reader.detections_gen()
        for detection in detections]

    for n, detection in enumerate(detections):
        assert np.array_equal(
            detection.state_vector, np.array([[10 + n], [20 + n]]))
        assert detection.timestamp.hour == 14
        assert detection.timestamp.minute == n
        assert detection.timestamp.date() == datetime.date(2018, 1, 1)
        assert isinstance(detection.metadata, dict)
        assert len(detection.metadata) == 2
        assert 'z' in detection.metadata.keys()
        assert 'identifier' in detection.metadata.keys()
        assert int(detection.metadata['z']) == 30 + n
        assert detection.metadata['identifier'] == '22018332'

    # run test with:
    #   - 'metadata_fields' for 'CSVDetectionReader' contains
    #       'z' but not 'identifier'
    #   - 'time_field_format' is specified
    csv_reader = CSVDetectionReader(csv_filename.strpath, ["x", "y"], "t",
                                    time_field_format="%Y-%m-%dT%H:%M:%SZ",
                                    metadata_fields=["z"])
    detections = [
        detection
        for _, detections in csv_reader.detections_gen()
        for detection in detections]

    for n, detection in enumerate(detections):
        assert np.array_equal(
            detection.state_vector, np.array([[10 + n], [20 + n]]))
        assert detection.timestamp.hour == 14
        assert detection.timestamp.minute == n
        assert detection.timestamp.date() == datetime.date(2018, 1, 1)
        assert isinstance(detection.metadata, dict)
        assert len(detection.metadata) == 1
        assert 'z' in detection.metadata.keys()
        assert int(detection.metadata['z']) == 30 + n

    # run test with:
    #   - 'metadata_fields' for 'CSVDetectionReader' contains
    #       column names that do not exist in CSV file
    #   - 'time' field represented as a Unix epoch timestamp
    with csv_filename.open('w') as csv_file:
        csv_file.write(dedent("""\
            x,y,z,identifier,t
            10,20,30,22018332,1514815200
            11,21,31,22018332,1514815260
            12,22,32,22018332,1514815320
            """))

    csv_reader = CSVDetectionReader(csv_filename.strpath, ["x", "y"], "t",
                                    metadata_fields=["heading"],
                                    timestamp=True)
    detections = [
        detection
        for _, detections in csv_reader.detections_gen()
        for detection in detections]

    for n, detection in enumerate(detections):
        assert np.array_equal(
            detection.state_vector, np.array([[10 + n], [20 + n]]))
        assert detection.timestamp.hour == 14
        assert detection.timestamp.minute == n
        assert detection.timestamp.date() == datetime.date(2018, 1, 1)
        assert isinstance(detection.metadata, dict)
        assert len(detection.metadata) == 0

def test_tsv(tmpdir):
    csv_filename = tmpdir.join("test.csv")
    with csv_filename.open('w') as csv_file:
        csv_file.write(dedent("""\
            x\ty\tz\tidentifier\tt
            10\t20\t30\t22018332\t2018-01-01T14:00:00Z
            11\t21\t31\t22018332\t2018-01-01T14:01:00Z
            12\t22\t32\t22018332\t2018-01-01T14:02:00Z
            """))

    # run test with:
    #   - 'metadata_fields' for 'CSVDetectionReader' == default
    #   - copy all metadata items
    csv_reader = CSVDetectionReader(csv_filename.strpath, ["x", "y"], "t", csv_options={'dialect': 'excel-tab'})
    detections = [
        detection
        for _, detections in csv_reader.detections_gen()
        for detection in detections]

    for n, detection in enumerate(detections):
        assert np.array_equal(
            detection.state_vector, np.array([[10 + n], [20 + n]]))
        assert detection.timestamp.hour == 14
        assert detection.timestamp.minute == n
        assert detection.timestamp.date() == datetime.date(2018, 1, 1)
        assert isinstance(detection.metadata, dict)
        assert len(detection.metadata) == 2
        assert 'z' in detection.metadata.keys()
        assert 'identifier' in detection.metadata.keys()
        assert int(detection.metadata['z']) == 30 + n
        assert detection.metadata['identifier'] == '22018332'
