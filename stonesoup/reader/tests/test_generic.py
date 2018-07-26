# -*- coding: utf-8 -*-
import datetime
from textwrap import dedent

import numpy as np

from ..generic import CSVDetectionReader


def test_csv(tmpdir):
    csv_filename = tmpdir.join("test.csv")
    with csv_filename.open('w') as csv_file:
        csv_file.write(dedent("""\
            x,y,z,t
            10,20,30,2018-01-01T14:00:00Z
            11,21,31,2018-01-01T14:01:00Z
            12,22,32,2018-01-01T14:02:00Z
            """))

    csv_reader = CSVDetectionReader(csv_filename, ["x", "y"], "t")
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
