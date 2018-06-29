# -*- coding: utf-8 -*-
import datetime
from textwrap import dedent

import numpy as np

from ..yaml import YAMLDetectionReader


def test_detections_yaml(tmpdir):
    filename = tmpdir.join("detections.yaml")
    with open(filename, 'w') as file:
        file.write(dedent("""\
            ---
            time: &id001 2018-01-01 14:00:00
            detections: !!set {}
            ...
            ---
            time: &id001 2018-01-01 14:01:00
            detections: !!set
              ? !stonesoup.types.detection.Detection
              - state_vector: !numpy.ndarray
                - [11]
                - [21]
              - timestamp: *id001
              :
            ...
            ---
            time: &id001 2018-01-01 14:02:00
            detections: !!set
              ? !stonesoup.types.detection.Detection
              - state_vector: !numpy.ndarray
                - [12]
                - [22]
              - timestamp: *id001
              :
              ? !stonesoup.types.detection.Clutter
              - state_vector: !numpy.ndarray
                - [12]
                - [22]
              - timestamp: *id001
              :
            """))

    reader = YAMLDetectionReader(filename)

    for n, (time, detections) in enumerate(reader.detections_gen()):
        assert len(detections) == n
        for detection in detections:
            assert np.array_equal(
                detection.state_vector, np.array([[10 + n], [20 + n]]))
            assert detection.timestamp.hour == 14
            assert detection.timestamp.minute == n
            assert detection.timestamp.date() == datetime.date(2018, 1, 1)
            assert detection.timestamp == time
