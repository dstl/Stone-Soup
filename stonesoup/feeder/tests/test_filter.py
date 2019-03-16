# -*- coding: utf-8 -*-
import datetime

from ..filter import MetadataReducer


def test_metadata_reducer(detector):
    feeder = MetadataReducer(detector, metadata_field="colour")

    multi_none = False
    for time, detections in feeder.detections_gen():
        all_colours = [detection.metadata.get('colour')
                       for detection in detections]
        if not multi_none:
            multi_none = len(
                [colour for colour in all_colours if colour is None]) > 1

        colours = [colour for colour in all_colours if colour is not None]
        assert len(colours) == len(set(colours))

        assert "red" in colours
        assert "blue" in colours
        if time < datetime.datetime(2019, 4, 1, 14, 0, 2):
            assert "green" in colours
        else:
            assert "green" not in colours

        assert all(time == detection.timestamp for detection in detections)

    assert multi_none
