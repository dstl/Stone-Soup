# -*- coding: utf-8 -*-
"""Test for feeder.simple module"""
import pytest

from ..base import Empty
from ..simple import FIFOFeeder
from ...types import Detection


def test_fifo():
    """FIFOFeeder test"""
    feeder = FIFOFeeder()
    detections = [Detection() for _ in range(10)]
    for detection in detections:  # Add them in order…
        feeder.put(detection)
    for detection in detections:  # …then check they come out in order
        assert feeder.get() is detection
    with pytest.raises(Empty):
        feeder.get()
