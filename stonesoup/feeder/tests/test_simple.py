# -*- coding: utf-8 -*-
"""Test for feeder.simple module"""
import pytest

from ..simple import FIFOFeeder
from ...types import Detection


def test_fifo():
    """FIFOFeeder test"""
    feeder = FIFOFeeder()
    detections = [Detection() for _ in range(10)]
    for detection in detections:  # Add them in order…
        feeder.append(detection)
    for detection in detections:  # …then check they come out in order
        assert feeder.popleft() is detection
    with pytest.raises(IndexError):
        feeder.popleft()
