# -*- coding: utf-8 -*-
"""Feeder classes are for manipulating data going into the tracker.

A feeder's primary role is to take detection data from inputs into the
framework, and feed them into the tracking algorithms. These can then
optionally be used to drop detections, deliver detections out of sequence,
synchronise out of sequence detections, etc. """
from .base import Empty, Feeder

from . import simple  # noqa: F401

__all__ = ['Empty', 'Feeder']
