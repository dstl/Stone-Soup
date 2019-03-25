# -*- coding: utf-8 -*-
from pathlib import Path

from ..base import Property

from .base import Writer, TrackWriter, MetricsWriter

from pymap3d import enu2geodetic, ecef2geodetic
from .kmlutils import SSKML

from enum import Enum

import numpy as np


class CoordinateSystems(Enum):
    """
    Class of type enum that lists coordinate systems
    supported by KML writers classes.
    
    Note: LONLAT is used by AISHUB Reader
    """
    LLA = 1
    ECEF = 2
    ENU = 3
    LONLAT = 4


class KMLMetricsWriter(MetricsWriter):
    """
    Class that writes SS metrics to a KML file.
    """
    pass


class KMLTrackWriter(TrackWriter):
    """
    Class that writes SS tracks and detections to a KML file.
    """
    path = Property(
        Path, doc="File to save data to. Str will be converted to Path")
    coordinate_system = Property(CoordinateSystems)
    reference_point = Property([float, float, float], default=(0.0, 0.0, 0.0))

    def __init__(self, tracker, path, *args, **kwargs):
        if not isinstance(path, Path):
            path = Path(path)  # Ensure Path
        super().__init__(tracker, path, *args, **kwargs)
        self._kml = SSKML()

    def write(self):
        measurement_model = self.tracker.updater.measurement_model
        tracks = set()
        detections = set()
        tracker_times = []
        for time, ctracks in self.tracker.tracks_gen():
            tracks.update(ctracks)
            detections.update(self.tracker.detector.detections)
            tracker_times.append(time)

        det_pos_array = np.array([detection.state_vector
                                  for detection in detections])
        det_time_array = np.array([detection.timestamp
                                   for detection in detections])
        tks_ids = []
        tks_pos_matrix = []
        tks_time_matrix = []
        for track in tracks:
            tks_pos_matrix.append(np.array(
                [measurement_model.matrix() @ state.state_vector
                 for state in track.states]))
            tks_time_matrix.append([state.timestamp for state in track.states])
            tks_ids.append(track.id)

        if (self.coordinate_system is CoordinateSystems.ENU):
            det_pos_array_lla = np.array(
                [enu2geodetic(*enu_pos, self.reference_point[1],
                              self.reference_point[0],
                              self.reference_point[2])
                    for enu_pos in det_pos_array])
            tks_pos_matrix_lla = []
            for tks_pos in tks_pos_matrix:
                tks_pos_lla = np.array(
                    [enu2geodetic(*enu_pos, self.reference_point[1],
                                  self.reference_point[0],
                                  self.reference_point[2])
                     for enu_pos in tks_pos])
                tks_pos_matrix_lla.append(tks_pos_lla)
        elif (self.coordinate_system is CoordinateSystems.ECEF):
            det_pos_array_lla = np.array(
                [ecef2geodetic(*ecef_pos, self.reference_point[1],
                               self.reference_point[0],
                               self.reference_point[2])
                    for ecef_pos in det_pos_array])
            tks_pos_matrix_lla = []
            for tks_pos in tks_pos_matrix:
                tks_pos_lla = np.array(
                    [ecef2geodetic(
                        *ecef_pos,
                        self.reference_point[1], self.reference_point[0],
                        self.reference_point[2]) for ecef_pos in tks_pos])
                tks_pos_matrix_lla.append(tks_pos_lla)
        elif (self.coordinate_system is CoordinateSystems.LONLAT):
            det_pos_array_lla = np.array(
                [(lonlat_pos[1], lonlat_pos[0], 0.0)
                 for lonlat_pos in det_pos_array])
            tks_pos_matrix_lla = []
            for tks_pos in tks_pos_matrix:
                tks_pos_lla = np.array(
                    [(lonlat_pos[1], lonlat_pos[0], 0.0)
                     for lonlat_pos in tks_pos])
                tks_pos_matrix_lla.append(tks_pos_lla)
        elif (self.coordinate_system is CoordinateSystems.LLA):
            # Nothing to do.
            det_pos_array_lla = det_pos_array
            tks_pos_matrix_lla = tks_pos_matrix
        # Now write to kml.
        self._kml.append_tracks(tks_pos_matrix_lla, tks_ids, tks_time_matrix)
        self._kml.append_detections(det_pos_array_lla, det_time_array)
        self._kml.write(self.path)


class KMLWriter(Writer):
    """
    Class that writes multiple tracks and metrics to a single kml/kmz file.
    """
    pass
