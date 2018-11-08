# -*- coding: utf-8 -*-
from pathlib import Path

from ..base import Property

from ..tracker import Tracker
from .base import Writer, TrackWriter, MetricsWriter

from pymap3d import enu2geodetic, ecef2geodetic
from .kmlutils import SSKML

from enum import Enum

import numpy as np

# I don't like this way of doing things!
# Coordinate transformations should happen somewhere else!
class CoordinateSystems(Enum):
    LLA = 1
    ECEF = 2
    ENU = 3


class KMLMetricsWriter(MetricsWriter):
    pass

class KMLTrackWriter(TrackWriter):
    """KML Track Writer"""
    path = Property(Path, doc="File to save data to. Str will be converted to Path")
    coordinate_system = Property(CoordinateSystems, default=None)
    reference_point = Property([float, float, float], default=(0.0, 0.0, 0.0))


    def __init__(self, tracker, path, *args, **kwargs):
        if not isinstance(path, Path):
            path = Path(path)  # Ensure Path
        super().__init__(tracker, path, *args, **kwargs)
        if (type(self.coordinate_system) is not CoordinateSystems):
            raise TypeError("Invalid track coordinate system provided.")
        # if (type(self.reference_point) is not type([])):
        #     raise TypeError("Invalid track reference coordinate provide.")
        # if (len(self.reference_point) != 3):
        #     raise ValueError("Track coordinate reference point must be a tuple of length 3.")
        self._kml = SSKML()
        #self._file = self.path.open('w')

    def write(self):
        

        measurement_model = self.tracker.updater.measurement_model
        tracks = set()
        detections = set()
        tracker_times = []
        for time, ctracks in self.tracker.tracks_gen():
            tracks.update(ctracks)
            detections.update(self.tracker.detector.detections)
            tracker_times.append(time)

        det_pos_array = np.array([detection.state_vector for detection in detections])
        det_time_array = np.array([detection.timestamp for detection in detections])
        tks_ids = []
        tks_pos_matrix = []
        tks_time_matrix = []
        for track in tracks:
            tks_pos_matrix.append(np.array([measurement_model.matrix() @ state.state_vector for state in track.states]))
            tks_time_matrix.append([state.timestamp for state in track.states])
            tks_ids.append(track.id)

        if (self.coordinate_system is CoordinateSystems.ENU):
            det_pos_array_lla = np.array([enu2geodetic(enu_pos[0], enu_pos[1], enu_pos[2], self.reference_point[1],\
               self.reference_point[0], self.reference_point[2]) for enu_pos in det_pos_array])
            tks_pos_matrix_lla = []
            for tks_pos in tks_pos_matrix:
                tks_pos_lla = np.array([enu2geodetic(enu_pos[0], enu_pos[1], enu_pos[2], self.reference_point[1],\
               self.reference_point[0], self.reference_point[2]) for enu_pos in tks_pos])
                tks_pos_matrix_lla.append(tks_pos_lla)


        elif (self.coordinate_system is CoordinateSystems.ECEF):
            det_pos_array_lla = np.array([ecef2geodetic(ecef_pos[0], ecef_pos[1], ecef_pos[2], self.reference_point[1],\
               self.reference_point[0], self.reference_point[2]) for ecef_pos in det_pos_array])
            tks_pos_matrix_lla = []
            for tks_pos in tks_pos_matrix:
                tks_pos_lla = np.array([ecef2geodetic(ecef_pos[0], ecef_pos[1], ecef_pos[2], self.reference_point[1],\
               self.reference_point[0], self.reference_point[2]) for ecef_pos in tks_pos])
                tks_pos_matrix_lla.append(tks_pos_lla)

        elif (self.coordinate_system is LLA):
            # Nothing to do.
            det_pos_array_lla = det_pos_array
            tks_pos_matrix_lla = tks_pos_matrix


        # Now write to kml.
        self._kml.appendTracks(tks_pos_matrix_lla, tks_ids, tks_time_matrix)
        self._kml.appendDetections(det_pos_array_lla, det_time_array)
        self._kml.write(self.path)




        # for time, _ in gen:
        #     data = {'time': time}
        #     if self.tracks_source:
        #         data['tracks'] = self.tracks_source.tracks
        #     if self.detections_source:
        #         data['detections'] = self.detections_source.detections
        #     if self.sensor_data_source:
        #         data['sensor_data'] = self.sensor_data_source.sensor_data
        #     if self.groundtruth_source:
        #         data['groundtruth_paths'] = \
        #             self.groundtruth_source.groundtruth_paths
 



class KMLWriter(Writer):
    """
    Class that writes multiple tracks and/or metrics to a single kml/kmz file.
    """
    pass





