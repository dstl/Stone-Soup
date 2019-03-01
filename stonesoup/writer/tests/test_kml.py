"""Tests for KML writer classes."""
import os
import numpy as np

from ...reader.aishub import JSON_AISDetectionReader
from ...models.transition.linear import CombinedLinearGaussianTransitionModel
from ...models.transition.linear import ConstantVelocity
from ...models.measurement.linear import LinearGaussian
from ...predictor.kalman import KalmanPredictor
from ...updater.kalman import KalmanUpdater
from ...hypothesiser.distance import MahalanobisDistanceHypothesiser
from ...hypothesiser.filtered import FilteredDetectionsHypothesiser
from ...dataassociator.neighbour import NearestNeighbour
from ...initiator.simple import SinglePointInitiator
from ...types.state import GaussianState
from ...deleter.error import CovarianceBasedDeleter
from ...tracker.simple import MultiTargetTracker

from ..kml import KMLTrackWriter, CoordinateSystems

AIS_JSON_STR = """
[{"ERROR": "false"},
[{"NAME": "Test_Vessel_A", "MMSI": 477266900,
   "LONGITUDE": 2419589, "TIME": "1516320059",
    "LATITUDE": 31185128},
  {"NAME": "Test_Vessel_B", "MMSI": 636092637,
   "LONGITUDE": 2429163, "TIME": "1516320095",
   "LATITUDE": 31168511},
 {"NAME": "Test_Vessel_C", "MMSI": 605086014,
  "LONGITUDE": 2621230, "TIME": "1516320140",
  "LATITUDE": 30749245},
 {"NAME": "Test_Vessel_D", "MMSI": 244620984,
  "LONGITUDE": 2747994, "TIME": "1516320161",
  "LATITUDE": 31151990},
 {"NAME": "Test_Vessel_E", "MMSI": 477266900,
  "LONGITUDE": 2419403, "TIME": "1516320603",
  "LATITUDE": 31185103},
 {"NAME": "Test_Vessel_F", "MMSI": 636092637,
  "LONGITUDE": 2429162, "TIME": "1516320635",
  "LATITUDE": 31168513},
 {"NAME": "Test_Vessel_G", "MMSI": 605086014,
  "LONGITUDE": 2621234, "TIME": "1516320673",
  "LATITUDE": 30749257},
 {"NAME": "Test_Vessel_H", "MMSI": 244620984,
  "LONGITUDE": 2747931, "TIME": "1516320722",
  "LATITUDE": 31151880},
 {"NAME": "Test_Vessel_I", "MMSI": 477266900,
  "LONGITUDE": 2419354, "TIME": "1516320962",
  "LATITUDE": 31185078},
 {"NAME": "Test_Vessel_J", "MMSI": 636092637,
  "LONGITUDE": 2429161, "TIME": "1516321174",
  "LATITUDE": 31168516}]]
 """


def test_kml_trackwriter(tmpdir):
    """Test for KMLTrackWriter."""
    # Write test data to JSON file.
    json_filename = tmpdir.join("test_ais.json")
    json_file = json_filename.open('w')
    json_file.write(AIS_JSON_STR)
    json_file.close()
    # Create demo tracker.
    transition_model = \
        CombinedLinearGaussianTransitionModel(
                                             (ConstantVelocity(1e-6),
                                              ConstantVelocity(1e-6)))
    measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2],
                                       noise_covar=np.diag([1e-3, 1e-3]))
    detections_source = JSON_AISDetectionReader(path=json_filename.strpath)
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)
    hypothesiser = MahalanobisDistanceHypothesiser(predictor,
                                                   updater,
                                                   missed_distance=0.2)
    hypothesiser_wrapper = FilteredDetectionsHypothesiser(hypothesiser,
                                                          "MMSI",
                                                          match_missing=True)
    data_associator = NearestNeighbour(hypothesiser_wrapper)
    initiator = \
        SinglePointInitiator(GaussianState(np.array([[0], [0], [0], [0]]),
                                           np.diag([50, 0.5, 50, 0.5])),
                             measurement_model=measurement_model)
    deleter = CovarianceBasedDeleter(covar_trace_thresh=1E5)
    tracker = MultiTargetTracker(initiator=initiator,
                                 deleter=deleter,
                                 detector=detections_source,
                                 data_associator=data_associator,
                                 updater=updater)
    # Now write KML Output.
    tracker_coord_system = CoordinateSystems.LLA
    output_filename = tmpdir.join("kmltrackwriter_test.kml")
    kml_track_writer = KMLTrackWriter(tracker, str(output_filename),
                                      coordinate_system=tracker_coord_system)
    kml_track_writer.write()
    # Check if the file was created.
    assert os.path.isfile(str(output_filename))
