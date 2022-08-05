#!/usr/bin/env python
"""
Tracking AIS Reports Using Stone Soup
=====================================
"""

# %%
# **Demonstrating the capabilities of Stone Soup using recorded AIS data**
#
# In this notebook we will load a CSV file of |AIS|_ data from the `The Solent`_, to use as
# detections in a Stone Soup tracker.
# We will build the tracker from individual components and display the track output on a map.
#
# .. _The Solent: https://en.wikipedia.org/wiki/The_Solent
# .. |AIS| replace:: :abbr:`AIS (Automatic Identification System)`
# .. _AIS: https://en.wikipedia.org/wiki/Automatic_identification_system


# %%
# Building the detector
# ---------------------
# First we will prepare our detector for the |AIS|_ data, which is in a
# :download:`CSV file <../../demos/SolentAIS_20160112_130211.csv>`, using the Stone Soup generic
# CSV reader.

from stonesoup.reader.generic import CSVDetectionReader
detector = CSVDetectionReader(
    "SolentAIS_20160112_130211.csv",
    state_vector_fields=("Longitude_degrees", "Latitude_degrees"),
    time_field="Time")

# %%
# We use a feeder class to mimic a detector, passing our detections into the tracker, one detection
# per vessel per minute. This is based on assumption that the identifier |MMSI|_ is unique per
# vessel.
#
# .. |MMSI| replace:: :abbr:`MMSI (Maritime Mobile Service Identity)`
# .. _MMSI: https://en.wikipedia.org/wiki/Maritime_Mobile_Service_Identity
import datetime

# Limit updates to one detection per minute...
from stonesoup.feeder.time import TimeSyncFeeder
detector = TimeSyncFeeder(detector, time_window=datetime.timedelta(seconds=60))

# ... but reduce so only one per MMSI
from stonesoup.feeder.filter import MetadataReducer
detector = MetadataReducer(detector, 'MMSI')

# %%
# In this instance, we want to convert the Latitude/Longitude information from the |AIS|_ file to
# |UTM|_ projection, as this approximates to a local Cartesian space better suited for the models
# we will use. The UTM zone will be fixed, based on the first detection processed.
#
# .. |UTM| replace:: :abbr:`UTM (Universal Transverse Mercator)`
# .. _UTM: https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system

from stonesoup.feeder.geo import LongLatToUTMConverter
detector = LongLatToUTMConverter(detector)

# %%
# Creating the models and filters
# -------------------------------
# Now we begin to build our tracker from Stone Soup components. The first component we will build
# is a linear transition model. We create a two-dimensional transition model by combining two
# individual one-dimensional `Ornstein Uhlenbeck`_ models. This is similar to
# :class:`~.ConstantVelocity`, which has an additional :attr:`~.OrnsteinUhlenbeck.damping_coeff`
# which models decaying velocity (we assume vessels will slow down to zero over time, rather than
# continually speed up)
#
# .. _Ornstein Uhlenbeck: https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, OrnsteinUhlenbeck)
transition_model = CombinedLinearGaussianTransitionModel(
    (OrnsteinUhlenbeck(0.5, 1e-4), OrnsteinUhlenbeck(0.5, 1e-4)))

# %%
# Next we build a measurement model to describe the uncertainty on our detections. In this case, we
# are just using the measured position (:math:`[0, 2]` dimensions of state space), assuming from
# ship's |GNSS|_ receiver system with covariance of :math:`\begin{bmatrix}15&0\\0&15\end{bmatrix}`
#
# .. |GNSS| replace:: :abbr:`GNSS (Global Navigation Satellite System)`
# .. _GNSS: https://en.wikipedia.org/wiki/Satellite_navigation
import numpy as np

from stonesoup.models.measurement.linear import LinearGaussian
measurement_model = LinearGaussian(
    ndim_state=4, mapping=[0, 2], noise_covar=np.diag([15, 15]))

# %%
# With these models we can now create the predictor and updater components of the tracker, passing
# in the respective models.
from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

# %%
from stonesoup.updater.kalman import KalmanUpdater
updater = KalmanUpdater(measurement_model)

# %%
# Creating data associators
# -------------------------
# To associate our detections to track objects generate hypotheses using a hypothesiser. In this
# case we are using a :class:`~.Mahalanobis` distance measure. In addition, we are exploiting the
# fact that the detections should have the same |MMSI|_ for a single vessel, by gating out
# detections that don't match the tracks |MMSI|_ (this being populated by detections used to create
# the track).
from stonesoup.gater.filtered import FilteredDetectionsGater
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
measure = Mahalanobis()
hypothesiser = FilteredDetectionsGater(
    DistanceHypothesiser(predictor, updater, measure, missed_distance=3),
    metadata_filter="MMSI"
)

# %%
# We will use a nearest-neighbour association algorithm, passing in the Mahalanobis distance
# hypothesiser built in the previous step.
from stonesoup.dataassociator.neighbour import NearestNeighbour
data_associator = NearestNeighbour(hypothesiser)

# %%
# Creating track initiators and deleters
# --------------------------------------
# We need a method to initiate tracks. For this we will create an initiator component
# and have it generate a :class:`~.GaussianState`. In this case, we'll use a measurement initiator
# which uses the measurement's value and model covariance where possible. In this case, as we are
# using position from |AIS|_, we only need to be concerned about defining the velocity prior. The
# prior we are using has the velocity state vector as zero, and variance of 10m/s.
from stonesoup.types.state import GaussianState
from stonesoup.initiator.simple import SimpleMeasurementInitiator
initiator = SimpleMeasurementInitiator(
    GaussianState([[0], [0], [0], [0]], np.diag([0, 10, 0, 10])),
    measurement_model)

# %%
# As well as an initiator we must also have a deleter. This deleter removes tracks which haven't
# been updated for a defined time period, in this case 10 minutes.
from stonesoup.deleter.time import UpdateTimeDeleter
deleter = UpdateTimeDeleter(time_since_update=datetime.timedelta(minutes=10))

# %%
# Building and running the tracker
# --------------------------------
# With all the individual components specified we can now build our tracker. This is as simple as
# passing in the components.
from stonesoup.tracker.simple import MultiTargetTracker
tracker = MultiTargetTracker(
    initiator=initiator,
    deleter=deleter,
    detector=detector,
    data_associator=data_associator,
    updater=updater,
)

# %%
# Our tracker is built and our detections are ready to be read in from the CSV file, now we set the
# tracker to work. This is done by initiating a loop to generate tracks at each time interval.
# We'll keep a record of all tracks generated over time in a :class:`set` called `tracks`; as a
# :class:`set` we can simply update this with `current_tracks` at each timestep, not worrying about
# duplicates.
tracks = set()
for step, (time, current_tracks) in enumerate(tracker, 1):
    tracks.update(current_tracks)
    if not step % 10:
        print("Step: {} Time: {}".format(step, time))

# %%
# Checking and plotting results
# -----------------------------
# The tracker has now run over the full data set and produced an output tracks. In this data set,
# from the below, we can see that we generated different number of tracks:
len(tracks)

# %%
# Versus the number of unique vessels/|MMSI|_:
len({track.metadata['MMSI'] for track in tracks})

# %%
# We will use the Folium_ python library to display these tracks on a map. The markers can be
# clicked on to reveal the track metadata. (Tracks with same |MMSI|_ will be same colour, but
# colour may be used for multiple |MMSI|_)
#
# .. _Folium: https://python-visualization.github.io/folium/
from collections import defaultdict
from itertools import cycle

import folium
import utm

colour_iter = iter(cycle(
    ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
     'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
     'darkpurple', 'pink', 'lightblue', 'lightgreen',
     'gray', 'black', 'lightgray']))
colour = defaultdict(lambda: next(colour_iter))

m = folium.Map(location=[50.75, -1], zoom_start=10)
for track in tracks:
    points = [
        utm.to_latlon(
            *state.state_vector[measurement_model.mapping, :],
            detector.zone_number, northern=detector.northern, strict=False)
        for state in track]
    folium.PolyLine(points, color=colour[track.metadata.get('MMSI')]).add_to(m)
    folium.Marker(
        points[-1],
        icon=folium.Icon(icon='fa-ship', prefix="fa", color=colour[track.metadata.get('MMSI')]),
        popup="\n".join("{}: {}".format(key, value) for key, value in track.metadata.items())
    ).add_to(m)


# %%

# sphinx_gallery_thumbnail_path = '_static/sphinx_gallery/AIS_Solent_Tracker_thumb.png'
m
