#!/usr/bin/env python

"""
Tracking Groundtruth ADS-B Data by Simulating Radar Detections
==============================================================
"""
# %%
# Introduction
# ------------
# Our goal in this demonstration is to plot time series data of Stone Soup's :class:`~.MultiTargetTracker`
# being applied to air traffic over and surrounding the UK.
# We will establish the individual components required for our tracker, including simulating radar
# detection data from our groundtruth, which will be read in from a CSV file of ADS–B data sourced
# from `The OpenSky Network`_ [#]_ [#]_. Finally, we will plot our tracks using the Folium plugin
# `TimestampedGeoJson`_.
#
# .. _The OpenSky Network: https://www.opensky-network.org
# .. _TimestampedGeoJson:  https://python-visualization.github.io/folium/plugins.html

# %%
# Reading the CSV File
# ---------------------
# To read in our groundtruth data from a CSV file, we can use Stone Soup’s
# :class:`~.CSVGroundTruthReader`. To convert our longitude and latitude data to Universal
# Transverse Mercator (UTM) projection, we will use :class:`~.LongLatToUTMConverter`.

from stonesoup.reader.generic import CSVGroundTruthReader
from stonesoup.feeder.geo import LongLatToUTMConverter
import utm

truthslonlat = CSVGroundTruthReader(
    "OpenSky_Plane_States.csv",
    state_vector_fields=("lon", "x_speed", "lat", "y_speed",
                         "geoaltitude", "vertrate"),  # List of columns names to be used in state vector
    path_id_field="icao24",                           # Name of column to be used as path ID
    time_field="time",                                # Name of column to be used as time field
    timestamp=True)                                   # Treat time field as a timestamp from epoch


groundtruth = LongLatToUTMConverter(truthslonlat, zone_number=30,  mapping=[0, 2])

# %%
# Constructing Sensors
# --------------------
# Now we will assemble our sensors used in this demonstration. We’ll introduce 2 stationary radar
# sensors, and also demonstrate Stone Soup’s ability to model moving sensors.
#
# We will use :class:`~.RadarElevationBearingRange` to establish our radar sensors.
# :class:`~.RadarElevationBearingRange` allows us to generate measurements of targets by using
# a :class:`~.CartesianToElevationBearingRange` model. We proceed to create a :class:`~.Platform`
# for each stationary sensor, and append these to our list of all platforms.
#
# Our moving sensor will be created similarly to the stationary case. We will need to
# make a movement controller to control the platform's movement, this is done by creating
# transition models, as well as setting transition times.
#
# :class:`~.PlatformDetectionSimulator` will then proceed to generate our detection data from
# the groundtruth (calls each sensor in platforms).

import datetime
import numpy as np

from stonesoup.types.array import StateVector
from stonesoup.sensor.radar import RadarElevationBearingRange
from stonesoup.types.state import State
from stonesoup.platform.base import FixedPlatform, MultiTransitionMovingPlatform
from stonesoup.simulator.platform import PlatformDetectionSimulator
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel,\
    ConstantVelocity, KnownTurnRate

# Create locations for reference later

# Using Heathrow as origin
*heathrow, utm_zone, _ = utm.from_latlon(51.47, -0.4543)
# Use heathrow utm grid num as reference number for utm conversions later
manchester = utm.from_latlon(53.35, -2.280, utm_zone)


# Create transition models for moving platforms
transition_modelStraight = CombinedLinearGaussianTransitionModel((ConstantVelocity(0.01),
                                                                  ConstantVelocity(0.01),
                                                                  ConstantVelocity(0.01)))

transition_modelLeft = CombinedLinearGaussianTransitionModel((KnownTurnRate((0.01, 0.01),
                                                              np.radians(3)), ConstantVelocity(0.01)))


# Create specific transition model for example moving platform
transition_models = [transition_modelStraight,
                     transition_modelLeft]
transition_times = [datetime.timedelta(seconds=160),
                    datetime.timedelta(seconds=20)]


# List sensors in stationary platforms
stationarySensors = [
    RadarElevationBearingRange(
        ndim_state=6,
        position_mapping=(0, 2, 4),
        noise_covar=np.diag([np.radians(1)**2, np.radians(1)**2, 7**2]),
        max_range=100000),

    RadarElevationBearingRange(
        ndim_state=6,
        position_mapping=(0, 2, 4),
        noise_covar=np.diag([np.radians(1)**2, np.radians(1)**2, 7**2]),
        max_range=100000)
    ]

# List sensors in moving platform
movingPlatformSensors = [
    RadarElevationBearingRange(
        ndim_state=6,
        position_mapping=(0, 2, 4),
        noise_covar=np.diag([np.radians(1.2)**2, np.radians(1.2)**2, 8**2]),
        max_range=60000)
    ]

platforms = []

# Create a platform for each stationary sensor and add to list of platforms
for sensor, platformLocation in zip(stationarySensors, (heathrow, manchester)):
    platformState = State([[platformLocation[0]], [0], [platformLocation[1]], [0], [0], [0]])
    platform = FixedPlatform(platformState, (0, 2, 4), sensors=[sensor])
    platforms.append(platform)

# Create moving platform
movingPlatformInitialLocation = utm.from_latlon(52.25, -0.9, utm_zone)
movingPlatformState = State([[movingPlatformInitialLocation[0]], [0],
                             [movingPlatformInitialLocation[1]], [250], [5000], [0]])
movingPlatforms = [MultiTransitionMovingPlatform(movingPlatformState,
                                                 position_mapping=(0, 2, 4),
                                                 transition_models=transition_models,
                                                 transition_times=transition_times,
                                                 sensors=movingPlatformSensors)]

# Add moving platform to list of platforms
platforms.extend(movingPlatforms)

# Simulate platform detections
detection_sim = PlatformDetectionSimulator(groundtruth, platforms)

# %%
# Individual Components
# ---------------------
# Now it's time to set up our individual components needed to construct our initiator, and
# ultimately our :class:`~.MultiTargetTracker`. We will be using the Extended Kalman Filter
# since our sensor model, :class:`~.CartesianToElevationBearingRange`, is not linear.
#
# To produce our linear transition model, we combine multiple one dimensional models into one
# singular model. Notice how we specify a different transition model for our initiator. We then
# pass these transition models to their respective predictors.
# The update step calculates the posterior state estimate by using both our prediction and
# sensor measurement. We don't need to define the measurement model here since the
# model :class:`~.CartesianToElevationBearingRange` is already provided in the measurements.
#
# The :class:`~.DistanceHypothesiser` generates track predictions at detection times, and
# scores each hypothesised prediction-detection pair using our set measure of :class:`~.Mahalanobis`
# distance. We allocate the detections to our predicted states by using the Global Nearest Neighbour
# method. The :class:`~.UpdateTimeDeleter` will identify the tracks for deletion and delete them
# once the time since last update has exceeded our specified time.
# By having `delete_last_pred = True`, the state that caused a track to be deleted will be deleted
# (if it is a prediction).

transition_model = CombinedLinearGaussianTransitionModel(
    (ConstantVelocity(20), ConstantVelocity(20), ConstantVelocity(2)))
init_transition_model = CombinedLinearGaussianTransitionModel(
    (ConstantVelocity(5), ConstantVelocity(5), ConstantVelocity(2)))

from stonesoup.predictor.kalman import ExtendedKalmanPredictor
predictor = ExtendedKalmanPredictor(transition_model)
init_predictor = ExtendedKalmanPredictor(init_transition_model)

from stonesoup.updater.kalman import ExtendedKalmanUpdater
updater = ExtendedKalmanUpdater(measurement_model=None)

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(), missed_distance=5)

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
data_associator = GNNWith2DAssignment(hypothesiser)

from stonesoup.deleter.time import UpdateTimeDeleter
deleter = UpdateTimeDeleter(datetime.timedelta(seconds=20), delete_last_pred=True)

# %%
# The Initiator
# -------------
# We can now create the initiator. The :class:`~.MultiMeasurementInitiator` will initiate
# and hold tracks until enough detections have been associated with the track. It will then
# proceed to release the tracks to the tracker.

from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.types.state import GaussianState

initiator = MultiMeasurementInitiator(
    prior_state=GaussianState(
        np.array([[0], [0], [0], [0], [0], [0]]),   # Prior State
        np.diag([15**2, 100**2, 15**2, 100**2, 15**2, 20**2])),
    measurement_model=None,
    deleter=deleter,
    data_associator=GNNWith2DAssignment(
        DistanceHypothesiser(init_predictor, updater, Mahalanobis(), missed_distance=3)),
    updater=updater,
    min_points=2
    )

# %%
# The Tracker
# -----------
# Next, we bring together all the components we’ve assembled to construct our :class:`~.MultiTargetTracker`.
# A loop is created to generate tracks at each time interval, and store them in a set called tracks.

from stonesoup.tracker.simple import MultiTargetTracker

kalman_tracker = MultiTargetTracker(
    initiator=initiator,
    deleter=deleter,
    detector=detection_sim,
    data_associator=data_associator,
    updater=updater,
)

tracks = set()
for step, (time, current_tracks) in enumerate(kalman_tracker, 1):
    tracks.update(current_tracks)

# %%
# It's easy for us to see how many tracks we've created by checking the length of the set tracks.

len(tracks)

# %%
# Plotting
# ------------------
# We will be using the Folium plotting library so we can visualize our tracks on a two-dimensional
# leaflet map. These Folium markers will show where our stationary sensors are located. Since our
# FOV angles are 360 degrees, we can easily use a fixed circle to display our radar's coverage.

import folium

m = folium.Map(
    location=[52.41, -0.4543], zoom_start=6)

folium.TileLayer('openstreetmap').add_to(m)


folium.Marker([51.47, -0.4543],
              tooltip="Heathrow Airport",
              icon=folium.Icon(icon='fa-circle', prefix="fa",      # Marker for Heathrow
              color="red")).add_to(m)

folium.Marker([53.35, -2.280],
              tooltip="Manchester Airport",
              icon=folium.Icon(icon='fa-circle', prefix="fa",      # Marker for Manchester
              color="green")).add_to(m)


folium.Circle(location=[51.47, -0.4543],
              popup='',
              fill_color='#000',                  # radar for Heathrow
              radius=100000,
              weight=2,
              color="#000").add_to(m)


folium.Circle(location=[53.35, -2.280],
              popup='',
              fill_color='#000',
              radius=100000,                     # radar for Manchester
              weight=2,
              color="#000").add_to(m)

# %%
# GeoJSON
# ~~~~~~~
# The Folium plugin `TimestampedGeoJson`_ will be used to plot our tracks using timestamped
# GeoJSONs. As a result, we want to convert our data into `GeoJSON format`_. Firstly, we create
# our feature collection which we will append our features to.
#
# .. _GeoJSON format: https://geojson.org/
# .. _TimestampedGeoJson:  https://python-visualization.github.io/folium/plugins.html

geo_features = list()
geo_json = {
    'type': "FeatureCollection",
    'features': geo_features,
}
# %%
# Each feature will have a properties object and a geometry object. Our properties object
# will contain information on the icon, popup, timestamps etc. The geometry object will be
# either a LineString, (for tracks and groundtruth path), or a MultiPoint (for icons of planes
# and moving sensor).

# %%
# Plotting the Moving Sensor
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Let us set up the icon and trail for our moving sensor. A radar dish icon [#images]_ will show its location
# at any given timestamp, and its trail will show where it has been.

trail_Size = 14  # trail_Size is number of timestamps we want track to trail for
for platform in movingPlatforms:
    points = []
    times_Sensor = []  # list of timestamps moving sensor has existed for.

    for state in platform.last_timestamp_generator():
        points.append( utm.to_latlon(state.state_vector[0], state.state_vector[2], utm_zone,
                                     northern=True, strict=False)[::-1])

        times_Sensor.append(state.timestamp.strftime('%Y-%m-%d %H:%M:%S'))

    for time_index, time in enumerate(times_Sensor):
        geo_features.append({  # attaching info about moving sensor to geo_json
            'type': "Feature",
            'properties':
                {'popup':  "Moving Sensor",
                'name':   '',
                'style':  {'color': 'black', 'weight': 4},
                'times':  [time]*trail_Size},

            'geometry':
                {'type': "LineString",
                'coordinates': points[:time_index+1][-trail_Size:]}
        })

        geo_features.append({  # attaching icon info about moving sensor to geo_json
            'type': "Feature",
            'properties':{
                'icon': 'marker',
                'iconstyle':{
                    'iconUrl': '../_static/sphinx_gallery/Radar_dish.png',
                    'iconSize': [24, 24],
                    'fillOpacity': 1,
                    'popupAnchor': [1, -17]},

                'popup': "Moving Sensor",
                'name':  '',
                'style': {'color': 'black', 'weight': 4},
                'times': [time]},

            'geometry':{
                'type': "MultiPoint",
                'coordinates': [points[time_index]]}
        })

# %%
# We also want to display our moving sensor’s FOV as time progresses.
# Since GeoJSON does not support circles, we will display the boundary of our moving sensor's
# FOV by drawing a LineString.

points_ = []  # setting up points for moving sensor (UTM)
radius = 60000  # 60 km, radius of our moving sensor
polygon_Sides = 60
for state in movingPlatforms:
    for time_index, time in enumerate(times_Sensor): # num_Timestamps = number of timestamps elapsed
        points_.append((state[time_index].state_vector[0], state[time_index].state_vector[2]))

    for (time_index,(x, y)) in enumerate(points_):  # finding points of circle
        time = times_Sensor[time_index]             # for range of moving sensor.
        angles = np.linspace(0, 2 * np.pi, polygon_Sides + 1, endpoint=True)

        points_list = [utm.to_latlon(x + np.sin(angle) * radius,
                       y + np.cos(angle) * radius, utm_zone, northern=True, strict=False)[::-1]
                       for angle in angles]

        geo_features.append({
            'type': "Feature",
            'properties':{
                'popup':  "Moving Sensor FOV",
                'name':    '',
                'style':   {'color': 'black', 'weight': 3},
                'times':   [time] * (polygon_Sides + 1)},

            'geometry':{
                'type':   "LineString",
                'coordinates': points_list}
        })

# %%
# Plotting Tracks
# ~~~~~~~~~~~~~~~
# Now we append our tracks to our feature collection list. We define `colour_iter` which will
# allow us to cycle through different colours when mapping our tracks.

from collections import defaultdict
from itertools import cycle

colour_iter = iter(cycle(
    ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
     '#0909FF','#F70D1A', '#FF6700', 'lightgreen', '#0AFFFF',
     '#12AD2B', '#E2F516', '#FFFF00', '#F52887']))
colour = defaultdict(lambda: next(colour_iter))

trail_Size = 14  # trail_Size is the number of timestamps we want track to trail for

for track in tracks:
    plot_times = [state.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                  for state in track.last_timestamp_generator()]

    plot_points = [
        utm.to_latlon(state.state_vector[0], state.state_vector[2], utm_zone, northern=True,
                      strict=False)[::-1] for state in track.last_timestamp_generator()]

    for time_index, time in enumerate(plot_times):
        geo_features.append({
            'type': "Feature",
            'properties':{
                'name':  track.id,
                'style': {'color': colour[track], 'weight': 6},
                'times': [time] * len(plot_points[:time_index+1][-trail_Size:])},

            'geometry':{
                'type': "LineString",
                'coordinates': plot_points[:time_index+1][-trail_Size:]}
        })

# %%
# Plotting Groundtruth
# ~~~~~~~~~~~~~~~~~~~~
# To plot our groundtruth, we firstly want to set up a dictionary that will allow us to easily
# access our groundtruth data. The dictionary will enable us to get data in the correct format
# needed for plotting, as well as for displaying key properties on popups.

all_truths = dict()
for time, truths in truthslonlat:
    for truth in truths:
        id_ = truth.metadata.get('icao24')
        if id_ not in all_truths:
            all_truths[id_] = dict()
            all_truths[id_]['times'] = list()
            all_truths[id_]['lonlats'] = list()
            all_truths[id_]['velocity'] = list()
            all_truths[id_]['heading'] = list()
        lon, lat = truth.state_vector[[0, 2]]
        all_truths[id_]['times'].append(time)
        all_truths[id_]['lonlats'].append((lon, lat))
        all_truths[id_]['velocity'].append(truth.metadata.get('velocity'))
        all_truths[id_]['heading'].append(truth.metadata.get('heading'))

# %%
# A list of all icao24 addresses in our data will also be created. These addresses are used to
# give an aircraft a unique identity. We will run through this list and use our groundtruth
# dictionary to get relevant data needed for plotting.
# The variable `trail_Size` will determine how much history of the track we want to be seen.
# We will plot a LineString to display our track, taking into account our specified `trail_Size`
# to determine the cut-off point.
#
# We begin with creating the trails of our groundtruth.

icao = list()
for time, truths in truthslonlat:
    for truth in truths:
        ids = truth.metadata.get('icao24')
        if ids not in icao:
            icao.append(ids)

for id in icao:
    trail_Size = 14
    for time_index, time in enumerate(all_truths[id]['times']):
        points=all_truths[id]['lonlats'][:time_index+1][-trail_Size:]
        geo_features.append({
            'type': "Feature",
            'properties':{
                'name': '',
                'style': {'color': 'black', 'weight': 2},
                'times': [time.strftime('%Y-%m-%d %H:%M:%S')]*len(points)},

            'geometry':{
                'type': "LineString",
                'coordinates': points}
                            })

# %%
# Our final task is to provide icons for our groundtruth. For each timestamp, each plane's
# heading will be rounded to the nearest 10 degrees, and an appropriate icon [#images]_ to reflect this
# heading will be chosen. Icons of planes can also be clicked on to display key data
# (remember to pause time to do this).

for id in icao:
    for time in range(len(all_truths[id]['times'])):
        if all_truths[id]['heading'][time] == '':  # if no heading given in data,
            break                                  # won't plot icon

        angle = round(float(all_truths[id]['heading'][time]), -1) # rounding angle to nearest 10 degrees
        if angle == 360:
            angle = 0
        angle = int(angle)

        geo_features.append({
            'type': "Feature",
            'properties':{
                'icon': 'marker',
                'iconstyle':{
                    'iconUrl': f'../_static/sphinx_gallery/Plane_Headings/Plane_{angle}.png',
                    'iconSize': [24, 24],
                    'fillOpacity': 1,
                    'popupAnchor': [1, -17],

                                         },

                'popup':
                     "ICAO24: " + id + "<dd>"
       
                     "Velocity: " + '%s' % float('%.5g' %
                     (float(all_truths[id]['velocity'][time]))) + " m/s" + "<dd>"
                                                                     
                     "Heading: " + '%s' % float('%.5g' %
                     (float(all_truths[id]["heading"][time]))) + "°" + "<dd>" 
                                                                            
                     "Longitude: " + '%s' % float('%.8g' %
                     (all_truths[id]["lonlats"][time][0])) + "<dd>" # rounding 8 sigfigs

                     "Latitude: " + '%s' % float('%.8g' %
                     (all_truths[id]["lonlats"][time][1])),

                'name': '',
                'style': {'color': 'black', 'weight': 2},
                'times': [all_truths[id]['times'][time].strftime('%Y-%m-%d %H:%M:%S')]},

            'geometry':{
                'type': "MultiPoint",
                'coordinates': [all_truths[id]['lonlats'][time]]}
        })

# %%
# The Results
# ~~~~~~~~~~~

from folium.plugins import TimestampedGeoJson, Fullscreen

Fullscreen().add_to(m)

TimestampedGeoJson(
    data=geo_json,
    transition_time=200,
    auto_play=True,
    add_last_point=False,
    period='PT10S',
    duration='PT0S').add_to(m)

# %%

# sphinx_gallery_thumbnail_path = '_static/sphinx_gallery/OpenSky_openstreet_thumb.png'

m

# %%
# References
# ----------
# .. [#] The OpenSky Network, http://www.opensky-network.org
# .. [#] Bringing up OpenSky: A large-scale ADS-B sensor network for research
#    Matthias Schäfer, Martin Strohmeier, Vincent Lenders, Ivan Martinovic, Matthias Wilhelm
#    ACM/IEEE International Conference on Information Processing in Sensor Networks, April 2014
# .. [#images] Radar and Plane icons provided by http://simpleicon.com/
