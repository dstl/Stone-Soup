#!/usr/bin/env python
# coding: utf-8

"""
Tracking ADS-B data from OpenSky
================================
"""
# %%
# Introduction
# ------------
# Our goal in this notebook is to plot time series data of Stone Soup's :class:`~.MultiTargetTracker`
# being applied to air traffic over and surrounding the UK.
# To do this, we will be using a CSV file of ADS–B data sourced from `The OpenSky Network`_.
# This data will be used as our groundtruth. We will establish the individual components
# required for our tracker, including generating detection data from our groundtruth, and then
# plot these tracks using the Folium plugin `TimestampedGeoJson`_.
#
# .. _The OpenSky Network: https://www.opensky-network.org
# .. _TimestampedGeoJson:  https://python-visualization.github.io/folium/plugins.html

# %%
# Reading the CSV File
# ---------------------
# To read in our groundtruth data from a CSV file, we can use Stone Soup’s
# :class:`~.CSVGroundTruthReader`. We want to convert our longitude / latitude into
# UTM projection, we can do this by using :class:`~.LongLatToUTMConverter`.

from stonesoup.reader.generic import CSVGroundTruthReader
from stonesoup.feeder.geo import LongLatToUTMConverter
import utm

truthslonlat = CSVGroundTruthReader(
    "OpenSky_Plane_States.csv",
    state_vector_fields = ("lon", "x_speed","lat","y_speed"),     # List of columns names to be used in state vector
    path_id_field = "icao24",                                     # Name of column to be used as path ID
    time_field = "time",                                          # Name of column to be used as time field
    timestamp = True)                                             # Treat time field as a timestamp from epoch


groundtruth = LongLatToUTMConverter(truthslonlat, zone_number = 30,  mapping = [0,2])

# %%
# Constructing Sensors
# --------------------
# Now we are going to assemble our sensors used in this demonstration. We’ll introduce 3 stationary radar
# sensors, and also demonstrate Stone Soup’s ability to model moving sensors.
#
# We use :class:`~.RadarRotatingBearingRange` to establish our radar sensors. We set their FOV angle,
# range and rpm. Because our timestamps in our data are in intervals of 10 seconds, it is
# reasonable assumption to have our FOV angle at 360 degrees.
#
# :class:`~.RadarRotatingBearingRange` allows us to generate measurements of targets by using
# a :class:`~.CartesianToBearingRange` model. We proceed to create a :class:`~.platform` for
# each stationary sensor, and append these to our list of all platforms
#
# Our moving sensor will be created the same way as in the stationary case, setting FOV angle,
# range, rpm etc. We will also need to make a movement controller to
# control the platform's movement, this is done by creating transition models, as well as
# setting transition times.
#
# :class:`~.PlatformDetectionSimulator` will then proceed to generate our detection data from
# the groundtruth, (calls each sensors in platforms).

import datetime
import numpy as np

from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.sensor.radar import RadarRotatingBearingRange
from stonesoup.types.state import State
from stonesoup.platform.base import Platform, FixedPlatform, MultiTransitionMovingPlatform
from stonesoup.simulator.platform import PlatformDetectionSimulator
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity, ConstantTurn, RandomWalk

# Create locations for reference later

# Using Heathrow as origin
*heathrow, utm_zone, _ = utm.from_latlon(51.47, -0.4543)
# Use heathrow utm gird num as reference number for utm conversions later
glasgow = utm.from_latlon(55.87, -4.433, utm_zone)
manchester = utm.from_latlon(53.35, -2.280, utm_zone)


# Create transition models for moving platforms
transition_modelStraight = CombinedLinearGaussianTransitionModel((ConstantVelocity(0.01),
                                                                  ConstantVelocity(0.01),
                                                                  ConstantVelocity(0.01))
                                                                )

transition_modelLeft = CombinedLinearGaussianTransitionModel((ConstantTurn((0.01,0.01),np.radians(3))
                                                              ,ConstantVelocity(0.01))
                                                            )
transition_modelRight = CombinedLinearGaussianTransitionModel((ConstantTurn((0.01,0.01),np.radians(-3)),
                                                               ConstantVelocity(0.01))
                                                             )

# Create specific transition model for example moving platform
transition_models = [transition_modelStraight,
                     transition_modelLeft]
transition_times = [datetime.timedelta(seconds=160),
                    datetime.timedelta(seconds=20)]


# List sensors in stationary platforms (sensor orientations are overwritten)
stationarySensors = [
    RadarRotatingBearingRange(
        ndim_state = 6,
        position_mapping = (0,2),
        noise_covar = np.diag([np.radians(1)**2, 8**2]),   # The sensor noise covariance matrix
        dwell_center = State(StateVector([[-np.pi]])),
        rpm = 12.5,
        max_range = 100000,
        fov_angle = np.radians(360)),
        # position=np.array([[heathrow[0]],[heathrow[1]],[0]]),
        # orientation=np.array([[0],[0],[0]]))

    RadarRotatingBearingRange(
        ndim_state = 6,
        position_mapping = (0,2),
        noise_covar = np.diag([np.radians(1)**2, 8**2]),
        dwell_center = State(StateVector([[-np.pi]])),
        rpm = 12.5,
        max_range = 100000,
        fov_angle = np.radians(360)),
        # position=StateVector([[glasgow[0]],[glasgow[1]],[0]]),
        # orientation=np.array([[0],[0],[0]])),

    RadarRotatingBearingRange(
        ndim_state = 6,
        position_mapping = (0,2),
        noise_covar = np.diag([np.radians(1)**2, 8**2]),
        dwell_center = State(StateVector([[-np.pi]])),
        rpm = 12.5,
        max_range = 100000,
        fov_angle = np.radians(360)),
        # position=np.array([[manchester[0]],[manchester[1]],[0]]),
        # orientation=np.array([[0],[0],[0]]))
    ]

# List sensors in moving platform (sensor orientations are overwritten)
movingPlatformSensors = [
    RadarRotatingBearingRange(
        ndim_state = 6,
        position_mapping = (0,2),
        noise_covar = np.diag([np.radians(1)**2, 12**2]),
        dwell_center = State(StateVector([0])),
        rpm = 20,
        max_range = 60000,
        fov_angle = np.radians(360)),
    ]

platforms = []

# Create a platform for each stationary sensor and add to list of platforms
for sensor, platformLocation in zip(stationarySensors, (heathrow, glasgow, manchester)):
    platformState = State([[platformLocation[0]], [0], [platformLocation[1]], [0], [0], [0]])
    platform = FixedPlatform(platformState, (0, 2, 4), sensors=[sensor])
    platforms.append(platform)


# Create moving platform
movingPlatformIntialLocation = utm.from_latlon(51.40, -3.436, utm_zone)
movingPlatformState = State([[movingPlatformIntialLocation[0]], [0], [movingPlatformIntialLocation[1]], [250], [0], [0]])
movingPlatforms = [MultiTransitionMovingPlatform(movingPlatformState,
                                                 position_mapping=(0, 2),
                                                 transition_models=transition_models,
                                                 transition_times=transition_times,
                                                 sensors=movingPlatformSensors)]
# Add moving platform to list of platforms
platforms.extend(movingPlatforms)


# Simulate platform detections
detection_sim = PlatformDetectionSimulator(groundtruth, platforms)

# Keep track of number of sensors for plotting later
numSensors = len(movingPlatformSensors) + len(stationarySensors)

# %%
# Individual Components
# ---------------------
# Now it's time to set up our individual components needed to construct our initiator, and
# ultimately our :class:`~.MultiTargetTracker`. We will be using the Extended Kalman Filter
# since our sensor model, :class:`~.CartesianToBearingRange`, is not linear.
#
# To produce our linear transition model, we combine multiple one dimensional models into one
# singular model. Notice how we specify a different transition model for our initiator. We then
# pass these transition models to their respective predictors.
# The update step calculates the posterior state estimate by using both our prediction and
# sensor measurement. We don't need to define the measurement model here since the
# model :class:`~.CartesianToBearingRange` is already provided in the measurements.
#
# The :class:`~.DistanceHypothesiser` generates track predictions at detection times, and
# scores each hypothesised prediction-detection pair using our set measure of :class:`~.Mahalanobis` distance.
# We will allocate the detections to our predicted states by using the Global Nearest Neighbour method.
# The :class:`~.UpdateTimeDeleter` will identify the tracks for deletion - the tracks will be deleted once
# the time since last update has exceeded our specified time.
# By having delete_last_pred = True, the state that caused a track to be deleted will be deleted,
# (if it is a prediction).

transition_model = CombinedLinearGaussianTransitionModel(
    (ConstantVelocity(1500), ConstantVelocity(1200), ConstantVelocity(1000)))
init_transition_model = CombinedLinearGaussianTransitionModel(
    (ConstantVelocity(8500), ConstantVelocity(7500), ConstantVelocity(1000)))

from stonesoup.predictor.kalman import ExtendedKalmanPredictor
predictor = ExtendedKalmanPredictor(transition_model)
init_predictor = ExtendedKalmanPredictor(init_transition_model)

from stonesoup.updater.kalman import ExtendedKalmanUpdater
updater = ExtendedKalmanUpdater(measurement_model = None)

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(), missed_distance = 3)

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
data_associator = GNNWith2DAssignment(hypothesiser)

from stonesoup.deleter.time import UpdateTimeDeleter
deleter = UpdateTimeDeleter(datetime.timedelta(seconds = 20), delete_last_pred = True )

# %%
# The Initiator
# -------------
# We can now create the initiator. The :class:`~.MultiMeasurementInitiator` will initiate,
# and hold tracks until enough detections have been associated with the track, it will then
# proceed to release the tracks to the tracker.

from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.types.state import GaussianState

initiator = MultiMeasurementInitiator(
    GaussianState(
        np.array([[0], [0], [0], [0], [0], [0]]),   #Prior State
        np.diag([8000, 90, 8000, 90, 0, 0])),
    measurement_model = None,
    deleter = deleter,
    data_associator = GNNWith2DAssignment(
        DistanceHypothesiser(init_predictor, updater, Mahalanobis(), missed_distance = 1.8)),
    updater = updater,
    min_points = 2
    )

# %%
# The Tracker
# -----------
# Next, we bring together all the components we’ve assembled to construct our :class:`~.MultiTargetTracker`.
# A loop is created to generate tracks at each time interval, and store them in a set labelled tracks.

from stonesoup.tracker.simple import MultiTargetTracker

kalman_tracker = MultiTargetTracker(
    initiator = initiator,
    deleter = deleter,
    detector = detection_sim,
    data_associator = data_associator,
    updater = updater,
)

tracks = set()
for step, (time, current_tracks) in enumerate(kalman_tracker.tracks_gen(), 1):
    tracks.update(current_tracks)

# %%
# It's easy for us to see how many tracks we've created by checking the length of the set tracks.

len(tracks)

# %%
# Plotting
# ------------------
# We will be using the Folium plotting library so we can visualize our tracks on a leaflet map.
# These Folium markers and circles are set up to show where are stationary sensors are located.
# Since our FOV angles are 360 degrees, we can easily use a fixed circle to display our radar's
# coverage.

import folium

m = folium.Map(
    location=[51.47,-0.4543], zoom_start = 5,
    tiles='http://{s}.tiles.wmflabs.org/bw-mapnik/{z}/{x}/{y}.png',
    attr='&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>')


folium.Marker([51.47,-0.4543],
              tooltip = "Heathrow Airport",
              icon = folium.Icon(icon='fa-circle', prefix="fa",      # Marker for Heathrow
              color = "red")).add_to(m)


folium.Marker([55.87, -4.433],
              tooltip = "Glasgow Airport",
              icon = folium.Icon(icon='fa-circle', prefix="fa",      # Marker for Glasgow
              color = "blue")).add_to(m)


folium.Marker([53.35, -2.280],
              tooltip = "Manchester Airport",
              icon = folium.Icon(icon='fa-circle', prefix="fa",      #Marker for Manchester
              color = "green")).add_to(m)


folium.Circle(location=[51.47, -0.4543],
              popup='',
              fill_color='#000',                  #radar for Heathrow
              radius=100000,
              weight=2,
              color="#000").add_to(m)


folium.Circle(location=[55.87, -4.433],
              popup='',
              fill_color='#000',
              radius=100000,                      #radar for Glasgow
              weight=2,
              color="#000").add_to(m)


folium.Circle(location=[53.35, -2.280],
              popup='',
              fill_color='#000',
              radius=100000,                     #radar for Manchester
              weight=2,
              color="#000").add_to(m)

# %%
# GeoJSON
# ~~~~~~~
# The Folium plugin `TimestampedGeoJson`_ will be used to plot our tracks using timestamped
# GeoJSONs. As a result, we want to convert our data into `GeoJSON format`_. Firstly, we create
# our feature collection, which we will append our features to.
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
# either a LineString, (for tracks and groundtruth path), or a Multipoint, (for icons of planes
# and moving sensor).

# %%
# Plotting the Moving Sensor
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Let us set up the icon and trail for our moving sensor. A radar dish icon will show its location
# at any given timestamp, and its trail will show where it has been.

points = []
timesSensor = []  # list of time stamps moving sensor has existed for, (will be all timestamps elapsed)
noTimestamps = len([1 for timestamp in movingPlatforms[0]])  # number of total timestamps elapsed

for state in movingPlatforms:
    for i in range(noTimestamps):
        points.append(
            utm.to_latlon(state[i].state_vector[0], state[i].state_vector[2], utm_zone,
                          northern=True, strict=False))
        timesSensor.append(state[i].timestamp.strftime('%Y-%m-%d %H:%M:%S'))

points = [(t[1], t[0]) for t in points] # we want (lon,lat), hence we switch from (lat,lon)

trailSize = 14  # trailSize is number of timestamps we want track to trail for
timestampNumber = 0

for time in timesSensor:
    timestampNumber = timestampNumber + 1
    trailEnd = timestampNumber - trailSize  # trailend determines timestamp for which the trail ends
    if trailEnd < 0: trailEnd = 0

    geo_features.append({  # attaching info about moving sensor to geo_json
        'type': "Feature",
        'properties': {
            'popup':  "Moving Sensor",
            'name':   '',
            'style':  {'color': 'black', 'weight': 4},
            'times':  [time for i in range(trailEnd, timestampNumber)]},

        'geometry':   {'type': "LineString",
                      'coordinates': [points[j] for j in range(trailEnd, timestampNumber)]}
    })

    geo_features.append({  # attaching icon info about moving sensor to geo_json
        'type': "Feature",
        'properties': {
                       'icon': 'marker',
                       'iconstyle': {
                           'iconUrl': 'http://simpleicon.com/wp-content/uploads/radar.png',
                           'iconSize': [24, 24],
                           'fillOpacity': 1,
                           'popupAnchor': [1, -17],
                           },

                       'popup': "Moving Sensor",
                       'name': '',
                       'style': {'color': 'black', 'weight': 4},
                       'times': [time]},

        'geometry':    {
                       'type': "MultiPoint",
                       'coordinates': [points[timestampNumber - 1]]}
    })

# %%
# We also want to display our moving sensor’s FOV as time progresses.
# Since GeoJSON does not support circles, we will display the boundary of our moving sensor's
# FOV by drawing a LineString.

points_ = []  # setting up points for moving sensor (UTM)
for state in movingPlatforms:
    for timestampNumber in range(noTimestamps):  # noTimestamps = number of timestamps elapsed in sim
        points_.append(
            (state[timestampNumber].state_vector[0], state[timestampNumber].state_vector[2]))

radius = 60000  # 60 km, radius of our moving sensor
polygonSides = 60
timestampNumber = 0
for ((x,y)) in points_:  # finding points of circle for range of moving sensor, adding them to geo_json
    time = timesSensor[timestampNumber]
    timestampNumber = timestampNumber + 1
    angles = np.linspace(0, 2 * np.pi, polygonSides + 1, endpoint=True)
    points_list = [(x + np.sin(angle) * radius,
                    y + np.cos(angle) * radius)
                   for angle in angles]

    for i in range(len(points_list)):
        points_list[i] = utm.to_latlon(points_list[i][0], points_list[i][1], utm_zone,
                                       northern=True, strict=False)

    points_list = [(t[1], t[0]) for t in points_list]  # we want (lon,lat), hence we switch from (lat,lon).

    geo_features.append({
        'type': "Feature",
        'properties':  {
                       'popup':   "Moving Sensor FOV",
                       'name':    '',
                       'style':   {'color': 'black', 'weight': 3},
                       'times':   [time for i in range(polygonSides + 1)]},

        'geometry':    {
                       'type': "LineString",
                       'coordinates': points_list}
    })

# %%
# Plotting Tracks
# ~~~~~~~~~~~~~~~
# Now, we append our tracks to our feature collection list. We define colour_iter, which will
# allow us to cycle through different colours when mapping our tracks.

from collections import defaultdict
from itertools import cycle

colour_iter = iter(cycle(
    ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
     'lightred', '#0909FF', 'darkblue', 'darkgreen', 'cadetblue',
     'darkpurple', 'pink', '#FF6700', 'lightgreen', '#0AFFFF',
     '#12AD2B', '#E2F516', '#FFFF00', '#F52887']))
colour = defaultdict(lambda: next(colour_iter))

col = 0  # col will be used to cycle through the colours of the tracks
trailSize = 14  # trailSize is the number of timestamps we want track to trail for

for track in tracks:

    plot_time = [state.timestamp.strftime('%Y-%m-%d %H:%M:%S') for state in track]

    plot_times = [plot_time[0]]

    for i in range(1, int((len(plot_time))) - 1):  # make sure plot_times only contains one copy of each timestamp for which
        if plot_time[i] == plot_times[-1]:  # the track has been alive. will have multiple otherwise since multiple sensors
            continue
        else:
            plot_times.append(plot_time[i])

    plot_points = [
        utm.to_latlon(state.state_vector[0], state.state_vector[2], utm_zone, northern=True,
                      strict=False)
        for state in track.last_timestamp_generator()]

    plot_points = [(t[1], t[0]) for t in plot_points]  # need it to be(lon,lat)

    col = col + 1

    timestampNumber = 0  # counts how many timestamps have elapsed in track's history
    for times_ in plot_times:
        timestampNumber = timestampNumber + 1
        trailEnd = timestampNumber - trailSize  # trailend determines track's timestamp for end of trail
        if trailEnd < 0: trailEnd = 0

        geo_features.append({
            'type': "Feature",
            'properties': {
                           'name': track.id,
                           'style': {'color': colour[col], 'weight': 6},
                           'times': [times_ for i in range(trailEnd, timestampNumber)]},  # want times to be as lon as coordinates

            'geometry':    {'type': "LineString",
                           'coordinates': [plot_points[j] for j in range(trailEnd,timestampNumber)]}
        })

# %%
# Plotting Groundtruth
# ~~~~~~~~~~~~~~~~~~~~
# To plot our groundtruth, we firstly want to set up a dictionary that will allow us to easily access
# our groundtruth data. The dictionary wil; enable us to get data in the correct format needed
# for plotting, as well as for displaying key properties on popups.
#
# A list of all icao24 addresses in our data will also be created. We will run
# through this list and use our groundtruth dictionary to get relevant data needed for plotting.
# The variable trailSize will determine how much history of the track we want to be seen.
# We will plot a LineString to display our track, taking into account our specified trailSize
# to determine the cut off point.
#
# We begin with creating the trails of our groundtruth.

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

icao = list()
for time, truths in truthslonlat:
    for truth in truths:
        ids = truth.metadata.get('icao24')
        if ids not in icao:
            icao.append(ids)

for id in icao:
    trailSize = 14  #
    timestampNumber = 0  #
    for times_ in all_truths[id]['times']:
        timestampNumber = timestampNumber + 1
        trailEnd = timestampNumber - trailSize  # k indexing element
        if trailEnd < 0: trailEnd = 0

        geo_features.append({
            'type': "Feature",
            'properties': {
                           'name': '',
                           'style': {'color': 'black', 'weight': 2},
                           'times': [times_.strftime('%Y-%m-%d %H:%M:%S') for i in range(trailEnd, timestampNumber)]},

            'geometry':    {'type': "LineString",
                           'coordinates': [all_truths[id]['lonlats'][j] for j in range(trailEnd, timestampNumber)]}
        })

# %%
# Our final task is to provide icons for our groundtruth. Let's define a set of links for
# png images of planes at various headings. These will be our icons.

planelinks=['https://i.ibb.co/9bmCCG8/Plane-0.png', #ordered from 0 degrees to 350, from north clockwise, interval of 10 degrees
            'https://i.ibb.co/LYTt0g6/Plane-10.png',
            'https://i.ibb.co/6ynvNJx/Plane-20.png',
            'https://i.ibb.co/QD3jL1P/Plane-30.png',
            'https://i.ibb.co/yQyNb00/Plane-40.png',
            'https://i.ibb.co/ZVvMRNN/Plane-50.png',
            'https://i.ibb.co/273Ghnq/Plane-60.png',
            'https://i.ibb.co/M21FdZ9/Plane-70.png',
            'https://i.ibb.co/FhJVw7f/Plane-80.png',
            'https://i.ibb.co/s930ZSx/Plane-90.png',
            'https://i.ibb.co/kmT7zBW/Plane-100.png',
            'https://i.ibb.co/bWRdPbQ/Plane-110.png',
            'https://i.ibb.co/RTYyDK5/Plane-120.png',
            'https://i.ibb.co/W6tBr1Z/Plane-130.png',
            'https://i.ibb.co/Mc28Hz7/Plane-140.png',
            'https://i.ibb.co/Cw3Rntx/Plane-150.png',
            'https://i.ibb.co/sgVDVyt/Plane-160.png',
            'https://i.ibb.co/NTnYPkC/Plane-170.png',
            'https://i.ibb.co/PMrwrnk/Plane-180.png',
            'https://i.ibb.co/gVmW86q/Plane-190.png',
            'https://i.ibb.co/jHkv2wr/Plane-200.png',
            'https://i.ibb.co/BBvBj0K/Plane-210.png',
            'https://i.ibb.co/QJHvS51/Plane-220.png',
            'https://i.ibb.co/KsT9ZTs/Plane-230.png',
            'https://i.ibb.co/nR8fBNS/Plane-240.png',
            'https://i.ibb.co/xLjf1DN/Plane-250.png',
            'https://i.ibb.co/qpGwpKq/Plane-260.png',
            'https://i.ibb.co/N1PPyfq/Plane-270.png',
            'https://i.ibb.co/HhrdLNg/Plane-280.png',
            'https://i.ibb.co/2hgYBsr/Plane-290.png',
            'https://i.ibb.co/GQ3ykRv/Plane-300.png',
            'https://i.ibb.co/D511XSW/Plane-310.png',
            'https://i.ibb.co/0sYLFP5/Plane-320.png',
            'https://i.ibb.co/ctdD2bG/Plane-330.png',
            'https://i.ibb.co/G2MZwQv/Plane-340.png',
            'https://i.ibb.co/ts0KjFj/Plane-350.png']

# %%
# A function named rounding will round a number to the nearest 10. This will be useful when
# deciding which icon to use for a given heading.

import math

def rounding(x, n = 10):
    res = math.ceil(x/n)*n
    if (x%n < n/2)and (x%n>0):
        res -= n
    return res

# %%
# There are 36 plane images at various headings. For each timestamp, each plane's heading
# will be rounded to the nearest 10 degrees, and an appropriate icon to reflect this heading
# will be chosen. Icons of planes can also be clicked on, (make sure to pause time), to display
# key data.

for id in icao:
    for timestampNumber in range((len(all_truths[id]['times']))):
        if all_truths[id]['heading'][
            timestampNumber] == '':  # might need this. if no heading given in data, won't plot icon.
            break

        angle = rounding(float(all_truths[id]['heading'][
                                   timestampNumber]))  # rounding angle to nearest 10 degrees
        angle = angle / 10  # diving angle to correspond with index of our plane icon urls
        if angle == 36: angle = 0
        angle = int(angle)

        geo_features.append({
            'type': "Feature",
            'properties': {'icon': 'marker',
                           'iconstyle': {'iconUrl': planelinks[angle],
                                         'iconSize': [24, 24],
                                         'fillOpacity': 1,
                                         'popupAnchor': [1, -17],

                                         },

                           'popup':
                                         "ICAO24: " + id + "<dd>"
                                         "Velocity: " + '%s' % float('%.5g' % (float(all_truths[id]['velocity'][timestampNumber]))) + " m/s" + "<dd>"
                                         "Heading: " + '%s' % float('%.5g' % (float(all_truths[id]["heading"][timestampNumber]))) + "°" + "<dd>"  # rounding 5 sigfigs
                                         "Longitude: " + '%s' % float('%.8g' % (all_truths[id]["lonlats"][timestampNumber][0])) + "<dd>"  # rounding 8 sigfigs
                                         "Latitude: " + '%s' % float('%.8g' % (all_truths[id]["lonlats"][timestampNumber][1])),

                           'name':       '',
                           'style':      {'color': 'black', 'weight': 2},
                           'times':      [all_truths[id]['times'][timestampNumber].strftime('%Y-%m-%d %H:%M:%S')]},
            'geometry':   {'type':       "MultiPoint",
                           'coordinates':  [all_truths[id]['lonlats'][timestampNumber]]}
        })

# %%
# The Results
# ~~~~~~~~~~~

from folium.plugins import TimestampedGeoJson, Fullscreen

Fullscreen().add_to(m)

(TimestampedGeoJson(
    data = geo_json,
    transition_time = 200,
    auto_play = True,
    add_last_point = False,
    period = 'PT10S',
    duration = 'PT0S')).add_to(m)

# %%

m

# %%
