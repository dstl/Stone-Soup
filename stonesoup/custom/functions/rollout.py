import copy
import itertools
import math
from datetime import timedelta, datetime
from enum import Enum
from typing import Union
from uuid import uuid4, UUID

import numpy as np
from pydantic import BaseModel
from scipy.stats import poisson
from shapely import Point, Polygon

from reactive_isr_core.data import TargetType, Availability, ActionStatus, Image, GeoLocation, \
    Algorithm

from stonesoup.custom.sensor.movable import MovableUAVCamera
from stonesoup.models.clutter import ClutterModel
from stonesoup.types.array import StateVector
from stonesoup.types.numeric import Probability
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track


class ActionTupleType(Enum):
    """Enum for the different types of action tuples"""
    NO_ACTION = 0
    ONBOARD = 1
    REMOTE = 2
    COMMS_AND_PROC = 3
    PROC_ONLY = 4

class ActionTuple(tuple):
    def __new__(self, tup=None):
        if not tup:
            coll_action, comms_action, proc_action = None, None, None
        else:
            coll_action, comms_action, proc_action = tup

        return tuple.__new__(ActionTuple, (coll_action, comms_action, proc_action))

    def __init__(self, *args, **kwargs):
        if self.coll_action is None and self.comms_action is None and self.proc_action is None:
            self._action_tuple_type = ActionTupleType.NO_ACTION
        elif self.coll_action is None:
            if self.comms_action is None:
                self._action_tuple_type = ActionTupleType.PROC_ONLY
            else:
                self._action_tuple_type = ActionTupleType.COMMS_AND_PROC
        else:
            if self.coll_action.node_id == self.proc_action.node_id:
                self._action_tuple_type = ActionTupleType.ONBOARD
            else:
                self._action_tuple_type = ActionTupleType.REMOTE

    def __bool__(self):
        return any(action is not None for action in self)

    def __str__(self):
        if self._action_tuple_type == ActionTupleType.NO_ACTION:
            return "No action"
        elif self._action_tuple_type == ActionTupleType.PROC_ONLY:
            return (f"PROC_ONLY(node={self.proc_action.node_id}, "
                    f"image={self.proc_action.image.id}, "
                    f"algorithm={self.proc_action.algorithm.name})")
        elif self._action_tuple_type == ActionTupleType.COMMS_AND_PROC:
            return (f"COMMS_PROC(from={self.comms_action.source_node_id}, "
                    f"to={self.comms_action.target_node_id}, "
                    f"image={self.proc_action.image.id}, "
                    f"algorithm={self.proc_action.algorithm.name})")
        elif self._action_tuple_type == ActionTupleType.ONBOARD:
            return (f"ONBOARD(image={self.coll_action.image.id}, "
                    f"algorithm={self.proc_action.algorithm.name})")
        else:
            return (f"REMOTE(from={self.coll_action.node_id}, "
                    f"to={self.proc_action.node_id}, "
                    f"image={self.coll_action.image.id}, "
                    f"algorithm={self.proc_action.algorithm.name})")

    def __repr__(self):
        return str(self)

    @property
    def coll_action(self):
        return self[0]

    @property
    def comms_action(self):
        return self[1]

    @property
    def proc_action(self):
        return self[2]

    @property
    def is_onboard(self):
        """Check if the processing is performed onboard"""
        return self._action_tuple_type == ActionTupleType.ONBOARD

    @property
    def type(self):
        return self._action_tuple_type


class CollectionAction(BaseModel):
    id: UUID
    node_id: UUID
    location: GeoLocation
    status: ActionStatus
    image: Image


class CommsAction(BaseModel):
    id: UUID
    source_node_id: UUID
    target_node_id: UUID
    image: Image
    status: ActionStatus
    start_time: Union[None, datetime]
    dt: timedelta

    @property
    def end_time(self):
        return self.start_time + self.dt


class ProcAction(BaseModel):
    id: UUID
    node_id: UUID
    image: Image
    algorithm: Algorithm
    status: ActionStatus
    start_time: Union[None, datetime]
    dt: timedelta

    @property
    def end_time(self):
        return self.start_time + self.dt


def cover_rectangle_with_minimum_overlapping_circles(x1, y1, x2, y2, radius):
    """
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1343643

    """
    width = x2 - x1
    height = y2 - y1

    p = Point(x1 + width/2, y1 + height/2).buffer(radius)
    pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    intersection = p.intersection(pol)
    if intersection.area >= 0.9*pol.area:
        return [(x1 + width/2, y1 + height/2)]

    # if width <= np.sqrt(3)/2*radius and height <= np.sqrt(3)/2*radius:
    z1 = height / (np.sqrt(3) * radius)
    re1 = z1 - math.floor(z1)
    n = math.floor(z1)
    if re1 <= 1/2:
        n += 1
    else:
        n += 2

    z2 = width / (3/2 * radius)
    re2 = z2 - math.floor(z2)
    m = math.floor(z2)
    if re2 <= 2/3:
        m += 1
    else:
        m += 2

    centers = []

    for k in range(1, n+1):
        for l in range(1, m+1):
            if l % 2 == 1:
                center = ((0.5 + (l-1) * 3/2) * radius, (k-1)*np.sqrt(3)*radius)
            else:
                center = ((0.5 + (l-1) * 3/2) * radius, (k-1)*np.sqrt(3)*radius + np.sqrt(3)/2*radius)
            offset_center = (center[0] + x1, center[1] + y1)
            cp = Point(offset_center)
            if cp.distance(pol) <= np.sqrt(3)/2*radius:
                centers.append(offset_center)
    return centers


def sample_dt(stats):
    dt = np.random.normal(stats.mu, stats.sigma)
    if dt < stats.lower_truncation:
        dt = stats.lower_truncation
    return timedelta(seconds=dt)


def extract_rois(rfis):
    """Extract all regions of interest from a list of rfis."""
    rois = []
    for rfi in rfis:
        for roi in rfi.region_of_interest:
            rois.append(roi)
    return rois


def get_processing_nodes(network_topology):
    return [node for node in network_topology.nodes
            if len(node.processing_capability.algorithms) > 0]


def get_edge(network_topology, source_node_id, target_node_id):
    return next(edge for edge in network_topology.edges
                if edge.source_node == source_node_id
                and edge.target_node == target_node_id)


def get_earliest_proc_start_time(ongoing_actions_per_node, node_id, default_time):
    """Get the earliest start time for processing actions for a given node.

    If there are no ongoing processing actions for the node, then the earliest start time is the
    default time. Otherwise, the earliest start time is the end time of the latest ongoing
    processing action.
    """
    if len(ongoing_actions_per_node[node_id]):
        return ongoing_actions_per_node[node_id][-1].end_time
    else:
        return default_time


def exists_ongoing_comms_proc_action_for_image(ongoing_actions, image_id):
    """Check if there is an ongoing comms or processing action for a given image."""
    return (any(action for action in ongoing_actions['comms'] if action.image.id == image_id)
            or any(action for action in ongoing_actions['proc'] if action.image.id == image_id))


def enumerate_actions_for_processing_node(node, image_store, network, ongoing_actions,
                                          earliest_proc_start_time, timestamp):
    """Enumerate all possible actions for a processing node that is not a sensor.

    If the node is not an asset (sensor), then we don't need to consider collection actions, but
    we do need to consider comms and processing actions for existing images

    Arguments
    ---------
    node : Node
        The processing node
    image_store : ImageStore
        The image store, containing all images that have been collected
    network : NetworkTopology
        The network topology, containing all nodes and edges
    earliest_proc_start_time : datetime
        The earliest start time for processing actions
    timestamp : datetime
        The current timestamp

    Returns
    -------
    possible_actions : List[Tuple[CollectionAction, CommsAction, ProcAction]]
        A list of possible actions for the node
    """
    possible_actions = [ActionTuple()]  # No action
    node_id = node.id
    for image in image_store.images:

        # If there is an ongoing comms or processing action for this image, then we don't need to
        # consider processing actions for it
        if exists_ongoing_comms_proc_action_for_image(ongoing_actions, image.id):
            continue

        # If the image is not on this node, then we need to consider a comms action with
        # the appropriate traversal time
        if image.node_id != node_id:
            edge = get_edge(network, image.node_id, node_id)
            dt = sample_dt(edge.traversal_time)
        else:
            dt = timedelta(seconds=0)
        comms_action = CommsAction(
            id=uuid4(),
            source_node_id=image.node_id,
            target_node_id=node_id,
            status=ActionStatus.CREATED,
            start_time=timestamp,
            image=image,
            dt=dt
        )
        # The earliest start time for processing actions is the end time of the comms
        # action or the earliest start time of the processing node
        earliest_proc_start_time_tmp = max(comms_action.end_time, earliest_proc_start_time)
        # Create a processing action for each algorithm
        for algorithm in node.processing_capability.algorithms:
            proc_action = ProcAction(
                id=uuid4(),
                node_id=node.id,
                algorithm=algorithm,
                status=ActionStatus.CREATED,
                start_time=earliest_proc_start_time_tmp,
                image=image,
                dt=sample_dt(algorithm.processing_statistics)
            )
            possible_actions.append(
                ActionTuple((None, comms_action, proc_action))
            )
    return possible_actions


def enumerate_actions_for_asset(node, fov_radius, network, rois, ongoing_proc_actions_per_node,
                                earliest_proc_start_time, timestamp):
    """Enumerate all possible actions for a processing node that is an asset (sensor).

    If the node is an asset (sensor), then we need to consider collection actions, comms actions
    and processing actions.

    NOTE: We assume that a sensor will not consider processing existing images.

    Arguments
    ---------
    node : Node
        The processing node
    fov_radius : float
        The field of view radius of the sensor (in km)
    network : NetworkTopology
        The network topology, containing all nodes and edges
    rois : List[GeoRegion]
        A list of regions of interest
    ongoing_proc_actions_per_node : Dict[UUID, List[ProcAction]]
        A dictionary of ongoing processing actions per node
    earliest_proc_start_time : datetime
        The earliest start time for processing actions for the node
    timestamp : datetime
        The current timestamp

    Returns
    -------
    possible_actions : List[Tuple[CollectionAction, CommsAction, ProcAction]]
        A list of possible actions for the node
    """
    possible_actions = [ActionTuple()]  # No action
    node_id = node.id
    # NOTE: This is an approximation of asset fov in lat/long degrees (1 degree = 111km)
    asset_fov_ll = fov_radius / 111
    # Get all possible collect locations
    possible_collect_locations = []
    for roi in rois:
        x1 = roi.corners[0].longitude
        y1 = roi.corners[0].latitude
        x2 = roi.corners[1].longitude
        y2 = roi.corners[1].latitude
        # For each roi, find the minimum number of overlapping circles required to cover it
        possible_collect_locations += cover_rectangle_with_minimum_overlapping_circles(
            x1, y1, x2, y2, asset_fov_ll
        )
    processing_nodes = get_processing_nodes(network)
    for location in possible_collect_locations:
        geo_location = GeoLocation(latitude=location[1], longitude=location[0], altitude=0)
        # Create dummy image
        image = Image(
            id=uuid4(),
            collection_time=timestamp,
            size=1,
            location=geo_location,
            fov_radius=fov_radius,
            node_id=node_id
        )
        # Create a collection action for each possible location
        coll_action = CollectionAction(
            id=uuid4(),
            node_id=node_id,
            location=geo_location,
            status=ActionStatus.CREATED,
            image=image
        )
        # Create comms and processing actions for each processing node
        for proc_node in processing_nodes:
            dt = timedelta(seconds=0)
            earliest_proc_start_time_tmp = earliest_proc_start_time
            if proc_node.id != node_id:
                # If the processing node is not the same as the collection node, then we
                # need to consider a comms action with the appropriate traversal time
                edge = get_edge(network, node_id, proc_node.id)
                dt = sample_dt(edge.traversal_time)
                earliest_proc_start_time_tmp = get_earliest_proc_start_time(
                    ongoing_proc_actions_per_node, proc_node.id, timestamp
                )
            comms_action = CommsAction(
                id=uuid4(),
                source_node_id=node_id,
                target_node_id=proc_node.id,
                image=image,
                status=ActionStatus.CREATED,
                start_time=timestamp,
                dt=dt
            )
            for algorithm in proc_node.processing_capability.algorithms:
                # Create a processing action for each algorithm
                proc_action = ProcAction(
                    id=uuid4(),
                    node_id=proc_node.id,
                    algorithm=algorithm,
                    image=image,
                    status=ActionStatus.CREATED,
                    start_time=max(comms_action.end_time, earliest_proc_start_time_tmp),
                    dt=sample_dt(algorithm.processing_statistics)
                )
                # Add triple of actions to list of possible actions
                possible_actions.append(ActionTuple((coll_action, comms_action, proc_action)))

    return possible_actions


def enumerate_action_configs(image_store, network, assets, rfis, ongoing_actions, timestamp):

    # Extract all rois from rfis
    rois = extract_rois(rfis)
    processing_nodes = get_processing_nodes(network)
    available_assets = [asset for asset in assets.assets
                        if asset.asset_status.availability == Availability.AVAILABLE]

    ongoing_proc_actions_per_node = {
        node.id: sorted([action for action in ongoing_actions['proc']
                         if action.node_id == node.id], key=lambda a: a.end_time)
        for node in processing_nodes
    }

    # Iterate over the processing nodes and enumerate all possible actions for each node
    possible_actions_per_node = dict()
    for node in processing_nodes:
        node_id = node.id
        # If there are ongoing actions for this node, set the earliest start time to the end time
        # of the latest ongoing action
        earliest_start_time = get_earliest_proc_start_time(
            ongoing_proc_actions_per_node, node_id, timestamp
        )
        # Check if the node is an asset (sensor)
        asset = next((asset for asset in available_assets
                      if asset.asset_description.id == node_id), None)
        # If the node is not an asset (sensor), then we don't need to consider collection actions
        if asset is None:
            possible_actions_per_node[node.id] = enumerate_actions_for_processing_node(
                node, image_store, network, ongoing_actions, earliest_start_time,
                timestamp
            )
        # If the node is an asset (sensor), then we need to consider collection actions
        else:
            possible_actions_per_node[node.id] = enumerate_actions_for_asset(
                node, asset.asset_description.fov_radius, network, rois,
                ongoing_proc_actions_per_node, earliest_start_time, timestamp
            )

    # Enumerate all possible action configurations
    possible_action_configs = []
    num_images_to_be_processed = len(image_store.images) - np.sum([
        exists_ongoing_comms_proc_action_for_image(ongoing_actions, image.id)
        for image in image_store.images
    ])
    for config in itertools.product(*possible_actions_per_node.values()):
        num_collection_actions = sum(1 for action in config if action and action.coll_action)
        # If the number of collection actions is less than the number of available assets, then
        # this is certainly not an optimal action configuration
        if num_collection_actions < len(available_assets):
            continue

        # Filter out action combinations where some remote processing nodes do nothing, when
        # there are available images to be processed
        num_remote_proc_actions = \
            sum(1 for action in config
                if action and action.type in (ActionTupleType.COMMS_AND_PROC,
                                              ActionTupleType.PROC_ONLY)
                )
        # If the number of remote processing actions is less than the number of images to be
        # processed, then this is certainly not an optimal action configuration
        if num_remote_proc_actions < num_images_to_be_processed:
            continue

        # Make a deep copy of the combination, so that we can modify its elements without
        # affecting the original ones
        config = copy.deepcopy(config)

        # Get all processing actions in the configuration
        proc_actions = [action.proc_action for action in config if action]

        # Ensure that no more than one processing action is performed on an image
        image_ids_to_be_processed = [action.image.id for action in proc_actions]
        # If an image id appears more than once in the list, then there is more than one
        # processing action for that image, hence we discard this action configuration
        if len(set(image_ids_to_be_processed)) != len(image_ids_to_be_processed):
            continue

        # Check that processing actions are consistent with each other (i.e. no overlapping
        # processing actions). This is done by ensuring that the start time of each processing
        # action is after the end time of the previous processing action for the same node.
        proc_node_ids = [action.node_id for action in proc_actions]
        # Get processing actions for each node
        proc_actions_per_node = {
            node_id: [action for action in proc_actions if action.node_id == node_id]
            for node_id in proc_node_ids
        }
        for node_id, p_actions in proc_actions_per_node.items():
            # Sort processing actions by start time
            p_actions.sort(key=lambda a: a.start_time)
            if len(p_actions) == 1:
                continue
            for previous_action, current_action in zip(p_actions, p_actions[1:]):
                # If the start time of the current processing action is before the end time of
                # the previous processing action, then set the start time of the current
                # processing action to the end time of the previous processing action
                if current_action.start_time < previous_action.end_time:
                    current_action.start_time = previous_action.end_time
        possible_action_configs.append(config)

    return possible_action_configs


def proc_actions_from_config_sequence(config_seq):
    proc_actions = []
    for config in config_seq:
        for action in config:
            if action is not None and action[2] is not None:
                proc_actions.append(action[2])
    return proc_actions


def queue_actions(config, image_store, ongoing_actions):
    """Queue actions for execution

    Add collection, comms and processing actions to the ongoing actions list
    """
    for node_actions in config:
        if not node_actions:
            continue
        coll_action, comms_action, proc_action = node_actions
        # If collection action is None or collection action is not for the same node as
        # processing action, it means that processing is not done on the sensor
        if coll_action is None or coll_action.node_id != proc_action.node_id:
            # Perform collection action
            if coll_action is not None:
                image_store.images.append(proc_action.image)
            # Perform comms action
            if comms_action is not None:
                if comms_action.dt != 0:
                    # If comms action is not instantaneous, add it to ongoing actions
                    ongoing_actions['comms'].append(comms_action)
                else:
                    # If comms action is instantaneous, find image and update node_id to
                    # processing node id
                    image = next(image for image in image_store.images
                                 if image.id == comms_action.image_id)
                    image.node_id = proc_action.node_id
        # Perform processing action
        ongoing_actions['proc'].append(proc_action)


def rollout_actions(config, image_store, network_topology, assets, rfis,
                    ongoing_actions, num_samples, num_timesteps, interval, timestamp):
    """Rollout actions for a given configuration

    Returns a list of lists of action configs, where each list of action configs is a rollout
    """

    # Initialise list of action configs. The first action config is the same for all rollouts
    all_configs = [[config] for _ in range(num_samples)]

    # For each rollout
    for config_list in all_configs:
        current_time = timestamp
        # Copy image store and ongoing actions, so that they can be modified without affecting
        # other rollouts
        image_store_tmp = copy.deepcopy(image_store)
        ongoing_actions_tmp = copy.deepcopy(ongoing_actions)
        # Queue actions for execution
        queue_actions(config, image_store_tmp, ongoing_actions_tmp)
        # Rollout
        for i in range(num_timesteps):
            current_time += interval
            # Get all possible action configs for the current time
            configs = enumerate_action_configs(image_store_tmp, network_topology, assets,
                                               rfis, ongoing_actions_tmp, current_time)
            # Select a random action config and add it to the rollout action configs
            current_config = configs[np.random.randint(len(configs))]
            config_list.append(current_config)
            # Queue actions for execution
            queue_actions(current_config, image_store_tmp, ongoing_actions_tmp)
    return all_configs


def get_sensor(location, fov_radius, prob_detection=None, false_alarm_density=None):
    # Create a sensor
    sensor_position = StateVector([location.longitude,
                                   location.latitude,
                                   location.altitude])
    if prob_detection is not None:
        prob_detection = Probability(prob_detection)
    sensor = MovableUAVCamera(ndim_state=6, mapping=[0, 2, 4],
                              noise_covar=np.diag([0.0001, 0.0001, 0.0001]),
                              location_x=sensor_position[0],
                              location_y=sensor_position[1],
                              position=sensor_position,
                              prob_detect=prob_detection,
                              fov_radius=fov_radius,
                              fov_in_km=True)
    # Configure the sensor clutter model based on the algorithm false alarm density
    # and the image footprint
    x, y = sensor.footprint.exterior.coords.xy
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    if false_alarm_density:
        clutter_model = ClutterModel(
            clutter_rate=false_alarm_density,
            distribution=np.random.default_rng().uniform,
            dist_params=((min_x, max_x), (min_y, max_y), (100., 100.))
        )
        sensor.clutter_model = clutter_model
    return sensor


def simulate_new_tracks(sensor, timestamp, birth_density):
    x, y = sensor.footprint.exterior.coords.xy
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    # Assume a Poisson process for the number of tracks. Tracks position are generated
    # uniformly across the image footprint, with a random velocity between 0 and 10 m/s
    new_tracks = set()
    for _ in range(poisson.rvs(1)):
        # Sample position
        x = np.random.default_rng().uniform(min_x, max_x)
        y = np.random.default_rng().uniform(min_y, max_y)
        z = 100.
        state_vector = StateVector([x, 0, y, 0, z, 0])
        # Extract covariance from birth density and adjust position covariance
        covariance = np.copy(birth_density.covar)
        covariance[[0, 2], [0, 2]] = np.random.default_rng().uniform(0.01, 0.1) # 0.01

        # Sample existence probability
        exist_prob = Probability(np.random.default_rng().uniform(0.5, 1)) #0.99

        # Sample target type
        all_target_types = list(TargetType)
        num_types = np.random.randint(1, len(all_target_types) + 1)
        target_type_inds = np.random.choice([i for i in range(len(TargetType))],
                                            size=num_types, replace=False)
        target_types = [all_target_types[i] for i in target_type_inds]
        target_type_confidences = {
            target_type: Probability(np.random.default_rng().uniform(0.1, 1))
            for target_type in target_types
        }

        # Create track
        init_metadata = {
            'target_type_confidences': target_type_confidences,
        }
        state = GaussianState(state_vector, covariance, timestamp=timestamp)
        track = Track(id=uuid4(), states=[state], init_metadata=init_metadata)
        track.exist_prob = exist_prob
        new_tracks.add(track)
    return new_tracks

