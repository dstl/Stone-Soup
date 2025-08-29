from abc import abstractmethod, ABC
from functools import lru_cache
from typing import Set, Union, Sequence, TYPE_CHECKING
try:
    from shapely import STRtree
    from shapely.geometry import Polygon, Point, MultiPoint, LineString, MultiLineString
    shapely = True
except ImportError:
    shapely = False

import numpy as np

from ..sensormanager.action import Actionable
from .base import PlatformMountable
from ..base import Property
from ..models.clutter.clutter import ClutterModel
from ..types.detection import TrueDetection, Detection
from ..types.groundtruth import GroundTruthState
from ..types.state import ParticleState, State, StateVector

if TYPE_CHECKING:
    from ..platform.base import Obstacle


class Sensor(PlatformMountable, Actionable):
    """Sensor Base class for general use.

    Most properties and methods are inherited from :class:`~.PlatformMountable`.

    Notes
    -----
    * Sensors must have a measure function.
    * Attributes that are modifiable via actioning the sensor should be
      :class:`~.ActionableProperty` types.
    * The sensor has a timestamp property that should be updated via its
      :meth:`~Actionable.act` method.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.timestamp = None

    def validate_timestamp(self):

        if self.timestamp:
            return True

        try:
            self.timestamp = self.movement_controller.state.timestamp
        except AttributeError:
            return False
        if self.timestamp is None:
            return False
        return True

    @abstractmethod
    def measure(self, ground_truths: set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> set[TrueDetection]:
        """Generate a measurement for a given state

        Parameters
        ----------
        ground_truths : Set[:class:`~.GroundTruthState`]
            A set of :class:`~.GroundTruthState`
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is `True`, in which
            case :meth:`~.Model.rvs` is used; if `False`, no noise will be added)

        Returns
        -------
        Set[:class:`~.TrueDetection`]
            A set of measurements generated from the given states. The timestamps of the
            measurements are set equal to that of the corresponding states that they were
            calculated from. Each measurement stores the ground truth path that it was produced
            from.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def measurement_model(self):
        """Measurement model of the sensor, describing general sensor model properties"""
        raise NotImplementedError


class SimpleSensor(Sensor, ABC):

    clutter_model: ClutterModel = Property(
        default=None,
        doc="An optional clutter generator that adds a set of simulated "
            ":class:`Clutter` objects to the measurements at each time step. "
            "The clutter is simulated according to the provided distribution.")

    def measure(self, ground_truths: set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> set[TrueDetection]:

        measurement_model = self.measurement_model

        detectable_ground_truths = [truth for truth in ground_truths
                                    if self.is_detectable(truth, measurement_model)]

        if noise is True:
            if len(detectable_ground_truths) > 1:
                noise_vectors_iter = iter(measurement_model.rvs(len(detectable_ground_truths),
                                                                **kwargs))
            else:
                noise_vectors_iter = iter([measurement_model.rvs(**kwargs)])

        detections = set()
        for truth in detectable_ground_truths:
            measurement_vector = measurement_model.function(truth, noise=False, **kwargs)

            if noise is True:
                measurement_noise = next(noise_vectors_iter)
            else:
                measurement_noise = noise

            # Add in measurement noise to the measurement vector
            measurement_vector += measurement_noise

            detection = TrueDetection(measurement_vector,
                                      measurement_model=measurement_model,
                                      timestamp=truth.timestamp,
                                      groundtruth_path=truth)
            detections.add(detection)

        # Generate clutter at this time step
        if self.clutter_model is not None:
            self.clutter_model.measurement_model = measurement_model
            clutter = self.clutter_model.function(ground_truths)
            detectable_clutter = [cltr for cltr in clutter
                                  if self.is_clutter_detectable(cltr)]
            detections = set.union(detections, detectable_clutter)

        return detections

    @abstractmethod
    def is_detectable(self, state: GroundTruthState, measurement_model=None) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_clutter_detectable(self, state: Detection) -> bool:
        raise NotImplementedError

    def is_visible(self, state: State) -> bool:
        return True


class SensorSuite(Sensor):
    """Sensor composition type

    Models a suite of sensors all returning detections at the same 'time'. Returns all detections
    in one go.
    Can append information of the sensors to the metadata of their corresponding detections.
    """

    sensors: Sequence[Sensor] = Property(doc="Suite of sensors to get detections from.")

    attributes_inform: set[str] = Property(
        doc="Names of attributes to store the value of at time of detection."
    )

    def measure(self, ground_truths: set[GroundTruthState], noise: Union[bool, np.ndarray] = True,
                **kwargs) -> set[TrueDetection]:
        """Call each sub-sensor's measure method in turn. Key word arguments are passed to the
        measure method of each sensor.

        Append additional metadata to each sensor's set of detections. Which keys are appended is
        dictated by :attr:`attributes_inform`."""

        all_detections = set()

        for sensor in self.sensors:

            detections = sensor.measure(ground_truths, noise, **kwargs)

            attributes_dict = {attribute_name: sensor.__getattribute__(attribute_name)
                               for attribute_name in self.attributes_inform}

            for detection in detections:
                detection.metadata.update(attributes_dict)

            all_detections.update(detections)

        return all_detections

    @property
    def measurement_model(self):
        """Measurement model of the sensor, describing general sensor model properties"""
        raise NotImplementedError


class VisibilityInformed2DSensor(SimpleSensor):
    """The base class of 2D sensors that evaluate the visibility of
    targets in known cluttered environments. Two different techniques
    are adopted for visibility checking. The first is a ray casting
    approach that is used with small to modest numbers of obstacles.
    The second adopts the STR Tree algorithm which is more efficient
    for large numbers of obstacles.
    """
    # TODO: Establish the suitable number of obstacles to use when
    # switching between STR Tree and ray casting.

    obstacles: Set['Obstacle'] = Property(default=None,
                                          doc="Set of :class:`~.Obstacle` type "
                                              "platforms that represent obstacles in the "
                                              "environment")

    moving_obstacle_flag: bool = Property(default=False,
                                          doc="Boolean flag indicating if obstacles are mobile")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.obstacles is not None and shapely and len(self.obstacles) > 100:
            self._str_tree_is_visible_trigger = True
            self._obstacle_tree = \
                STRtree([Polygon(obstacle.vertices.T) for obstacle in self.obstacles])
        else:
            self._str_tree_is_visible_trigger = False

        self._relevant_obs = []
        self._relevant_obs_idx = []

        if self.obstacles:
            self._all_verts = [obstacle.vertices for obstacle in self.obstacles]
            self._all_rel_edges = [obstacle.relative_edges for obstacle in self.obstacles]
        else:
            self._all_verts = []
            self._all_rel_edges = []

    @lru_cache(maxsize=None)
    def _position_cache(self):
        # Cache for position allows for vertices and relative edges to be
        # calculated only when necessary. Maxsize set to unlimited as it
        # is cleared before assigning a new value
        return self.position

    @property
    def _relevant_obstacles(self):
        self._get_relevant_obstacles()
        # return self._relevant_obs
        return self._relevant_obs_idx

    def _get_relevant_obstacles(self):

        if self.moving_obstacle_flag:
            # Call vertices for each obstacle to update self._all_verts and self._all_rel_edges
            _ = [obstacle.vertices for obstacle in self.obstacles]

        if self.max_range < np.inf and (np.any(self._position_cache() != self.position) or
                                        not self._relevant_obs):

            self._position_cache.cache_clear()
            self._relevant_obs_idx = \
                np.where([np.any(np.sqrt(
                                         np.sum((vertices[0:2, :]-self.position[0:2])**2, axis=0))
                                 < self.max_range)
                         for vertices in self._all_verts])[0].astype(int)
        else:
            self._relevant_obs_idx = \
                np.linspace(0, len(self.obstacles)-1, len(self.obstacles)).astype(int)

    def get_obstacle_tree(self):

        if self.moving_obstacle_flag:
            return STRtree([Polygon(obstacle.vertices.T) for obstacle in self.obstacles])
        else:
            return self._obstacle_tree

    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> Set[TrueDetection]:

        # In the event that obstacles have been pased to the measure function,
        # they are removed from the set of known obstacles.
        if isinstance(self, VisibilityInformed2DSensor) and self.obstacles:
            ground_truths = ground_truths - self.obstacles

        return super().measure(ground_truths, noise, **kwargs)

    def is_visible(self, state):
        """Function for evaluating the visibility of states in the
        environment based on a 2D line of sight intersection check with
        obstacles edges. Note that this method does not check sensor field of
        view in evaluating visibility. If no obstacles are provided, the
        method will return `True` or `True` array of equivalent shape of state.

        Parameters
        ----------
        state : :class:`~.State`
            A state object that describes `n` positions to check line of sight to from
            the sensor position.

        Returns
        -------
        : :class:`~numpy.ndarray`
            (1, n) array of booleans indicating the visibility of `state`. True represents
            that the state is visible."""

        # Check number of states if `state` is `ParticleState`
        if isinstance(state, ParticleState):
            nstates = len(state)
        else:
            nstates = 1

        if not self.obstacles:
            return np.full(nstates, True)

        if self._str_tree_is_visible_trigger:

            if isinstance(state, StateVector):
                line_segments = \
                    [LineString([self.position[0:2], state[self.position_mapping[0:2], :]])]
            else:
                if isinstance(state, ParticleState):
                    position_concat = np.tile(self.position, [nstates])
                    line_segments = \
                        MultiLineString(
                            [*np.array([position_concat,
                                        state.state_vector[self.position_mapping[0:2], :]])
                             .transpose(2, 0, 1)]).geoms
                else:
                    nstates = 1
                    line_segments = \
                        [LineString([self.position[0:2],
                                     state.state_vector[self.position_mapping[0:2], :]])]

            intersections = np.full((nstates,), True)

            obstacle_tree = self.get_obstacle_tree()

            non_vis_rays = obstacle_tree.query(line_segments, predicate='intersects')[0, :]
            intersections[np.unique(non_vis_rays)] = False

            return intersections

        else:

            intersections = self._ray_cast_check(state, nstates)

            intersections = np.invert(np.any(intersections, 0))
            if nstates == 1:
                intersections = intersections[0]

            return intersections

    def in_obstacle(self, state):
        """Function for evaluating whether states are inside the boundry of obstacles
        in the environment. If no obstacles are provided, the method will return
        `True` or `True` array of equivalent shape of state.

        Parameters
        ----------
        state : :class:`~.State`
            A state object that describes `n` positions to check line of sight to from
            the sensor position.

        Returns
        -------
        : :class:`~numpy.ndarray`
            (1, n) array of booleans indicating whether `state` is inside an obstacle
            and is `True` when a state is inside an obstacle."""

        if isinstance(state, ParticleState):
            nstates = len(state)
        else:
            nstates = 1

        in_obstacles = np.full((nstates), False)
        if not self.obstacles:
            return in_obstacles

        if self._str_tree_is_visible_trigger:

            if isinstance(state, StateVector):
                point_sequence = [Point(state[self.position_mapping[0:2], :].T)]
            else:
                if isinstance(state, ParticleState):
                    point_sequence = \
                        MultiPoint(state.state_vector[self.position_mapping[0:2], :].T).geoms
                else:
                    point_sequence = [Point(state.state_vector[self.position_mapping[0:2], :].T)]

            obstacle_tree = self.get_obstacle_tree()

            in_obs_states = obstacle_tree.query(point_sequence, predicate='within')[0, :]
            in_obstacles[in_obs_states] = True
        else:

            intersections = self._ray_cast_check(state, nstates)

            in_obstacles = sum(intersections, 0) % 2 != 0

        return in_obstacles

    def _ray_cast_check(self, state, nstates):
        # method for performing raycast visibility check.

        relevant_obstacle_idx = self._relevant_obstacles

        relative_edges = np.hstack([self._all_rel_edges[n] for n in relevant_obstacle_idx])

        # Calculate relative vector between sensor position and state position
        if isinstance(state, StateVector):
            relative_ray = np.array([state[self.position_mapping[0], :]
                                    - self.position[0, 0],
                                    state[self.position_mapping[1], :]
                                    - self.position[1, 0]])
        else:
            relative_ray = np.array([state.state_vector[self.position_mapping[0], :]
                                    - self.position[0, 0],
                                    state.state_vector[self.position_mapping[1], :]
                                    - self.position[1, 0]])

        relative_sensor_to_edge = self.position[0:2] - \
            np.hstack([self._all_verts[n] for n in relevant_obstacle_idx])

        # Initialise the intersection vector
        intersections = np.full((relative_edges.shape[1], nstates), False)

        # Perform intersection check
        for n in range(relative_edges.shape[1]):
            denom = relative_ray[1, :]*relative_edges[0, n] \
                - relative_ray[0, :]*relative_edges[1, n]
            alpha = (relative_edges[1, n]*relative_sensor_to_edge[0, n]
                     - relative_edges[0, n]*relative_sensor_to_edge[1, n])/denom
            beta = (relative_ray[0, :]*relative_sensor_to_edge[1, n]
                    - relative_ray[1, :]*relative_sensor_to_edge[0, n])/denom

            intersections[n, :] = np.logical_and.reduce((alpha >= 0,
                                                         alpha <= 1,
                                                         beta >= 0,
                                                         beta <= 1))
            intersections[n, :] = (alpha >= 0) & (alpha <= 1) & (beta >= 0) & (beta <= 1)

        return intersections
