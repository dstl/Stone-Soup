import numpy as np
import datetime
import os
import sys
from typing import Collection, Sequence
from enum import Enum
import traci
from shapely.geometry import Polygon
from shapely import union_all, is_valid

from .base import GroundTruthReader
from ..base import Property
from ..types.array import StateVector
from ..types.groundtruth import GroundTruthState, GroundTruthPath
from ..types.state import State
from ..buffered_generator import BufferedGenerator
from ..platform.base import Obstacle


class SUMOGroundTruthReader(GroundTruthReader):
    r"""A Groundtruth reader for a SUMO simulation.

    At each time step, kinematic information from the objects in the SUMO simulation will be
    extracted and placed into a :class:`~.GroundTruthState`. States with the same ID will be placed
    into a :class:`~.GroundTruthPath` in sequence.

    The state vector for each truth object is, by default,  of the form:

    .. math::

        [\mathbf{x}, \mathbf{v}_x, \mathbf{y}, \mathbf{v}_y, \mathbf{z}]

    .. note::

        By default, the Carteisan co-ordinates use the UTM-projection with the origin shifted such
        that the bottom left corner of the network is the origin (0,0).
        See: https://sumo.dlr.de/docs/Geo-Coordinates.html

        To extract lat/lon co-ordinates, it is required that the network is geo-referenced.

        This reader requires the installation of SUMO, see:
        https://sumo.dlr.de/docs/Installing/index.html

        This reader requires a SUMO configuration file.

    Parameters
    ----------
    """
    sumo_cfg_file_path: str = Property(
        doc='Path to SUMO config file')

    sumo_server_path: str = Property(
        doc='Path to SUMO server, relative from SUMO_HOME environment variable. '
            '"/bin/sumo" to run on command line "/bin/sumo-gui" will run using the SUMO-GUI, '
            'this will require pressing play within the GUI.')

    sim_start: datetime.datetime = Property(
        default=None,
        doc='Start time for the simulation. Will default to datetime.datetime.now()')

    sim_steps: int = Property(
        default=200,
        doc='Number of steps you want your SUMO simulation to run for. Use numpy.inf to have no '
            'limit')

    position_mapping: Sequence[int] = Property(
        default=[0, 2, 4],
        doc='Mapping for x, y, z position in state vector')

    velocity_mapping: Sequence[int] = Property(
        default=[1, 3],
        doc='Mapping for x and y velocities in state vector')

    person_metadata_fields: Collection[str] = Property(
        default=None,
        doc='Collection of metadata fields for people that will be added to the metadata of '
            'each GroundTruthState. Possible fields are documented in '
            'https://sumo.dlr.de/docs/TraCI/Person_Value_Retrieval.html. See also '
            'PersonMetadataEnum. Underscores are required in place of spaces. '
            'An example would be: ["speed", "color", "slope", "road_id]')

    vehicle_metadata_fields: Collection[str] = Property(
        default=None,
        doc='Collection of metadata fields for vehicles that will be added to the metadata'
            ' of each GroundTruthState. Possible fields are documented in '
            'https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html. See also '
            'VehicleMetadataEnum. Underscores are required in place of spaces. '
            'An example would be: ["speed", "acceleration", "lane_position"]')

    geographic_coordinates: bool = Property(
        default=False,
        doc='If True, geographic co-ordinates (longitude, latitude) will be added to the metadata '
            'of each state, as well as the lat/lon of the origin of the local co-ordinate frame')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If network is geo-referenced, and geographic_coordinates == True,
        # then self.origin will be lat/lon position of the origin of the local Cartesian frame.
        self.origin = None
        self.step = 0
        # Resort to default for sim_start
        if self.sim_start is None:
            self.sim_start = datetime.datetime.now()

        self.sumoCmd = [self.sumo_server_path, "-c", self.sumo_cfg_file_path]

        # Raise errors if metadata fields are incorrect
        if set(self.person_metadata_fields) & {data.name for data in PersonMetadataEnum} != set(
               self.person_metadata_fields):
            raise ValueError(f"""Invalid person metadata field(s): {', '.join(str(field) for
                            field in self.person_metadata_fields if field not in
                            [data.name for data in PersonMetadataEnum])}""")

        if set(self.vehicle_metadata_fields) & {data.name for data in VehicleMetadataEnum} != set(
               self.vehicle_metadata_fields):
            raise ValueError(f"""Invalid vehicle metadata field(s): {', '.join(str(field) for
                             field in self.vehicle_metadata_fields if field not in
                             [data.name for data in VehicleMetadataEnum])}""")

        if 'SUMO_HOME' in os.environ:
            # Add SUMO_HOME env variable. Use this for server/config-file paths
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            raise RuntimeError("Environment variable 'SUMO_HOME' is not set")

    @staticmethod
    def calculate_velocity(speed, angle, radians=False):
        if not radians:
            v_x, v_y = speed * np.sin(np.radians(angle)), speed * np.cos(np.radians(angle))
        elif radians:
            v_x, v_y = speed * np.sin(angle), speed * np.cos(angle)
        return v_x, v_y

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self):

        traci.start(self.sumoCmd)
        if self.geographic_coordinates:
            self.origin = traci.simulation.convertGeo(0, 0)   # lon, lat

        groundtruth_dict = dict()
        while self.step < self.sim_steps:
            # Need to get id list at each timestamp since not all ids may be present through the
            # whole of the simulation (spawning, exiting etc).
            vehicle_ids = traci.vehicle.getIDList()
            person_ids = traci.person.getIDList()

            # Update simulation time
            time = self.sim_start + datetime.timedelta(seconds=traci.simulation.getTime())
            updated_paths = set()

            # Loop through people
            for id_ in person_ids:
                if id_ not in groundtruth_dict.keys():
                    groundtruth_dict[id_] = GroundTruthPath(id=id_)
                    # Subscribe to all specified metadata fields
                    traci.person.subscribe(id_, tuple(data.value for data in PersonMetadataEnum if
                                                      data.name in self.person_metadata_fields))

                # Initialise and insert StateVector information
                state_vector = StateVector([0.]*5)
                np.put(state_vector, self.position_mapping, traci.person.getPosition3D(id_))
                np.put(state_vector, self.velocity_mapping,
                       self.calculate_velocity(traci.person.getSpeed(id_),
                                               traci.person.getAngle(id_),
                                               radians=False))

                # Get information that is subscribed to for metadata
                subscription = traci.person.getSubscriptionResults(id_)
                metadata = {PersonMetadataEnum(key).name: subscription[key]
                            for key in subscription.keys()}
                # Add latitude / longitude to metadata
                if self.geographic_coordinates:
                    lon, lat = traci.simulation.convertGeo(*state_vector[self.position_mapping, :])
                    metadata['longitude'] = lon
                    metadata['latitude'] = lat

                # Add co-ordinates of origin to metadata
                if self.origin:
                    metadata['origin'] = self.origin

                state = GroundTruthState(
                    state_vector=state_vector,
                    timestamp=time,
                    metadata=metadata)

                groundtruth_path = groundtruth_dict[id_]
                groundtruth_path.append(state)
                updated_paths.add(groundtruth_path)

            # Loop through vehicles
            for id_ in vehicle_ids:
                if id_ not in groundtruth_dict.keys():
                    groundtruth_dict[id_] = GroundTruthPath(id=id_)
                    # Subscribe to all specified metadata fields
                    traci.vehicle.subscribe(id_,
                                            tuple(data.value for data in VehicleMetadataEnum if
                                                  data.name in self.vehicle_metadata_fields))

                # Initialise and insert StateVector information
                state_vector = StateVector([0.]*5)
                np.put(state_vector, self.position_mapping, traci.vehicle.getPosition3D(id_))
                np.put(state_vector, self.velocity_mapping,
                       self.calculate_velocity(traci.vehicle.getSpeed(id_),
                                               traci.vehicle.getAngle(id_),
                                               radians=False))

                # Get information that is subscribed to for metadata
                subscription = traci.vehicle.getSubscriptionResults(id_)
                metadata = {VehicleMetadataEnum(key).name: subscription[key]
                            for key in subscription.keys()}

                # Add latitude / longitude to metadata
                if self.geographic_coordinates:
                    lon, lat = traci.simulation.convertGeo(*state_vector[self.position_mapping, :])
                    metadata['longitude'] = lon
                    metadata['latitude'] = lat

                # Add co-ordinates of origin to metadata
                if self.origin:
                    metadata['origin'] = self.origin

                state = GroundTruthState(
                    state_vector=state_vector,
                    timestamp=time,
                    metadata=metadata)

                groundtruth_path = groundtruth_dict[id_]
                groundtruth_path.append(state)
                updated_paths.add(groundtruth_path)

            # Progress simulation
            traci.simulationStep()

            self.step += 1
            yield time, updated_paths

        traci.close()

    def obstacle_gen(self):
        """
        Import polygons from SUMO as Obstacle platforms. All polygons
        with a type id starting with `'building'` will be imported as
        an obstacle. Individual buildings with shared faces, or part
        shared faces, between a neighborough building will be merged
        into a single obstacle.

        Returns
        -------
        : list[Obstacle],
            List of :class:`~.Obstacle` platforms.
        """
        traci.start(self.sumoCmd)
        if self.geographic_coordinates:
            self.origin = traci.simulation.convertGeo(0, 0)   # lon, lat

        # get polygon IDs
        polygon_ids = traci.polygon.getIDList()

        # Loop through polygon IDs
        raw_obstacles = []
        obstacles = []
        for id_ in polygon_ids:
            # Only retain 'building' polygons
            if traci.polygon.getType(id_).startswith('building'):
                # Get the shape information
                shape_ = traci.polygon.getShape(id_)
                # Check that the polygon is closed (filled), if it has more than two vertices
                # (excluding the final vertex) and if the shape is valid according to Shapely
                if traci.polygon.getFilled(id_) and \
                        len(shape_[:-1]) > 2 and \
                        is_valid(Polygon(shape_[:-1])):
                    raw_obstacles.append(Polygon(shape_[:-1]))

        traci.close(wait=False)

        # Merge touching polygons
        merged_obstalces = union_all(raw_obstacles)

        # Create obstacle platforms from the data
        for shape in merged_obstalces.geoms:
            state = State(StateVector(shape.centroid.xy))
            obstacles.append(Obstacle(shape_data=shape.exterior.xy-state.state_vector,
                                      states=state,
                                      orientation=StateVector([0, 0, 0]),
                                      position_mapping=(0, 1)))

        return obstacles

    def road_network_gen(self, vehicles_of_interest=None):
        """
        Import SUMO road network. Each edge in SUMO contains a minimum of one,
        straight, line segment and typically multiple straight line segments to
        approximate curvature. This method extracts coordinates for each line segments
        within each edge in the provided SUMO config.

        Parameters
        ----------
        vehicles_of_interest: Collection[str], optional
            Collection of vehicle types from :class:`~.VehicleTypeEnum`
            specifying roads to extract from SUMO. Default value is None
            and all roads will be processed. If populated with valid vehicle
            types, only edges that allow the vehicle type are processed and returned.

        Returns
        -------
        : dict
            Pairs of `edge_id`: `edge_coordinates`. `edge_id` is a
            SUMO assigned ID number for each edge in the SUMO config.
            `edge_coordinates` is an `np.ndarray` with shape (2, n_segments),
            where the first axis corresponds to :math:`x` and :mathL`y` dimensions
            and the second axis corresponds to the number of straigt line segments
            in `edge_id`.

        """

        if (vehicles_of_interest and
            not all(type_ in [data.name for data in VehicleTypeEnum]
                    for type_ in vehicles_of_interest)):
            raise ValueError(f"""Invalid vehicles_of_interest value(s): {', '.join(str(field) for
                             field in vehicles_of_interest if field not in
                             [data.name for data in VehicleTypeEnum])}""")

        traci.start(self.sumoCmd)
        if self.geographic_coordinates:
            self.origin = traci.simulation.convertGeo(0, 0)   # lon, lat

        # Get list of 'edges' that define the road network
        edge_IDs = list(traci.edge.getIDList())

        # Initialise output dict
        road_network = {}
        complete_list = []

        # Each edge has a single 'FromJunction' and 'ToJunction'. Using
        # each junction position, line segments constructing the road
        # network can be defined.
        for edge_ID in edge_IDs:
            if edge_ID in complete_list:
                continue
            # Checks to see if `edge_ID` has a complementarty edge travelling in
            # the opposite direction. If it does these will be processed simultaneously
            # such that only one edge is added to the road network.
            if edge_ID[0] == '-' and edge_ID[1:] in edge_IDs:
                alternative_edge = edge_ID[1:]
            elif '-'+edge_ID in edge_IDs:
                alternative_edge = '-'+edge_ID
            else:
                alternative_edge = None

            if alternative_edge:

                # Get number of lanes for both edges
                lane_count = [traci.edge.getLaneNumber(edge_ID)]
                lane_count.append(traci.edge.getLaneNumber(alternative_edge))

                # Initialise lists
                allowed_vehicles, shape_1, shape_2 = [], [], []
                # Get shape data and allowed vehicles for each lane in the first edge
                for n in range(lane_count[0]):
                    shape_1.append(np.array(traci.lane.getShape(edge_ID+'_'+str(n))))
                    allowed_vehicles.extend(list(traci.lane.getAllowed(edge_ID+'_'+str(n))))
                # Remove duplicate allowed vehicle entries
                allowed_vehicles = list(set(allowed_vehicles))

                # If the desired vehicles has been defined (not None) and it is not in the
                # list of allowed vehicles on the edge, continue to the next `edge_ID`.
                if vehicles_of_interest and not any(type_ in allowed_vehicles for type_ in
                                                    vehicles_of_interest):
                    continue

                # processes shape of `edge_ID` by checking for different numbers of
                # defining points and handling this appropriately.
                shape_1 = self._process_lane_shape(shape_1)

                # Repeat the above process for the complementory edge. Shape of edge
                # is inverted to account for opposite directionality
                allowed_vehicles = []
                for n in range(lane_count[1]):
                    shape_2.append(
                        np.flipud(np.array(traci.lane.getShape(alternative_edge+'_'+str(n)))))
                    allowed_vehicles.extend(
                        list(traci.lane.getAllowed(alternative_edge+'_'+str(n))))

                # Check complementory edge to see if desired vehcles are permitted. If not,
                # shape information from `edge_ID` is copied to allow the single edge to be
                # included.
                if vehicles_of_interest and not any(type_ in allowed_vehicles for type_ in
                                                    vehicles_of_interest):
                    shape_2 = shape_1

                shape_2 = self._process_lane_shape(shape_2)

                # Checks to see if the shape information for both edges is compatible
                # (same number of points). If not, a simple assignment is calculated to check
                # which points can be merged.
                if shape_1[0].shape != shape_2[0].shape:
                    size_diff = shape_1[0].shape[0] - shape_2[0].shape[0]
                    slice_array = [slice(0, -np.abs(size_diff)), slice(np.abs(size_diff), None)]
                    if size_diff > 0:
                        first_element_assign = \
                            np.ptp(np.sqrt(
                                (shape_1[0][:-int(size_diff), 0]-shape_2[0][:, 0])**2 +
                                (shape_1[0][:-int(size_diff), 1]-shape_2[0][:, 1])**2))
                        last_element_assign = \
                            np.ptp(np.sqrt(
                                (shape_1[0][int(size_diff):, 0]-shape_2[0][:, 0])**2 +
                                (shape_1[0][int(size_diff):, 1]-shape_2[0][:, 1])**2))
                        shape_1 = \
                            [shape_1_[slice_array[np.argmin([first_element_assign,
                                                             last_element_assign])], :]
                                for shape_1_ in shape_1]
                    else:
                        size_diff = np.abs(size_diff)
                        first_element_assign = \
                            np.ptp(np.sqrt(
                                (shape_1[0][:, 0]-shape_2[0][:-int(size_diff), 0])**2 +
                                (shape_1[0][:, 1]-shape_2[0][:-int(size_diff), 1])**2))
                        last_element_assign = \
                            np.ptp(np.sqrt(
                                (shape_1[0][:, 0]-shape_2[0][int(size_diff):, 0])**2 +
                                (shape_1[0][:, 1]-shape_2[0][int(size_diff):, 1])**2))
                        shape_2 = \
                            [shape_2_[slice_array[np.argmin([first_element_assign,
                                                             last_element_assign])], :]
                                for shape_2_ in shape_2]

                # Calculate mean shape of the edges
                mean_lane_shape = np.mean([*shape_1, *shape_2], axis=0)
                # Store the result
                road_network[edge_ID] = mean_lane_shape.T
                # Store list of alternative edge to prevent duplication
                complete_list.append(alternative_edge)
                # edge_IDs.remove(alternative_edge)

            else:
                # Process single edge as above
                lane_count = traci.edge.getLaneNumber(edge_ID)
                allowed_vehicles, shape_1 = [], []
                for n in range(lane_count):
                    shape_1.append(np.array(traci.lane.getShape(edge_ID+'_'+str(n))))
                    allowed_vehicles.extend(list(traci.lane.getAllowed(edge_ID+'_'+str(n))))
                allowed_vehicles = list(set(allowed_vehicles))
                if vehicles_of_interest and not any(type_ in allowed_vehicles for type_ in
                                                    vehicles_of_interest):
                    continue

                shape_1 = self._process_lane_shape(shape_1)

                mean_lane_shape = np.mean(shape_1, axis=0)

                road_network[edge_ID] = mean_lane_shape.T

        traci.close(wait=False)

        return road_network

    def _process_lane_shape(self, shape):

        # Checks if an edge contains multiple lanes, and ensures that lanes have
        # same number of segments by only considering the shortest lane.

        if len(set([shape_.shape for shape_ in shape])) != 1:
            shape_vec = np.array([shape_.shape[0] for shape_ in shape])
            min_shape = np.min(shape_vec)

            shape = [shape_[0:end_index, :] for shape_, end_index
                     in zip(shape, shape_vec-(shape_vec-min_shape))]

        return shape


class PersonMetadataEnum(Enum):
    """
    A metadata Enum used to map the named variable of the person to the relevant id. Subscribing
    to this id will retrieve the value and add it to the metadata of the GroundTruthState.
    See https://sumo.dlr.de/docs/TraCI/Person_Value_Retrieval.html for a full list.

    """
    id_list = 0x0
    count = 0x1
    speed = 0x40
    position = 0x42
    position_3D = 0x39
    angle = 0x43
    slope = 0x36
    road_id = 0x50
    type_id = 0x4f
    color = 0x45
    edge_position = 0x56
    length = 0x44
    minGap = 0x4c
    width = 0x4d
    waiting_time = 0x7a
    next_edge = 0xc1
    remaining_stages = 0xc2
    vehicle = 0xc3
    taxi_reservations = 0xc6


class VehicleMetadataEnum(Enum):
    """
    A metadata Enum used to map the named variable of the vehicle to the relevant id. Subscribing
    to this id will retrieve the value and add it to the metadata of the GroundTruthState.
    See https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html for a full list.

    """
    id_list = 0x0
    count = 0x1
    speed = 0x40
    lateral_speed = 0x32
    acceleration = 0x72
    position = 0x42
    position_3D = 0x39
    angle = 0x43
    road_id = 0x50
    lane_id = 0x51
    lane_index = 0x52
    type_id = 0x4f
    route_id = 0x53
    route_index = 0x69
    edges = 0x54
    color = 0x45
    lane_position = 0x56
    distance = 0x84
    signal_states = 0x5b
    routing_mode = 0x89
    TaxiFleet = 0x20
    CO2_emissions = 0x60
    CO_emissions = 0x61
    HC_emissions = 0x62
    PMx_emissions = 0x63
    NOx_emissions = 0x64
    fuel_consumption = 0x65
    noise_emission = 0x66
    electricity_consumption = 0x71
    best_lanes = 0xb2
    stop_state = 0xb5
    length = 0x44
    vmax = 0x41
    accel = 0x46
    decel = 0x47
    tau = 0x48
    sigma = 0x5d
    speedFactor = 0x5e
    speedDev = 0x5f
    vClass = 0x49
    emission_class = 0x4a
    shape = 0x4b
    minGap = 0x4c
    width = 0x4d
    height = 0xbc
    person_capacity = 0x38
    waiting_time = 0x7a
    accumulated_waiting_time = 0x87
    next_TLS = 0x70
    next_stops = 0x73
    person_id_list = 0x1a
    speed_mode = 0xb3
    lane_change_mode = 0xb6
    slope = 0x36
    allowed_speed = 0xb7
    line = 0xbd
    Person_Number = 0x67
    via_edges = 0xbe
    speed_without_TraCI = 0xb1
    valid_route = 0x92
    lateral_lane_position = 0xb8
    max_lateral_speed = 0xba
    lateral_gap = 0xbb
    lateral_alignment = 0xb9
    parameter = 0x7e
    action_step_length = 0x7d
    last_action_time = 0x7f
    stops = 0x74
    timeLoss = 0x8c
    loaded_list = 0x24
    teleporting_list = 0x25
    next_links = 0x33


class VehicleTypeEnum(Enum):
    """
    An Enum for vehicle types, for use with :meth:`road_network_gen`.
    See https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html
    for full details on vehicle types. This implementation only uses
    road vehicles and pedestrians.
    """

    private = 'private'
    emergency = 'emergency'
    authority = 'authority'
    army = 'army'
    vip = 'vip'
    pedestrian = 'pedestrian'
    passenger = 'passenger'
    hov = 'hov'  # High occupancy vehicle
    taxi = 'taxi'
    bus = 'bus'
    coach = 'coach'
    delivery = 'delivery'
    truck = 'truck'
    trailer = 'trailer'
    motorcycle = 'motorcycle'
    moped = 'moped'
    bicycle = 'bicycle'
    evehicle = 'evehicle'
    custom1 = 'custom1'
    custom2 = 'custom2'
