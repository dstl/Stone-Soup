import numpy as np
import datetime
import os
import sys
from typing import Collection, Sequence
from enum import Enum

from .base import GroundTruthReader
from ..base import Property
from ..types.array import StateVector
from ..types.groundtruth import GroundTruthState, GroundTruthPath
from ..buffered_generator import BufferedGenerator


class SUMOGroundTruthReader(GroundTruthReader):
    r"""A Groundtruth reader for a SUMO simulation.

    At each time step, kinematic information from the objects in the SUMO simulation will be
    extracted and placed into a :class:`~.GroundTruthState`. States with the same ID will be placed
    into a :class:`~.GroundTruthPath` in sequence.

    The state vector for each truth object is, by default,  of the form:

    .. math::

        [\mathbf{x}, \mathbf{v}_x, \mathbf{y}, \mathbf{v}_y, \mathbf{z}]

    .. note::

        By default, the Carteisan co-ordinates use the UTM-projection with the origin shifted such that the bottom left
        corner of the network is the origin (0,0). See: https://sumo.dlr.de/docs/Geo-Coordinates.html

        To extract lat/long co-ordinates, it is required that the network is geo-referenced.

        This reader requires the installation of SUMO, see:
        https://sumo.dlr.de/docs/Installing/index.html

        This reader requires a SUMO configuration file.

    Parameters
    ----------
    """
    sumo_cfg_file_path: str = Property(
        doc='Path to SUMO config file')

    sumo_server_path: str = Property(
        doc='Path to SUMO server, relative from SUMO_HOME environment variable. "/bin/sumo" to run '
            'on command line "/bin/sumo-gui" will run using the SUMO-GUI, this will require '
            'pressing play within the GUI.')

    sim_start: datetime.datetime = Property(
        default=None,
        doc='Start time for the simulation. Will default to datetime.datetime.now()')

    sim_steps: int = Property(
        default=200,
        doc='Number of steps you want your SUMO simulation to run for. Use numpy.inf to have no'
            ' limit')

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
        doc='Collection of metadata fields for vehicles that will be added to the metadata of each '
            'GroundTruthState. Possible fields are documented in '
            'https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html. See also '
            'VehicleMetadataEnum. Underscores are required in place of spaces. '
            'An example would be: ["speed", "acceleration", "lane_position"]')

    geographic_coordinates: bool = Property(
        default=False,
        doc='If True, geographic co-ordinates (longitude, latitude) will be added to the metadata '
            ' of each state, as well as the lat/long of the origin of hte local co-ordinate frame')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If network is geo-referenced, and geographic_coordinates == True, then self.origin will be lat/long
        # position of the origin of the local Cartesian frame.
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

    @staticmethod
    def calculate_velocity(speed, angle, radians=False):
        if not radians:
            v_x, v_y = speed * np.sin(np.radians(angle)), speed * np.cos(np.radians(angle))
        elif radians:
            v_x, v_y = speed * np.sin(angle), speed * np.cos(angle)
        return v_x, v_y

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self):

        # Add SUMO_HOME env variable. Use this for server/config-file paths
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
            import traci
        else:
            sys.exit("Declare environment variable 'SUMO_HOME'")

        traci.start(self.sumoCmd)
        if self.geographic_coordinates:
            self.origin = traci.simulation.convertGeo(0, 0)   # long, lat

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
                    long, lat = traci.simulation.convertGeo(*state_vector[self.position_mapping, :])
                    metadata['longitude'] = long
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
                    traci.vehicle.subscribe(id_, tuple(data.value for data in VehicleMetadataEnum if
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
                    long, lat = traci.simulation.convertGeo(*state_vector[self.position_mapping, :])
                    metadata['longitude'] = long
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


class PersonMetadataEnum(Enum):
    """
    A metadata Enum used to map the named variable of the person to the relevant id. Subscribing to
    this id will retrieve the value and add it to the metadata of the GroundTruthState.
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
    A metadata Enum used to map the named variable of the vehicle to the relevant id. Subscribing to
    this id will retrieve the value and add it to the metadata of the GroundTruthState.
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
