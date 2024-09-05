import numpy as np
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from stonesoup.models.transition.linear import *
from stonesoup.platform.base import *
from stonesoup.sensor.radar.radar import *
from stonesoup.types.groundtruth import *
from stonesoup.types.state import *

stonesoup_yaml = YAML(typ=["rt", "stonesoup"], plug_ins=["stonesoup.serialise"])


def generate_default_yaml(
    scenario_config="ReinforcementLearning/configs/scenario_config.yaml",
):

    with open(scenario_config, "w") as file:
        data = CommentedMap()

        data["transition_model"] = CombinedLinearGaussianTransitionModel(
            model_list=[
                ConstantVelocity(noise_diff_coeff=0.0, seed=None),
                ConstantVelocity(noise_diff_coeff=0.0, seed=None),
            ],
            seed=None,
        )
        data["turn_model"] = KnownTurnRate(
            turn_noise_diff_coeffs=[0.0, 0.0], turn_rate=np.pi / 2, seed=None
        )
        data.yaml_set_comment_before_after_key(
            key="transition_model",
            before="""\nThis is the transition model,
it is responsible for determining how targets move from timestep to
timestep within StoneSoup. The model list containts 1D models for
determining how objects move along each axis (x,y) over time. These
are either ConstantVelocity models, or Constant Accelaration models.""",
        )
        data["targets"] = [
            GroundTruthPath(
                states=[
                    GroundTruthState(
                        state_vector=StateVector([[0], [0], [-1], [-1]]),
                        metadata={"type": "known"},
                    )
                ]
            ),
            GroundTruthPath(
                states=[
                    GroundTruthState(
                        state_vector=StateVector([[1], [1], [0], [-1]]),
                        metadata={"type": "known"},
                    )
                ]
            ),
            GroundTruthPath(
                states=[
                    GroundTruthState(
                        state_vector=StateVector([[-1.0], [1.0], [-1.0], [-0.5]]),
                        metadata={"type": "known"},
                    )
                ]
            ),
        ]
        data.yaml_set_comment_before_after_key(
            key="targets",
            before="""\nThese are the targets that the sensor manager will be responsible for
tracking. Each target is represented by a GroundTruthPath, which consisits of a list of
GroundTruthStates. The GroundTruthState is a representation of a targets position and direction.
They consist of a StateVector, with the stucture of [xposition, xdirection,
yposition, ydirection] The GroundTruthPath is a representation of the path of a particular
target between states and timesteps.""",
        )
        # Error when platform directions are 0, 0. Orientation of a zero-velocity moving platform is not defined
        data["sensor_manager"] = {
            "max_sensors_per_platform": 5,
            "platform": MovingPlatform(
                movement_controller=MovingMovable(
                    states=State([[0], [1], [1], [1]]),
                    position_mapping=(0, 2),
                    transition_model=CombinedLinearGaussianTransitionModel(
                        [
                            ConstantVelocity(0),
                            ConstantVelocity(0),
                        ]
                    ),
                    velocity_mapping=(1, 3),
                ),
                sensors=(
                    RadarRotatingBearingRange(
                        position_mapping=(0, 2),
                        noise_covar=np.diag([0.00, 0.00]),
                        dwell_centre=StateVector([0]),
                        rpm=60,
                        fov_angle=np.pi / 2,
                        rotation_offset=StateVector([0, 0, 0]),
                        mounting_offset=StateVector([0, 0]),
                        clutter_model=None,
                        ndim_state=2,
                        max_range=100,
                        resolution=0.017453292519943295,
                    ),
                    RadarRotatingBearingRange(
                        position_mapping=(0, 2),
                        noise_covar=np.diag([0.00, 0.00]),
                        dwell_centre=StateVector([0]),
                        rpm=[60],
                        fov_angle=np.pi / 2,
                        rotation_offset=StateVector([0, 0, 0]),
                        mounting_offset=StateVector([0, 1]),
                        clutter_model=None,
                        ndim_state=2,
                        max_range=100,
                        resolution=0.017453292519943295,
                    ),
                    RadarRotatingBearingRange(
                        position_mapping=(0, 2),
                        noise_covar=np.diag([0.00, 0.00]),
                        dwell_centre=StateVector([0]),
                        rpm=[60],
                        fov_angle=np.pi / 2,
                        rotation_offset=StateVector([0, 0, 0]),
                        mounting_offset=StateVector([0, 1]),
                        clutter_model=None,
                        ndim_state=2,
                        max_range=100,
                        resolution=0.017453292519943295,
                    ),
                ),
            ),
        }

        data.yaml_set_comment_before_after_key(
            key="sensor_manager",
            before="""\nThe sensor manager is responsible for controlling the platforms
and their sensors. Moving platforms have movement_controllers that can use to move the
platform. They consist of a states list that has the position and directional values for
each dimension. They have the format [x_position, x_direction, y_position, y_direction]
with additional pairs depending on the dimensions the platform is opeating in. The
velocity mappng and position mapping, specify which index values in states represents
what values. Finally, there is the transition model, which much like the one for targets,
is used for describing how the platform moves through each dimension.
Consititing of the same COnstantVelocity or ConstantAcceleration objects.
The MovingPlatform also has the sensors attached to it in a tupe value.
A common sensor being the RadarRotatingBearingRange sensor, which has its
own parameters for defining its properties. Note: fov_angle should be in radians NOT degrees.


Note: The maximum number of sensors is currently fixed at 5.""",
        )

        data["unknown_targets"] = 2

        data.yaml_set_comment_before_after_key(
            key="unknown_targets",
            before="""\nThese are the no of unknown targets present in the scenario.""",
        )

        data["episode_threshold"] = 400

        data.yaml_set_comment_before_after_key(
            key="episode_threshold",
            before="""\nThese are the no of episode threshold""",
        )

        data["scenario_dimensions"] = [500, 500]

        data.yaml_set_comment_before_after_key(
            key="scenario_dimensions",
            before="""\nThis is the range of the scenario dimensions, format [x,y].
Note that the [0,0] will always be the centre of the scenario.
This means the [400,400] represents ranges -200 to -200 in both
dimensions.""",
        )

        data["velocity_limit"] = 10

        data.yaml_set_comment_before_after_key(
            key="velocity_limit",
            before="""\nThis is the maximum velocity that any object can travel.""",
        )

        data["actionable_sensors"] = True

        data.yaml_set_comment_before_after_key(
            key="actionable_sensors",
            before="""\nThis indicates if sensors are actionable. If som the actions
will be specified in sensor_level_actions.""",
        )

        data["sensor_level_actions"] = [
            {"dwell_centre": [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]},
            {"dwell_centre": [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]},
        ]

        data.yaml_set_comment_before_after_key(
            key="sensor_level_actions",
            before="""\nThis is a list of dictionaries containing sensor property names as keys,
linking to a list of discrete values that the property could be. Each disctionary represents a sensor.
The first dictionary in the list represents the first sensor, the second represents the second sensor,
etc. You don't have to add a dictionary for each sensor. If there are more sensors than dictionaries, then
only the first n sensors will have actionable sensors, where n is the number of dictionaries in the list.
Note: Do not add more dictionaries than there are sensors as this will result in an indexing error.""",
        )

        stonesoup_yaml.dump(data, file)


if __name__ == "__main__":

    generate_default_yaml()
