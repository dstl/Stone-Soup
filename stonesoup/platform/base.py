from typing import MutableSequence

from stonesoup.base import Property, Base
from stonesoup.movable import Movable, FixedMovable, MovingMovable, MultiTransitionMovable
from stonesoup.sensor.sensor import Sensor


class Platform(Base):
    """A platform that can carry a number of different sensors.

    The location of platform mounted sensors will be maintained relative to
    the sensor position. Platforms move within a 2 or 3 dimensional
    rectangular cartesian space.

    A simple platform is considered to always be aligned with its principle
    velocity. It does not take into account issues such as bank angle or body
    deformation (e.g. flex).

    Movement is controlled by the Platform's :attr:`Platform.movement_controller`, and access to
    attributes of the Platform is proxied to the movement controller, to allow the Platform to
    report it's own position, orientation etc.

    If a ``movement_controller`` argument is not supplied to the constructor, the Platform will
    try to construct one using unused arguments passed to the Platform's constructor.

    .. note:: This class is abstract and not intended to be instantiated. To get the behaviour
        of this class use a subclass which gives movement
        behaviours. Currently these are :class:`~.FixedPlatform` and
        :class:`~.MovingPlatform`
    """

    movement_controller: Movable = Property(
        default=None,
        doc=":class:`~.Movable` object to control the Platform's movement. Default is None, but "
            "it can be constructed transparently by passing Movable's constructor parameters to "
            "the Platform constructor.")
    sensors: MutableSequence[Sensor] = Property(
        default=None, readonly=True,
        doc="A list of N mounted sensors. Defaults to an empty list")

    _default_movable_class = None  # Will be overridden by subclasses

    def __getattribute__(self, name):
        # This method is called if we try to access an attribute of self. First we try to get the
        # attribute directly, but if that fails, we want to try getting the same attribute from
        # self.state instead. If that, in turn,  fails we want to return the error message that
        # would have originally been raised, rather than an error message that the State has no
        # such attribute.
        #
        # An alternative mechanism using __getattr__ seems simpler (as it skips the first few lines
        # of code, but __getattr__ has no mechanism to capture the originally raised error.
        try:
            # This tries first to get the attribute from self.
            return Base.__getattribute__(self, name)
        except AttributeError as original_error:
            if name.startswith("_"):
                # Don't proxy special/private attributes to `movement_controller`, just raise the
                # original error
                raise original_error
            else:
                # For non _ attributes, try to get the attribute from self.state instead of self.
                try:
                    controller = Base.__getattribute__(self, 'movement_controller')
                    return getattr(controller, name)
                except AttributeError:
                    # If we get the error about 'Movable' not having the attribute, then we want to
                    # raise the original error instead
                    raise original_error

    def __init__(self, *args, **kwargs):
        platform_arg_names = self._properties.keys()
        platform_args = {key: value for key, value in kwargs.items() if key in platform_arg_names}
        other_args = {key: value for key, value in kwargs.items() if key not in platform_arg_names}
        super().__init__(**platform_args)
        if self.movement_controller is None:
            self.movement_controller = self._default_movable_class(*args, **other_args)
        if self.sensors is None:
            self._property_sensors = []
        for sensor in self.sensors:
            sensor.movement_controller = self.movement_controller

    @staticmethod
    def _tuple_or_none(value):
        return None if value is None else tuple(value)

    @sensors.getter
    def sensors(self):
        return self._tuple_or_none(self._property_sensors)

    def add_sensor(self, sensor: Sensor) -> None:
        """ Add a sensor to the platform

        Parameters
        ----------
        sensor : :class:`~.BaseSensor`
            The sensor object to add
        """
        self._property_sensors.append(sensor)
        sensor.movement_controller = self.movement_controller

    def remove_sensor(self, sensor: Sensor) -> None:
        """ Remove a sensor from the platform

        Parameters
        ----------
        sensor : :class:`~.BaseSensor`
            The sensor object to remove
        """
        self.pop_sensor(self._property_sensors.index(sensor))

    def pop_sensor(self, index: int = -1):
        """
        Remove and return a sensor from the platform by index. If no index is specified, remove
        and return the last sensor in :attr:`self.sensors`

        Parameters
        ----------
        index : int
            The index of the sensor to remove. Defaults to the last item in the list.
        """
        return self._property_sensors.pop(index)

    # The position, orientation and velocity getters are not required, as __getattribute__ will do
    # the job, but the setters are required, and this seems the cleanest way to implement them
    @property
    def position(self):
        return self.movement_controller.position

    @position.setter
    def position(self, value):
        self.movement_controller.position = value

    @property
    def velocity(self):
        return self.movement_controller.velocity

    @velocity.setter
    def velocity(self, value):
        self.movement_controller.velocity = value

    @property
    def orientation(self):
        return self.movement_controller.orientation

    @orientation.setter
    def orientation(self, value):
        self.movement_controller.orientation = value

    def __getitem__(self, item):
        return self.movement_controller.__getitem__(item)


class FixedPlatform(Platform):
    _default_movable_class = FixedMovable
    pass


class MovingPlatform(Platform):
    _default_movable_class = MovingMovable
    pass


class MultiTransitionMovingPlatform(Platform):
    _default_movable_class = MultiTransitionMovable
    pass
