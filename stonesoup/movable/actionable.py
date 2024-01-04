from abc import ABC

from ..sensormanager.actionable import Actionable


class MovableActionable(Actionable):
    """Movable Actionable base class

    Base class for actionable sensors which has all the functionality of base
    :class:`~.Actionable` class, with additional platform-specific features.
    """

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
