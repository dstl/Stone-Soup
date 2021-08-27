from abc import abstractmethod
from typing import Set, Union

import numpy as np

from .actionable import Actionable
from .base import PlatformMountable
from ..types.detection import TrueDetection
from ..types.groundtruth import GroundTruthState


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

    @abstractmethod
    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> Set[TrueDetection]:
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
