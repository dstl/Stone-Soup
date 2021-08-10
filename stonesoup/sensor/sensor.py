from abc import abstractmethod, ABC
from typing import Set, Union

import numpy as np

from stonesoup.types.detection import TrueDetection
from stonesoup.types.groundtruth import GroundTruthState
from .base import PlatformMountable


class Sensor(PlatformMountable, ABC):
    """Sensor Base class for general use.

    Most properties and methods are inherited from :class:`~.PlatformMountable`.

    Sensors must have a measure function.
    """

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
