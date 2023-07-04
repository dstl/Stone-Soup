from enum import Enum
from uuid import UUID

import numpy as np
import scipy
from scipy.stats import mvn, norm

from stonesoup.base import Base, Property
from stonesoup.sensor.action import Action, ActionGenerator
from stonesoup.types.state import GaussianState


class ProcessOutput(Base):
    run_time: float = Property(doc="Processing time in seconds")


class ProcessingJobState(Enum):
    PENDING = 0
    RUNNING = 1
    COMPLETED = 2


class ProcessingJob(Base):
    id: UUID = Property(doc="Unique identifier for the job")
    algorithm: str = Property(doc="Algorithm to be used")
    probability_of_detection: float = Property(doc="Probability of detection")
    clutter_density: float = Property(doc="Clutter density")
    processing_time: GaussianState = Property(doc="Processing time in seconds")
    state: ProcessingJobState = Property(doc="State of the job",
                                         default=ProcessingJobState.PENDING)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_time = None
        self._end_time = None

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    def start(self, timestamp):
        self.state = ProcessingJobState.RUNNING
        self._start_time = timestamp
        noise = np.max([0, norm.rvs(self.processing_time.mean, np.sqrt(self.processing_time.covar))])
        self._end_time = timestamp + noise


class ProcessingAction(Action):
    target_value: UUID = Property(doc="Target value.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._end_time = mvn.rvs(self.processing_time.mean, self.processing_time.covar)

    @property
    def end_time(self):
        return self._end_time

    def act(self, current_time, timestamp, init_value):
        if current_time >= self.end_time:
            return True
        else:
            return False


class ProcessingActionGenerator(ActionGenerator):
    """Generates possible actions for processing data in a given time period."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def default_action(self):
        return ProcessingAction(generator=self,
                                end_time=self.end_time,
                                target_value=True)

    def __call__(self, resolution=None, epsilon=None):
        """
        Parameters
        ----------
        resolution: float
            Resolution of the action space
        epsilon: float
            Probability of taking a random action

        Returns
        -------
        Iterator[Action]
            Iterator of actions
        """
        if resolution is None:
            resolution = self.resolution
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.random() < epsilon:
            yield self.default_action
        else:
            for value in np.arange(self.limits[0], self.limits[1], resolution):
                yield self._action_cls(generator=self,
                                       end_time=self.end_time,
                                       target_value=value)

