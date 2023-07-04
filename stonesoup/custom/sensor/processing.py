import datetime
import random
from queue import Queue
from typing import Set, Union
from uuid import UUID

import numpy as np

from stonesoup.base import Property
from stonesoup.custom.sensor.action.processing import ProcessingActionInput, \
    ProcessingActionGenerator, ProcessingJobState
from stonesoup.models.clutter import ClutterModel
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.sensor.actionable import ActionableProperty
from stonesoup.sensor.sensor import Sensor
from stonesoup.types.array import CovarianceMatrix
from stonesoup.types.detection import TrueDetection
from stonesoup.types.groundtruth import GroundTruthState


class ProcessingNode(Sensor):
    ndim_state: int = Property(
        doc="Number of state dimensions. This is utilised by (and follows in\
                       format) the underlying :class:`~.CartesianToElevationBearing`\
                       model")
    mapping: np.ndarray = Property(
        doc="Mapping between the targets state space and the sensors\
                       measurement capability")
    noise_covar: CovarianceMatrix = Property(
        doc="The sensor noise covariance matrix. This is utilised by\
                       (and follow in format) the underlying \
                       :class:`~.LinearGaussian` model")
    current_job_id: UUID = ActionableProperty(
        doc="The current job id",
        default=None,
        generator_cls=ProcessingActionGenerator
    )
    clutter_model: ClutterModel = Property(
        default=None,
        doc="An optional clutter generator that adds a set of simulated "
            ":class:`Clutter` objects to the measurements at each time step. "
            "The clutter is simulated according to the provided distribution.")

    def __init__(self, *args, **kwargs):
        self._current_job = None
        super().__init__(*args, **kwargs)
        self._job_queue = []

    @current_job_id.setter
    def current_job_id(self, value):
        self._property_current_job_id = value
        if self.current_job is not None:
            self._current_job = next((job for job in self.job_queue if job.id == value), None)
        else:
            self._current_job = None

    @property
    def job_queue(self):
        return self._job_queue

    @property
    def current_job(self):
        return self._current_job

    @property
    def measurement_model(self):
        return LinearGaussian(
            ndim_state=self.ndim_state,
            mapping=self.mapping,
            noise_covar=self.noise_covar)

    def act(self, timestamp: datetime.datetime):
        if not self.validate_timestamp():
            self.timestamp = timestamp
            return

        if self.current_job is None or self.current_job.state == ProcessingJobState.COMPLETED:
            self._job_queue.remove(self.current_job)
            self.current_job_id = None
        elif self.current_job is not None:
            if self.current_job.state == ProcessingJobState.RUNNING \
                    and timestamp >= self.current_job.end_time:
                self.current_job.state = ProcessingJobState.COMPLETED
        self.timestamp = timestamp


    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> Set[TrueDetection]:
        detections = set()
        measurement_model = self.measurement_model

        if self.current_job is not None and self.current_job.state == ProcessingJobState.COMPLETED:
            for ground_truth in ground_truths:
                if random.random() < self.current_job.probability_of_detection:
                    measurement = measurement_model.function(ground_truth.state_vector, noise, **kwargs)
                    detection = TrueDetection(state_vector=measurement,
                                              groundtruth_path=ground_truth,
                                              sensor_state=self)
                    detections.add(detection)
            if self.clutter_model is not None:



