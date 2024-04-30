from datetime import datetime
import numpy as np
from datetime import datetime, timedelta
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.models.measurement.nonlinear import NonLinearGaussianMeasurement
from stonesoup.types.state import StateVector, StateVectors

class MyMeasurementModel(NonLinearGaussianMeasurement):
    """
    Cubic measurement, adapted from https://livrepository.liverpool.ac.uk/3015339/1/PL_smoothing_accepted1.pdf
    """

    def __init__(self, *args, **kwargs):
        """
        Ensure that the translation offset is initiated
        """
        super().__init__(*args, **kwargs)
        self.n = 3

    @property
    def ndim_meas(self) -> int:
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 2

    def function(self, state, noise=False, **kwargs) -> StateVector:
        r"""Model function :math:`h(\vec{x}_t,\vec{v}_t)`

        Parameters
        ----------
        state: :class:`~.State`
            An input state
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is added)

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_state`, 1)
            The model function evaluated given the provided time interval.
        """

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs()
            else:
                noise = 0

        # Cubic measurement
        x_meas = [state.state_vector[self.mapping[0]] ** self.n / 20]
        y_meas = [state.state_vector[self.mapping[1]] ** self.n / 20]

        return StateVectors([x_meas, y_meas]) + noise

def main():
    start_time = datetime.now().replace(microsecond=0)
    np.random.seed(1991)

    # Create ground truth

    q_x = 0.05
    q_y = 0.05
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                              ConstantVelocity(q_y)])
    timesteps = [start_time]
    truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=timesteps[0])])

    num_steps = 20
    for k in range(1, num_steps + 1):
        timesteps.append(start_time + timedelta(seconds=k))  # add next timestep to list of timesteps
        truth.append(GroundTruthState(
            transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
            timestamp=timesteps[k]))

    # Generate measurements

    from stonesoup.types.detection import Detection
    measurement_model = MyMeasurementModel(
        ndim_state=4,  # Number of state dimensions (position and velocity in 2D)
        mapping=(0, 2),  # Mapping measurement vector index to state index
        noise_covar=np.array([[0.0000000005, 0],  # Covariance matrix for Gaussian PDF
                              [0, 0.000000005]])
    )

    measurements = []
    for state in truth:
        measurement = measurement_model.function(state, noise=True)
        measurements.append(Detection(measurement,
                                      timestamp=state.timestamp,
                                      measurement_model=measurement_model))

    from stonesoup.types.hypothesis import SingleHypothesis
    from stonesoup.types.track import Track
    from stonesoup.updater.kalman import KalmanUpdater, UnscentedKalmanUpdater

    from stonesoup.predictor.kalman import KalmanPredictor
    from stonesoup.types.state import GaussianState

    predictor = KalmanPredictor(transition_model)
    updater_ukf = UnscentedKalmanUpdater()
    # updater_iplf = IPLFUpdater()

    prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

    # UKF filtering
    track_ukf = Track()
    for measurement in measurements:
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
        post = updater_ukf.update(hypothesis)
        track_ukf.append(post)
        prior = track_ukf[-1]

    # IPLF filtering
    track_iplf = Track()
    for measurement in measurements:
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
        post = updater_ukf.update(hypothesis)
        track_iplf.append(post)
        prior = track_iplf[-1]

    print()

if __name__ == "__main__":
    main()
