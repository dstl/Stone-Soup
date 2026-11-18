from abc import abstractmethod

from scipy.stats import multivariate_normal
import numpy as np

from ..base import Base, Property
from .kalman import KalmanUpdater
from ..types.update import GaussianMixtureUpdate
from ..types.state import TaggedWeightedGaussianState
from ..types.numeric import Probability
from ..types.detector_context import SimpleDetectorContext


class PointProcessUpdater(Base):
    r"""
    Base updater class for the implementation of any Gaussian Mixture (GM)
    point process derived multi target filters such as the
    Probability Hypothesis Density (PHD),
    Cardinalised Probability Hypothesis Density (CPHD) or
    Linear Complexity with Cumulants (LCC) filters
    """
    updater: KalmanUpdater = Property(
        doc="Underlying updater used to perform the \
             single target Kalman Update.")
    clutter_spatial_density: float = Property(
        default=1e-26,
        doc="Spatial density of the clutter process uniformly\
             distributed across the state space.")
    normalisation: bool = Property(
        default=True,
        doc="Flag for normalisation")
    prob_detection: Probability = Property(
        default=1,
        doc="Probability of a target being detected at the current timestep")
    prob_survival: Probability = Property(
        default=1,
        doc="Probability of a target surviving until the next timestep")

    def update(self, hypotheses, detector_context=None, **kwargs):
        """
        Updates the current components in a
        :class:`GaussianMixture` by applying the underlying \
        :class:`KalmanUpdater` updater to each component \
        with the supplied measurements.

        Parameters
        ==========
        hypotheses : list of :class:`MultipleHypothesis`
            Measurements obtained at time :math:`k+1`

        Returns
        =======
        updated_components : :class:`GaussianMixtureUpdate`
            GaussianMixtureMultiTargetTracker with updated \
            components at time :math:`k+1`
        """
        updated_components = list()
        weight_sum_list = list()
        detector_context = self._detector_context(detector_context)
        # Loop over all measurements
        for multi_hypothesis in hypotheses[:-1]:
            updated_measurement_components = list()
            # Initialise weight sum for measurement to clutter intensity
            weight_sum = 0
            # For every valid single hypothesis, update that component with
            # measurements and calculate new weight
            for hypothesis in multi_hypothesis:
                measurement_prediction = \
                    self.updater.predict_measurement(
                            hypothesis.prediction, hypothesis.measurement.measurement_model)
                measurement = hypothesis.measurement
                prediction = hypothesis.prediction
                # Calculate new weight and add to weight sum
                q = multivariate_normal.pdf(
                    measurement.state_vector.flatten(),
                    mean=measurement_prediction.mean.flatten(),
                    cov=measurement_prediction.covar
                )
                new_weight = detector_context.prob_detection(hypothesis)\
                    * prediction.weight * q
                if prediction.tag != 'birth':
                    new_weight *= self.prob_survival
                weight_sum += new_weight
                # Perform single target Kalman Update
                temp_updated_component = self.updater.update(hypothesis)
                updated_component = TaggedWeightedGaussianState(
                    tag=prediction.tag if prediction.tag != "birth" else None,
                    weight=new_weight,
                    state_vector=temp_updated_component.mean,
                    covar=temp_updated_component.covar,
                    timestamp=temp_updated_component.timestamp
                )
                # Add updated component to mixture
                updated_measurement_components.append((updated_component, hypothesis.measurement))
            weight_sum_list.append(weight_sum)
            for component, detection in updated_measurement_components:
                if self.normalisation:
                    component.weight /= \
                        (weight_sum + detector_context.clutter_spatial_density(detection))
                updated_components.append(component)

        # Calculate the correction terms
        l1 = self._calculate_update_terms(weight_sum_list, hypotheses, detector_context)

        for missed_detected_hypotheses in hypotheses[-1]:
            # Add all active components except birth component back into
            # mixture as miss detected components
            if missed_detected_hypotheses.prediction.tag != "birth":
                component = TaggedWeightedGaussianState(
                    tag=missed_detected_hypotheses.prediction.tag,
                    weight=missed_detected_hypotheses.prediction.weight
                    * (1-detector_context.prob_detection(missed_detected_hypotheses))
                    * l1 * self.prob_survival,
                    state_vector=missed_detected_hypotheses.prediction.mean,
                    covar=missed_detected_hypotheses.prediction.covar,
                    timestamp=missed_detected_hypotheses.prediction.timestamp)
                updated_components.append(component)

        # Ensure that no component has a weight of 0. This avoids divide
        # by 0 bugs in the GaussianMixtureReducer
        for component in updated_components:
            if component.weight == 0:
                component.weight = np.finfo(float).eps

        # Return updated components
        return GaussianMixtureUpdate(hypothesis=hypotheses,
                                     components=updated_components)

    def _detector_context(self, detector_context):
        if detector_context is not None:
            return detector_context
        return SimpleDetectorContext(
            prob_detection=self.prob_detection,
            clutter_spatial_density=self.clutter_spatial_density)

    @abstractmethod
    def _calculate_update_terms(self, updated_sum_list, hypotheses, detector_context):
        raise NotImplementedError


class PHDUpdater(PointProcessUpdater):
    """
    A implementation of the Gaussian Mixture
    Probability Hypothesis Density (GM-PHD) multi-target filter

    References
    ----------

    [1] B.-N. Vo and W.-K. Ma, “The Gaussian Mixture Probability Hypothesis
    Density Filter,” Signal Processing,IEEE Transactions on, vol. 54, no. 11,
    pp. 4091–4104, 2006. https://ieeexplore.ieee.org/document/1710358.

    [2] D. E. Clark, K. Panta and B. Vo, "The GM-PHD Filter Multiple Target Tracker," 2006 9th
    International Conference on Information Fusion, 2006, pp. 1-8, doi: 10.1109/ICIF.2006.301809.
    https://ieeexplore.ieee.org/document/4086095.
    """
    @staticmethod
    def _calculate_update_terms(updated_sum_list, hypotheses, detector_context):
        return 1


class LCCUpdater(PointProcessUpdater):
    """
    A implementation of the Gaussian Mixture
    Linear Complexity with Cumulants (GM-LCC) multi-target filter

    References
    ----------

    [1] D. E. Clark and F. De Melo. “A Linear-Complexity Second-Order
        Multi-Object Filter via Factorial Cumulants”.
        In: 2018 21st International Conference on
        Information Fusion (FUSION). 2018. DOI: 10.
        23919/ICIF.2018.8455331. https://ieeexplore.ieee.org/document/8455331..
    """
    mean_number_of_false_alarms: float = Property(
        default=1,
        doc="Mean number of false alarms (clutter) expected per timestep")
    variance_of_false_alarms: float = Property(
        default=1,
        doc="Variance on the number of false alarms (clutter) expected per timestep")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.second_order_cumulant = 0

    def _calculate_update_terms(self, updated_sum_list, hypotheses, detector_context):
        """
        Calculate the higher order terms used in the LCC Filter
        """
        # Get the predicted weight sum
        predicted_weight_sum =\
            Probability.sum(hypothesis.prediction.weight for hypothesis in
                            hypotheses[-1]) * self.prob_survival
        # Second order predicted cumulant c(2)
        predicted_c2 = self.second_order_cumulant * self.prob_survival**2
        # Detected predicted weight mu_d
        detected_weight_sum = Probability.sum(
            detector_context.prob_detection(hypothesis) * hypothesis.prediction.weight
            for hypothesis in hypotheses[-1]) * self.prob_survival
        # Detected predicted weight mu_phi
        misdetected_weight_sum = Probability.sum(
            (1-detector_context.prob_detection(hypothesis)) * hypothesis.prediction.weight
            for hypothesis in hypotheses[-1]) * self.prob_survival
        # Calculate the alpha of the predicted Panjer process
        alpha_pred = ((predicted_weight_sum +
                       self.mean_number_of_false_alarms)**2)\
            / (predicted_c2+self.second_order_false_alarm_cumulant+1e-26)
        # Calculate l1 and l2 correction factors
        denominator = alpha_pred + detected_weight_sum \
            + self.mean_number_of_false_alarms
        number_of_measurements = len(hypotheses)-1
        numerator = alpha_pred + number_of_measurements
        l1 = numerator/denominator
        l2 = numerator/(denominator**2)
        # Calculate updated c(2)
        detected_c2 = \
            Probability.sum(
                weight_sum/((weight_sum + detector_context.clutter_spatial_density(
                    multi_hypothesis[0].measurement)) ** 2)
                for weight_sum, multi_hypothesis in zip(updated_sum_list, hypotheses[:-1]))
        misdetected_c2 = (misdetected_weight_sum**2)*l2
        self.second_order_cumulant = misdetected_c2 - detected_c2
        # Return the l1 correction factor for miss detected weight update
        return np.float64(l1)

    @property
    def second_order_false_alarm_cumulant(self):
        return self.variance_of_false_alarms - self.mean_number_of_false_alarms
