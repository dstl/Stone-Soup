from .base import DataAssociator
from ..base import Property
from ..hypothesiser import Hypothesiser
from ..hypothesiser.probability import PDAHypothesiser
from ..types.detection import MissedDetection
from ..types.hypothesis import (
    SingleProbabilityHypothesis, ProbabilityJointHypothesis)
from ..types.multihypothesis import MultipleHypothesis
from ..types.numeric import Probability
import itertools
import numpy as np
import math


class PDA(DataAssociator):
    """Probabilistic Data Association (PDA)

    Given a set of detections and a set of tracks, each track has a
    probability that it is associated to each specific detection.
    """

    hypothesiser: Hypothesiser = Property(
        doc="Generate a set of hypotheses for each prediction-detection pair")

    def associate(self, tracks, detections, timestamp, **kwargs):

        # Generate a set of hypotheses for each track on each detection
        hypotheses = self.generate_hypotheses(tracks, detections, timestamp, **kwargs)

        # Ensure association probabilities are normalised
        for track, hypothesis in hypotheses.items():
            hypothesis.normalise_probabilities(total_weight=1)

        return hypotheses


class JPDA(DataAssociator):
    r"""Joint Probabilistic Data Association (JPDA)

    Given a set of Detections and a set of Tracks, each Detection has a
    probability that it is associated with each specific Track. Rather than
    associate specific Detections/Tracks, JPDA calculates the new state of a
    Track based on its possible association with ALL Detections.  The new
    state is a Gaussian Mixture, reduced to a single Gaussian.
    If

    .. math::

          prob_{association(Detection, Track)} <
          \frac{prob_{association(MissedDetection, Track)}}{gate\ ratio}

    then Detection is assumed to be outside Track's gate, and the probability
    of association is dropped from the Gaussian Mixture.  This calculation
    takes place in the function :meth:`enumerate_JPDA_hypotheses`.
    """

    hypothesiser: PDAHypothesiser = Property(
        doc="Generate a set of hypotheses for each prediction-detection pair")

    def associate(self, tracks, detections, timestamp, **kwargs):

        # Calculate MultipleHypothesis for each Track over all
        # available Detections
        hypotheses = self.generate_hypotheses(tracks, detections, timestamp, **kwargs)

        # enumerate the Joint Hypotheses of track/detection associations
        joint_hypotheses = \
            self.enumerate_JPDA_hypotheses(tracks, hypotheses)

        # Calculate MultiMeasurementHypothesis for each Track over all
        # available Detections with probabilities drawn from JointHypotheses
        new_hypotheses = dict()

        for track in tracks:

            single_measurement_hypotheses = list()

            # record the MissedDetection hypothesis for this track
            prob_misdetect = Probability.sum(
                joint_hypothesis.probability
                for joint_hypothesis in joint_hypotheses
                if not joint_hypothesis.hypotheses[track].measurement)

            single_measurement_hypotheses.append(
                SingleProbabilityHypothesis(
                    hypotheses[track][0].prediction,
                    MissedDetection(timestamp=timestamp),
                    measurement_prediction=hypotheses[track][0].measurement_prediction,
                    probability=prob_misdetect))

            # record hypothesis for any given Detection being associated with
            # this track
            for hypothesis in hypotheses[track]:
                if not hypothesis:
                    continue
                pro_detect_assoc = Probability.sum(
                    joint_hypothesis.probability
                    for joint_hypothesis in joint_hypotheses
                    if joint_hypothesis.hypotheses[track].measurement is hypothesis.measurement)

                single_measurement_hypotheses.append(
                    SingleProbabilityHypothesis(
                        hypothesis.prediction,
                        hypothesis.measurement,
                        measurement_prediction=hypothesis.measurement_prediction,
                        probability=pro_detect_assoc))

            result = MultipleHypothesis(single_measurement_hypotheses, True, 1)

            new_hypotheses[track] = result

        return new_hypotheses

    @classmethod
    def enumerate_JPDA_hypotheses(cls, tracks, multihypths):

        joint_hypotheses = list()

        if not tracks:
            return joint_hypotheses

        # perform a simple level of gating - all track/detection pairs for
        # which the probability of association is a certain multiple less
        # than the probability of missed detection - detection is outside the
        # gating region, association is impossible
        possible_assoc = list()

        for track in tracks:
            track_possible_assoc = list()
            for hypothesis in multihypths[track]:
                # Always include missed detection (gate ratio < 1)
                track_possible_assoc.append(hypothesis)
            possible_assoc.append(track_possible_assoc)

        # enumerate all valid JPDA joint hypotheses
        enum_JPDA_hypotheses = (
            joint_hypothesis
            for joint_hypothesis in itertools.product(*possible_assoc)
            if cls.isvalid(joint_hypothesis))

        # turn the valid JPDA joint hypotheses into 'JointHypothesis'
        for joint_hypothesis in enum_JPDA_hypotheses:
            local_hypotheses = {}

            for track, hypothesis in zip(tracks, joint_hypothesis):
                local_hypotheses[track] = \
                    multihypths[track][hypothesis.measurement]

            joint_hypotheses.append(
                ProbabilityJointHypothesis(local_hypotheses))

        # normalize ProbabilityJointHypotheses relative to each other
        sum_probabilities = Probability.sum(hypothesis.probability
                                            for hypothesis in joint_hypotheses)
        for hypothesis in joint_hypotheses:
            hypothesis.probability /= sum_probabilities

        return joint_hypotheses

    @staticmethod
    def isvalid(joint_hypothesis):

        # 'joint_hypothesis' represents a valid joint hypothesis if
        # no measurement is repeated (ignoring missed detections)

        measurements = set()
        for hypothesis in joint_hypothesis:
            measurement = hypothesis.measurement
            if not measurement:
                pass
            elif measurement in measurements:
                return False
            else:
                measurements.add(measurement)

        return True


class JPDAwithLBP(JPDA):
    """ Joint Probabilistic Data Association with Loopy Belief Propagation

    This is a faster alternative of the standard :class:`~.JPDA` algorithm, which makes use of
    Loopy Belief Propagation (LBP) to efficiently approximately compute the marginal association
    probabilities of tracks to measurements. See Williams and Lau (2014) for further details.

    Reference
    ----------
    Jason L. Williams and Rosalyn A. Lau, Approximate evaluation of marginal association
    probabilities with belief propagation, IEEE Transactions on Aeroapce and Electronic Systems,
    vol 50(4), pp. 2942-2959, 2014.
    """

    def associate(self, tracks, detections, timestamp, **kwargs):
        """Associate tracks and detections

        Parameters
        ----------
        tracks : set of :class:`stonesoup.types.track.Track`
            Tracks which detections will be associated to.
        detections : set of :class:`stonesoup.types.detection.Detection`
            Detections to be associated to tracks.
        timestamp : :class:`datetime.datetime`
            Timestamp to be used for missed detections and to predict to.

        Returns
        -------
        : mapping of :class:`stonesoup.types.track.Track` : :class:`stonesoup.types.hypothesis.Hypothesis`
            Mapping of track to Hypothesis
        """

        # Calculate MultipleHypothesis for each Track over all
        # available Detections
        hypotheses = {
            track: self.hypothesiser.hypothesise(track, detections, timestamp)
            for track in tracks}

        multi_hypotheses, assoc_prob_list = self._compute_multi_hypotheses(tracks, detections, hypotheses, timestamp)
        return multi_hypotheses, assoc_prob_list

    @staticmethod
    def _calc_likelihood_matrix(tracks, detections, hypotheses):
        """ Compute the likelihood matrix (i.e. single target association weights)

        Parameters
        ----------
        tracks: list of :class:`stonesoup.types.track.Track`
            Current tracked objects
        detections : list of :class:`stonesoup.types.detection.Detection`
            Retrieved measurements
        hypotheses: dict
            Key value pairs of tracks with associated detections

        Returns
        -------
        :class:`numpy.ndarray`
            An indicator matrix of shape (num_tracks, num_detections + 1) indicating the possible
            (aka. valid) associations between tracks and detections. The first column corresponds
            to the null hypothesis (hence contains all ones).
        :class:`numpy.ndarray`
            A matrix of shape (num_tracks, num_detections + 1) containing the unnormalised
            likelihoods for all combinations of tracks and detections. The first column corresponds
            to the null hypothesis.
        """

        # Ensure tracks and detections are lists (not sets)
        tracks, detections = list(tracks), list(detections)

        # Construct validation and likelihood matrices
        # Both matrices have shape (num_tracks, num_detections + 1), where the first column
        # corresponds to the null hypothesis.
        num_tracks, num_detections = len(tracks), len(detections)
        likelihood_matrix = np.zeros((num_tracks, num_detections + 1))
        for i, track in enumerate(tracks):
            for hyp in hypotheses[track]:
                if not hyp:
                    likelihood_matrix[i, 0] = hyp.weight
                else:
                    j = next(d_i for d_i, detection in enumerate(detections)
                             if hyp.measurement == detection)
                    likelihood_matrix[i, j + 1] = hyp.weight

        # change the normalisation of the likelihood matrix to have the no measurement
        # association hypothesis normalised to unity
        for i in range(num_tracks):
            for j in range(1, num_detections + 1):
                likelihood_matrix[i, j] = likelihood_matrix[i, j] / likelihood_matrix[i, 0]
            likelihood_matrix[i, 0] = 1.0

        return likelihood_matrix.astype(float)

    """
    Perform loopy belief propagation (Williams and Lau, 2014) to determine the approximate 
    marginal association probabilities (of tracks to measurements). This requires:
    1. likelihood_matrix = single target association weights
    2. n_iterations = number of iterations between convergence checks
    3. delta = deviation tolerance(of approximate weights from true weights)
    """

    @staticmethod
    def _loopy_belief_propagation(likelihood_matrix, n_iterations, delta):

        # number of tracks
        num_tracks = len(likelihood_matrix)

        # number of measurements/detections
        # num_measurements = len(likelihood_matrix[0]) - 1
        num_measurements = len(likelihood_matrix.transpose()) - 1

        # initialise
        iteration: int = 0
        alpha: float = 1.0
        d: float = 0.0

        # allocate memory
        nu = np.zeros((num_tracks, num_measurements))
        nu_tilde = np.zeros((num_tracks, num_measurements))
        mu = np.zeros((num_tracks, num_measurements))
        assoc_prob_matrix = np.zeros((num_tracks, num_measurements + 1))

        # determine W_star
        w_star: float = 0
        for i in range(num_tracks):
            # sum over j > 0
            sum_total = 0.0
            for j in range(1, num_measurements + 1):
                sum_total += likelihood_matrix[i][j]
            # return maximum over all i values
            if sum_total > w_star:
                w_star = sum_total

        # initialise
        for i in range(num_tracks):
            for j in range(num_measurements):
                nu[i][j] = 1

        # loopy belief propagation
        while iteration == 0 or (alpha * d) / (1 - alpha) >= 0.5 * math.log10(1 + delta):

            for k in range(1, n_iterations + 1):

                # increment the number of iterations
                iteration += 1

                # calculate L-R message
                for i in range(num_tracks):
                    for j in range(1, num_measurements + 1):
                        # calculate s
                        s = 1
                        for jj in range (1, num_measurements + 1):
                            if jj != j:
                                s += (likelihood_matrix[i][jj] * nu[i][jj - 1])
                        # calculate mu[i][j]
                        mu[i][j - 1] = likelihood_matrix[i][j] / s

                # save values for convergence check
                if k == n_iterations:
                    for i in range(num_tracks):
                        for j in range(1, num_measurements + 1):
                            nu_tilde[i][j - 1] = nu[i][j - 1]

                # calculate R-L messages
                for j in range(1, num_measurements + 1):
                    for i in range(num_tracks):
                        # calculate s
                        s: float = 1.0
                        for ii in range(num_tracks):
                            if ii != i:
                                s += mu[ii][j - 1]
                        # update nu
                        nu[i][j - 1] = 1 / s

            # check for convergence
            d = 0.0
            for i in range(num_tracks):
                for j in range(1, num_measurements + 1):
                    if d < abs(math.log10(nu[i][j - 1] / nu_tilde[i][j - 1])):
                        d = abs(math.log10(nu[i][j - 1] / nu_tilde[i][j - 1]))

            # determine alpha
            if d > 0:
                alpha = math.log10((1 + w_star * d) / (1 + w_star))
                alpha = (alpha / math.log10(d))
            else:
                alpha = 0.0

            # if w_star has a very large value, alpha = 1 which causes division by zero in the convergence check
            # therefore, set alpha to be a nominal value just short of unity
            if alpha == 1:
                alpha = (1 - 1e-10)

        # calculate marginal probabilities (beliefs)
        for i in range(num_tracks):

            # calculate s
            s = 1
            for j in range(1, num_measurements + 1):
                s += (likelihood_matrix[i][j] * nu[i][j - 1])

            # calculate association probabilities
            assoc_prob_matrix[i][0] = (1 / s)
            for j in range(1, num_measurements + 1):
                assoc_prob_matrix[i][j] = (likelihood_matrix[i][j] * nu[i][j - 1]) / s

        # return the matrix of marginal association probabilities
        return assoc_prob_matrix.astype(float)

    @classmethod
    def _compute_multi_hypotheses(cls, tracks, detections, hypotheses, time):

        # Tracks and detections must be in a list, so we can keep track of their order
        track_list = list(tracks)
        detection_list = list(detections)

        # calculate the single target association weights
        likelihood_matrix = cls._calc_likelihood_matrix(tracks, detections, hypotheses)
        # print(likelihood_matrix)

        # Run Loopy Belief Propagation to determine the marginal association probability matrix
        n_iterations: int = 1
        delta: float = 0.001
        assoc_prob_matrix = cls._loopy_belief_propagation(likelihood_matrix, n_iterations, delta)
        # print(assoc_prob_matrix)


        # Extract the class names of the detection list elements
        assoc_prob_matrix_labels = [type(detection).__name__ for detection in detection_list]

        # Combine the association probability matrix with the labels
        assoc_prob_list = [assoc_prob_matrix, assoc_prob_matrix_labels]

        # Calculate MultiMeasurementHypothesis for each Track over all
        # available Detections with probabilities drawn from the association matrix
        new_hypotheses = dict()

        for i, track in enumerate(track_list):

            single_measurement_hypotheses = list()

            # Null measurement hypothesis
            null_hypothesis = next((hyp for hyp in hypotheses[track] if not hyp), None)
            prob_misdetect = Probability(assoc_prob_matrix[i, 0])
            single_measurement_hypotheses.append(
                SingleProbabilityHypothesis(
                    null_hypothesis.prediction,
                    MissedDetection(timestamp=time),
                    measurement_prediction=null_hypothesis.measurement_prediction,
                    probability=prob_misdetect))

            # True hypotheses
            for hypothesis in hypotheses[track]:
                if not hypothesis:
                    continue

                # Get the detection index
                j = next(d_i + 1 for d_i, detection in enumerate(detection_list)
                         if hypothesis.measurement == detection)

                pro_detect_assoc = Probability(assoc_prob_matrix[i, j])
                single_measurement_hypotheses.append(
                    SingleProbabilityHypothesis(
                        hypothesis.prediction,
                        hypothesis.measurement,
                        measurement_prediction=hypothesis.measurement_prediction,
                        probability=pro_detect_assoc))

            new_hypotheses[track] = MultipleHypothesis(single_measurement_hypotheses, True, 1)

        return new_hypotheses, assoc_prob_list



def extract_assoc_prob_matrix(assoc_prob_matrix, assoc_prob_matrix_labels):
    """
    Extracts the probability association values with their measurement label.

    Parameters:
    - assoc_prob_matrix: List of 2D numpy arrays with association probabilities.
    - assoc_prob_matrix_labels: List of label arrays corresponding to the association probabilities.

    Returns:
    - A dictionary with 'Detection' and 'Clutter' keys, each containing a dictionary with a single key 0
      and numpy array of the association probabilities.
    """
    # Extract the probability association values with their measurement label
    assoc_prob_matrix = [arr[:, 1:] for arr in assoc_prob_matrix]
    assoc_prob_matrix_labels = [arrl[:] for arrl in assoc_prob_matrix_labels]

    assoc_prob_matrix = [np.max(arr, axis=0).reshape(1, -1) for arr in assoc_prob_matrix]

    # Initialize the result dictionary
    assoc_prob_matrix_final = {
        'Detection': [],
        'Clutter': []
    }

    # Process each pair of ndarray and label list
    for val_ndarray, lab_list in zip(assoc_prob_matrix, assoc_prob_matrix_labels):
        for i, label in enumerate(lab_list):
            column_values = val_ndarray[:, i]
            if label == 'TrueDetection':
                assoc_prob_matrix_final['Detection'].extend(column_values)
            elif label == 'Clutter':
                assoc_prob_matrix_final['Clutter'].extend(column_values)

    # Convert lists to NumPy arrays
    assoc_prob_matrix_final['Detection'] = np.array(assoc_prob_matrix_final['Detection'])
    assoc_prob_matrix_final['Clutter'] = np.array(assoc_prob_matrix_final['Clutter'])

    # Convert to dictionary format specified in the problem statement
    assoc_prob_matrix_final = {
        'Detection': {0: assoc_prob_matrix_final['Detection']},
        'Clutter': {0: assoc_prob_matrix_final['Clutter']}
    }

    return assoc_prob_matrix_final