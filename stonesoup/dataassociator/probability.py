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

from ._ehm import EHMTree, TrackClusterer


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
    probabilities with belief propagation, IEEE Transactions on Aerospace and Electronic Systems,
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
        """  # noqa: E501

        # Calculate MultipleHypothesis for each Track over all available Detections
        hypotheses = {
            track: self.hypothesiser.hypothesise(track, detections, timestamp)
            for track in tracks}

        if not hypotheses or not detections:  # No tracks or no detections
            return hypotheses
        else:
            return self._compute_multi_hypotheses(tracks, detections, hypotheses, timestamp)

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
                             if hyp.measurement is detection)
                    likelihood_matrix[i, j + 1] = hyp.weight

        # change the normalisation of the likelihood matrix to have the no measurement
        # association hypothesis normalised to unity
        likelihood_matrix /= likelihood_matrix[:, [0]]

        return likelihood_matrix.astype(float)

    @staticmethod
    def _loopy_belief_propagation(likelihood_matrix, n_iterations, delta):
        """
        Perform loopy belief propagation (Williams and Lau, 2014) to determine the approximate
        marginal association probabilities (of tracks to measurements). This requires:
        1. likelihood_matrix = single target association weights
        2. n_iterations = number of iterations between convergence checks
        3. delta = deviation tolerance(of approximate weights from true weights)
        """
        # number of tracks
        num_tracks = likelihood_matrix.shape[0]

        # number of measurements/detections
        num_measurements = likelihood_matrix.shape[1] - 1

        # initialise
        iteration: int = 0
        alpha: float = 1.0
        d: float = 0.0

        # allocate memory
        nu = np.ones((num_tracks, num_measurements))
        nu_tilde = np.zeros((num_tracks, num_measurements))
        assoc_prob_matrix = np.zeros((num_tracks, num_measurements + 1))

        # determine W_star
        w_star: float = np.max(np.sum(likelihood_matrix[:, 1:], axis=1))

        # loopy belief propagation
        while iteration == 0 or (alpha * d) / (1 - alpha) >= 0.5 * np.log10(1 + delta):

            for k in range(1, n_iterations + 1):

                # increment the number of iterations
                iteration += 1

                # calculate L-R message
                val = likelihood_matrix[:, 1:] * nu
                # Minus val to remove j = j'
                s = 1 + np.sum(val, axis=1, keepdims=True) - val
                mu = likelihood_matrix[:, 1:] / s

                # save values for convergence check
                if k == n_iterations:
                    nu_tilde = nu.copy()

                # calculate R-L messages
                nu = 1 / (1 + np.sum(mu, axis=0, keepdims=True) - mu)

            # check for convergence
            d = np.max(np.abs(np.log10(nu / nu_tilde)))

            # determine alpha
            if d > 0:
                alpha = np.log10((1 + w_star*d) / (1 + w_star))
                alpha /= np.log10(d)
            else:
                alpha = 0.0

            # if w_star has a very large value, alpha = 1 which causes division by zero in the
            # convergence check therefore, set alpha to be a nominal value just short of unity
            if alpha == 1:
                alpha = (1 - 1e-10)

        # calculate marginal probabilities (beliefs)
        s = 1 + np.sum(likelihood_matrix[:, 1:] * nu, axis=1, keepdims=True)
        assoc_prob_matrix[:, :1] = 1 / s
        assoc_prob_matrix[:, 1:] = (likelihood_matrix[:, 1:] * nu) / s

        # return the matrix of marginal association probabilities
        return assoc_prob_matrix.astype(float)

    @classmethod
    def _compute_multi_hypotheses(cls, tracks, detections, hypotheses, time):

        # Tracks and detections must be in a list, so we can keep track of their order
        track_list = list(tracks)
        detection_list = list(detections)

        # calculate the single target association weights
        likelihood_matrix = cls._calc_likelihood_matrix(track_list, detection_list, hypotheses)

        # Run Loopy Belief Propagation to determine the marginal association probability matrix
        n_iterations: int = 1
        delta: float = 0.001
        assoc_prob_matrix = cls._loopy_belief_propagation(likelihood_matrix, n_iterations, delta)

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
                         if hypothesis.measurement is detection)

                pro_detect_assoc = Probability(assoc_prob_matrix[i, j])
                single_measurement_hypotheses.append(
                    SingleProbabilityHypothesis(
                        hypothesis.prediction,
                        hypothesis.measurement,
                        measurement_prediction=hypothesis.measurement_prediction,
                        probability=pro_detect_assoc))

            new_hypotheses[track] = MultipleHypothesis(single_measurement_hypotheses, True, 1)

        return new_hypotheses


class JPDAwithEHM(JPDA):
    r"""Joint Probabilistic Data Association with Efficient Hypothesis Management (EHM)

    This is a faster alternative of the standard :class:`~.JPDA` algorithm, which makes use of
    Efficient Hypothesis Management (EHM) to compute the exact marginal association probabilities
    of tracks to measurements. See [#]_ for further details.

    References
    ----------
    .. [#] S. Maskell, M. Briers, and R. Wright. "Fast mutual exclusion." Signal and Data
       Processing of Small Targets 2004. Vol. 5428. SPIE, 2004.
    """

    def associate(self, tracks, detections, timestamp, **kwargs):

        # Calculate MultipleHypothesis for each Track over all
        # available Detections
        hypotheses = self.generate_hypotheses(tracks, detections, timestamp, **kwargs)

        # Partition tracks into independent clusters and order tracks in each cluster
        clusters = TrackClusterer(hypotheses)

        # Update the hypotheses with the new association probabilities for each cluster
        new_hypotheses = dict()
        for cluster in clusters.clustered_hypotheses:

            # Run EHM2 on cluster and get cluster hypotheses with new probabilities
            tree = self._get_tree(cluster)
            cluster_hypotheses = tree.get_posterior_hypotheses()

            # Update hypotheses for each track in the cluster
            for track, new_track_hypotheses in cluster_hypotheses.items():

                single_measurement_hypotheses = list()
                for this_hypothesis, new_probability in new_track_hypotheses:
                    single_measurement_hypotheses.append(
                        SingleProbabilityHypothesis(
                            this_hypothesis.prediction,
                            this_hypothesis.measurement,
                            measurement_prediction=this_hypothesis.measurement_prediction,
                            probability=new_probability))

                new_hypotheses[track] = MultipleHypothesis(single_measurement_hypotheses, True, 1)

        return new_hypotheses

    @staticmethod
    def _get_tree(cluster):
        return EHMTree(cluster, make_tree=False)


class JPDAwithEHM2(JPDAwithEHM):
    r"""Joint Probabilistic Data Association with Efficient Hypothesis Management 2 (EHM2)

    This is a faster alternative of the standard :class:`~.JPDA` algorithm, which makes use of
    Efficient Hypothesis Management 2 (EHM2) to compute the exact marginal association
    probabilities of tracks to measurements. EHM2 takes advantage of conditional independence of
    track-measurement pairs to achieve better computational performance than EHM in certain
    scenarios. See [#]_ for further details.

    References
    ----------
    .. [#] P. Horridge and S. Maskell, "Real-Time Tracking Of Hundreds Of Targets With Efficient
       Exact JPDAF Implementation," 2006 9th International Conference on Information Fusion,
       Florence, Italy, 2006, pp. 1-8
    """

    @staticmethod
    def _get_tree(cluster):
        return EHMTree(cluster, make_tree=True)
