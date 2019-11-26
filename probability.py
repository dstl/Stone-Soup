# -*- coding: utf-8 -*-

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


class SimplePDA(DataAssociator):
    """Simple Probabilistic Data Association (PDA)

    Given a set of detections and a set of tracks, each detection has a
    probability that it is associated each specific track.  For each track,
    associate the highest probability (remaining) detection hypothesis with
    that track.

    This particular data associator assumes no gating; all detections have the
    possibility to be associated with any track.  This can lead to excessive
    computation time.
    """

    hypothesiser = Property(
        Hypothesiser,
        doc="Generate a set of hypotheses for each prediction-detection pair")

    def associate(self, tracks, detections, time):
        """Associate detections with predicted states.

        Parameters
        ----------
        tracks : list of :class:`Track`
            Current tracked objects
        detections : list of :class:`Detection`
            Retrieved measurements
        time : datetime
            Detection time to predict to

        Returns
        -------
        dict
            Key value pair of tracks with associated detection
        """

        # Generate a set of hypotheses for each track on each detection
        hypotheses = {
            track: self.hypothesiser.hypothesise(track, detections, time)
            for track in tracks}

        return associate_highest_probability_hypotheses(tracks, hypotheses)


class JPDA(DataAssociator):
    r"""Joint Probabilistic Data Association (JPDA)

    Given a set of Detections and a set of Tracks, each Detection has a
    probability that it is associated with each specific Track.  Rather than
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

    hypothesiser = Property(
        PDAHypothesiser,
        doc="Generate a set of hypotheses for each prediction-detection pair")
    gate_ratio = Property(
        float,
        doc="If probability of Detection/Track association is less than this "
            "many times less than probability of MissedDetection, treat "
            "probability of association as 0."
    )

    def associate(self, tracks, detections, time):
        """Associate detections with predicted states.

        Parameters
        ----------
        tracks : list of :class:`Track`
            Current tracked objects
        detections : list of :class:`Detection`
            Retrieved measurements
        time : datetime
            Detection time to predict to

        Returns
        -------
        dict
            Key value pair of tracks with associated detection
        """

        # Calculate MultipleHypothesis for each Track over all
        # available Detections
        hypotheses = {
            track: self.hypothesiser.hypothesise(track, detections, time)
            for track in tracks}

        # enumerate the Joint Hypotheses of track/detection associations
        joint_hypotheses = \
            self.enumerate_JPDA_hypotheses(tracks, hypotheses, self.gate_ratio)

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
                    MissedDetection(timestamp=time),
                    measurement_prediction=hypotheses[track][0]
                    .measurement_prediction,
                    probability=prob_misdetect))

            # record hypothesis for any given Detection being associated with
            # this track
            for hypothesis in hypotheses[track]:
                if not hypothesis:
                    continue
                pro_detect_assoc = Probability.sum(
                    joint_hypothesis.probability
                    for joint_hypothesis in joint_hypotheses
                    if joint_hypothesis.
                        hypotheses[track].measurement is hypothesis.measurement)

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
    def enumerate_JPDA_hypotheses(cls, tracks, multihypths, gate_ratio):

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
            missed_probability = \
                multihypths[track].get_missed_detection_probability()
            missed_gate = missed_probability/gate_ratio
            for hypothesis in multihypths[track]:
                # Always include missed detection (gate ratio < 1)
                if not hypothesis or hypothesis.probability >= missed_gate:
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

# ==========================================
#
# HELPER METHODS
#
# ==========================================


def associate_highest_probability_hypotheses(tracks, hypotheses):
    """Associate Detections with Tracks according to highest probability hypotheses

        Parameters
        ----------
        tracks : list of :class:`Track`
            Current tracked objects
        hypotheses : list of :class:`ProbabilityMultipleHypothesis`
            Hypothesis containing probability each of the Detections is
            associated with the specified Track (or MissedDetection)

        Returns
        -------
        dict
            Key value pair of tracks with associated detection
    """
    associations = {}

    if not tracks or not hypotheses:
        return associations

    associated_measurements = set()
    while tracks > associations.keys():
        # Define a 'greedy' association
        highest_probability_hypothesis = None

        for track in tracks - associations.keys():
            for hypothesis in hypotheses[track]:
                # A measurement may only be associated with a single track
                current_probability = hypothesis.probability
                if hypothesis.measurement in \
                        associated_measurements:
                    continue
                # best_hypothesis is 'greater than' other
                if (highest_probability_hypothesis is None
                        or current_probability >
                        highest_probability_hypothesis.probability):
                    highest_probability_hypothesis = \
                        hypothesis
                    highest_probability_track = track

        associations[highest_probability_track] = \
            hypotheses[highest_probability_track]
        if highest_probability_hypothesis:
            associated_measurements.add(
                highest_probability_hypothesis.measurement)

    return associations
