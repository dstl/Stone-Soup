from pyehm.plugins.stonesoup import JPDAWithEHM2

from stonesoup.types.detection import MissedDetection
from stonesoup.types.hypothesis import SingleProbabilityHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.numeric import Probability


class JIPDAWithEHM2(JPDAWithEHM2):

    @classmethod
    def _compute_multi_hypotheses(cls, tracks, detections, hypotheses, time):

        # Tracks and detections must be in a list so we can keep track of their order
        track_list = list(tracks)
        detection_list = list(detections)

        # Calculate validation and likelihood matrices
        validation_matrix, likelihood_matrix = \
            cls._calc_validation_and_likelihood_matrices(track_list, detection_list, hypotheses)

        # Run EHM
        assoc_prob_matrix = cls._run_ehm(validation_matrix, likelihood_matrix)

        # Calculate MultiMeasurementHypothesis for each Track over all
        # available Detections with probabilities drawn from the association matrix
        new_hypotheses = dict()

        for i, track in enumerate(track_list):

            single_measurement_hypotheses = list()

            # Null measurement hypothesis
            null_hypothesis = next((hyp for hyp in hypotheses[track] if not hyp), None)
            w = null_hypothesis.metadata['w']
            prob_misdetect = Probability(assoc_prob_matrix[i, 0])/(1+w)
            single_measurement_hypotheses.append(
                SingleProbabilityHypothesis(
                    hypotheses[track][0].prediction,
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

            track.exist_prob = Probability.sum(hyp.probability
                                               for hyp in single_measurement_hypotheses)

            new_hypotheses[track] = MultipleHypothesis(single_measurement_hypotheses, True, 1)

        return new_hypotheses