import numpy as np

from .. import DataAssociator
from ...base import Property
from ...hypothesiser.mfa import MFAHypothesiser
from ...types.multihypothesis import MultipleHypothesis
from ._init import init_hyp_info, Hyp
from ._step import MAX_ITERATION_COUNT, AlgorithmState, algorithm_step, prune_hypotheses


class MFADataAssociator(DataAssociator):
    """Data associator using multi-frame assignment algorithm over a sliding window.

    References
    ----------
    1. Xia, Y., Granström, K., Svensson, L., García-Fernández, Á.F., and Williams, J.L.,
       2019. Multiscan Implementation of the Trajectory Poisson Multi-Bernoulli Mixture Filter.
       J. Adv. Information Fusion, 14(2), pp. 213–235.

    """

    hypothesiser: MFAHypothesiser = Property(
        doc='Generate a set of hypotheses for each prediction-detection pair')
    slide_window: int = Property(doc='Length of MFA slide window')

    def associate(self, tracks, detections, timestamp, **kwargs):
        # No tracks, nothing to do
        if not tracks:
            return {}
        # Generate a set of hypotheses for each track on each detection
        # and shuffle hypothesis data into format required by the MFA algorithm
        tracks_list = []
        # TODO: Avoid dependency on indexes
        detections_tuple = tuple(detections)
        hypotheses = []
        hyps = []
        for trackID, (track, multihypothesis) in enumerate(
                self.generate_hypotheses(tracks, detections, timestamp,
                                         detections_tuple=detections_tuple, **kwargs).items()):
            tracks_list.append(track)
            hypotheses.append(multihypothesis)
            hyps.extend([
                Hyp.create(
                    trackID=trackID,
                    cost=-np.log(individual_hypothesis.prediction.weight),
                    measHistory=individual_hypothesis.prediction.tag,  # measurement indices
                    slide_window=self.slide_window
                )
                for individual_hypothesis in multihypothesis
            ])
        hyp_info = init_hyp_info(hyps, self.slide_window)

        # Run the MFA algorithm
        alg_state = AlgorithmState.initialise(self.slide_window, len(hyp_info.hyps))
        for iteration in range(MAX_ITERATION_COUNT):
            algorithm_step(alg_state, hyp_info)
            if alg_state.should_break:
                break
        best_hypotheses = [hyp_info.hyps[i] for i in alg_state.get_best_hypothesis_indices()]

        # Construct new hypotheses from the results: do n-scan pruning against the best hypotheses
        # for this window, and only keep those hypotheses that match the results of pruning.
        pruned_hypotheses = prune_hypotheses(best_hypotheses, hyp_info.hyps)
        new_hypotheses = {track: [] for track in tracks}
        for trackID, hyps in pruned_hypotheses.items():
            track = tracks_list[trackID]
            tags = [hyp.measHistory for hyp in hyps]
            valid_hyps = []
            for h in hypotheses[trackID]:
                if h.prediction.tag in tags:
                    valid_hyps.append(h)
            new_hypotheses[track] = MultipleHypothesis(valid_hyps)

        return new_hypotheses
