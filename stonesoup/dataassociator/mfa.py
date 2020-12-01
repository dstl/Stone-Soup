import os
import importlib
import numpy as np

from .base import DataAssociator
from ..base import Property
from ..hypothesiser import Hypothesiser
from ..types.multihypothesis import MultipleHypothesis
from ..wrapper.matlab import MatlabWrapper


class MFADataAssociator(DataAssociator, MatlabWrapper):
    """MFADataAssociator
    Performs data association over a sliding window of historic scans using the Multi-Frame
    Assignment algorithm.
    """

    hypothesiser: Hypothesiser = Property(
        doc='Generate a set of hypotheses for each prediction-detection pair')
    slide_window: int = Property(doc='Length of MFA slide window')

    @property
    def dir_path(self):
        return os.path.join(os.path.dirname(importlib.util.find_spec('stonesoup').origin),
                            'dataassociator/_matlab')

    def associate(self, tracks, detections, timestamp, **kwargs):
        tracks_list = [track for track in tracks]

        # Generate a set of hypotheses for each track on each detection
        hypotheses = {track: self.hypothesiser.hypothesise(track, detections, timestamp)
                      for track in tracks_list}

        # Create MFA input
        hypothesesIn = {'cost': [], 'trackID': [], 'measHistory': []}
        for idx, (track, multihypothesis) in enumerate(hypotheses.items()):
            for hyp in multihypothesis:
                hypothesesIn['cost'].append(-hyp.prediction.weight.log_value)
                hypothesesIn['trackID'].append(idx+1)
                hypothesesIn['measHistory'].append(hyp.prediction.tag)

        # Fill-in zeros when measHistory is less than sliding window
        for i, history in enumerate(hypothesesIn['measHistory']):
            if len(history)<self.slide_window:
                diff = self.slide_window - len(history)
                history = [0 for _ in range(diff)] + history
            indices = [i+1 for i in range(len(history))]
            hypothesesIn['measHistory'][i] = np.array([indices, history])

        hypothesesIn['cost'] = self.matlab_array(np.array(hypothesesIn['cost']))
        hypothesesIn['trackID'] = self.matlab_array(np.array(hypothesesIn['trackID']))
        hypothesesIn['measHistory'] = self.matlab_array(np.array(hypothesesIn['measHistory']).T)

        # Run MFA
        _, hyps_out = self.matlab_engine.mfa3(hypothesesIn['cost'],
                                              hypothesesIn['trackID'],
                                              hypothesesIn['measHistory'],
                                              self.slide_window, nargout=2)
        # Convert to numpy arrays
        for key, value in hyps_out.items():
            hyps_out[key] = np.array(value)

        # Tags look-up table (necessary to speed up construction of hypotheses)
        lut = {track: [] for track in tracks}
        for track_id, tag in zip(hyps_out['trackID'].ravel(), hyps_out['measHistory']):
            track = tracks_list[track_id - 1]
            l = len(track.state.components[0].tag)+1
            lut[track].append(tag[-l:].tolist())

        # Construct new hypotheses
        new_hypotheses = {track: [] for track in tracks}
        for track, tags in lut.items():
            multihypothesis = hypotheses[track]
            valid_hyps = [hyp for hyp in multihypothesis if hyp.prediction.tag in tags]
            new_hypotheses[track] = MultipleHypothesis(valid_hyps)

        return new_hypotheses