# -*- coding: utf-8 -*-
import numpy as np

from .base import DataAssociator
from ._assignment import assign2D
from ..base import Property
from ..hypothesiser import Hypothesiser
from ..types.hypothesis import SingleHypothesis, SingleProbabilityHypothesis


class NearestNeighbour(DataAssociator):
    """Nearest Neighbour Associator

    Scores and associates detections to a predicted state using the Nearest
    Neighbour method.
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

        # Only associate tracks with one or more hypotheses
        associate_tracks = {track
                            for track, track_hypotheses in hypotheses.items()
                            if track_hypotheses}

        associations = {}
        associated_measurements = set()
        while associate_tracks > associations.keys():
            # Define a 'greedy' association
            best_hypothesis = None
            for track in associate_tracks - associations.keys():
                for hypothesis in hypotheses[track]:
                    # A measurement may only be associated with a single track
                    if hypothesis.measurement in associated_measurements:
                        continue
                    # best_hypothesis is 'greater than' other
                    if (best_hypothesis is None
                            or hypothesis > best_hypothesis):
                        best_hypothesis = hypothesis
                        best_hypothesis_track = track

            associations[best_hypothesis_track] = best_hypothesis
            if best_hypothesis:
                associated_measurements.add(best_hypothesis.measurement)

        return associations


class GlobalNearestNeighbour(DataAssociator):
    """Global Nearest Neighbour Associator

    Scores and associates detections to a predicted state using the Global
    Nearest Neighbour method, assuming a distance-based hypothesis score.
    """

    hypothesiser = Property(
        Hypothesiser,
        doc="Generate a set of hypotheses for each prediction-detection pair")

    def associate(self, tracks, detections, time):
        """Associate a set of detections with predicted states.

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

        # Link hypotheses into a set of joint_hypotheses and evaluate
        joint_hypotheses = self.enumerate_joint_hypotheses(hypotheses)
        associations = max(joint_hypotheses)

        return associations


class GNNWith2DAssignment(DataAssociator):
    """Global Nearest Neighbour Associator

    Associates detections to a predicted state using the
    Global Nearest Neighbour method, utilising a 2D matrix of
    distances and a "shortest path" assignment algorithm.
    """

    hypothesiser = Property(
        Hypothesiser,
        doc="Generate a set of hypotheses for each prediction-detection pair")

    def associate(self, tracks, detections, time):
        """Associate a set of detections with predicted states.

        Parameters
        ----------
        tracks : set of :class:`Track`
            Current tracked objects
        detections : set of :class:`Detection`
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

        # Create dictionary for associations
        associations = {}

        # Extract detected tracks
        detected_tracks = [track
                           for track, track_hypotheses in hypotheses.items()
                           if any(track_hypotheses)]

        # Store associations for undetected/missed tracks
        # NOTE: It is assumed that if a track is undetected/missed, then it
        #       will only have a single missed detection hypothesis
        for track in hypotheses.keys() - set(detected_tracks):
            if hypotheses[track]:
                associations[track] = hypotheses[track][0]

        # No need to perform data association if all tracks are missed
        if not detected_tracks:
            return associations

        # Convert sets to indexable lists
        detections = list(detections)

        # Generate 2d array "matrix" of hypotheses mapping track to detection
        hypothesis_matrix = np.empty(
            (len(detected_tracks), len(detections) + len(detected_tracks)),
            SingleHypothesis)
        for i, track in enumerate(detected_tracks):
            row = np.empty(
                (hypothesis_matrix.shape[1]), SingleHypothesis)
            for hypothesis in hypotheses[track]:
                if not hypothesis:
                    row[len(detections) + i] = hypothesis
                else:
                    row[detections.index(hypothesis.measurement)] = hypothesis
            hypothesis_matrix[i] = row

        # Determine type of hypothesis used, probability or distance
        # Probability is maximise problem, distance is minimise problem
        # Mixed hypotheses cannot be computed at this time
        hypothesis_types = {
            isinstance(hypothesis, SingleProbabilityHypothesis)
            for row in hypothesis_matrix for hypothesis in row
            if hypothesis is not None}
        if len(hypothesis_types) > 1:
            raise RuntimeError(
                "2d assignment does not support mixed hypothesis types")
        probability_flag = hypothesis_types.pop()

        # Generate 2d array "matrix" of distances
        # Use probabilities instead for probability based hypotheses
        distance_matrix = np.empty(hypothesis_matrix.shape)
        for x in range(hypothesis_matrix.shape[0]):
            for y in range(hypothesis_matrix.shape[1]):
                if hypothesis_matrix[x][y] is None:
                    distance_matrix[x][y] = -np.inf if probability_flag \
                        else np.inf
                else:
                    if probability_flag:
                        distance_matrix[x][y] = \
                            hypothesis_matrix[x][y].probability
                    else:
                        distance_matrix[x][y] = \
                            hypothesis_matrix[x][y].distance

        # Use "shortest path" assignment algorithm on distance matrix
        # to assign tracks to nearest detection
        # Maximise flag = true for probability instance
        # (converts minimisation problem to maximisation problem)
        gain, col4row, row4col = assign2D(
            distance_matrix, probability_flag)

        # Ensure the problem was feasible
        if gain.size <= 0:
            raise RuntimeError("Assignment was not feasible")

        # Generate dict of key/value pairs
        for j, track in enumerate(detected_tracks):
            associations[track] = hypothesis_matrix[j][col4row[j]]

        return associations
