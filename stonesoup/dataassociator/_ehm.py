import numpy as np
from stonesoup.types.numeric import Probability


class EHMTree:
    r"""EHMTree

    Construct a tree for Efficient Hypothesis Management / EHM 2 algorithms
    """

    def __init__(self, track_hypotheses, make_tree=True):
        """ Construct the EHMTree object from a list of track hypotheses

        Parameters
        ----------
        track_hypotheses: list of (track, list of hypotheses)
            Each Hypothesis should have a .measurement and .probability variable where probability
            is of type stonesoup.types.numeric.Probability and (not measurement) is True for null
            measurement assignments
        make_tree: bool
            If True, branch the tree according to the EHM 2 algorithm, otherwise have a list of
            tracks as in EHM
        """
        # Dict of track -> level
        self.levels = {}
        # List of tracks corresponding to root levels
        self.root_tracks = []

        # Make conditional dependence tree
        self._make_tree(track_hypotheses, make_tree)

        # Make empty nodes
        self._make_nodes()

        # Set backward weights
        self._set_backward_weights()

        # Set forward weights
        self._set_forward_weights()

    def get_posterior_hypotheses(self):

        """
        Get the hypotheses of each track along with their revised probabilities under the Mutual
        Exclusion constraint.

        Returns
        -------
        dict of track -> list of (hypothesis, probability)
            Dict of Track to list of (Hypothesis, Probability) pairs, where the Probability is the
            revised probability
        """
        # Get posterior hypothesis for each track
        posterior_hypotheses = {}
        for track, level in self.levels.items():
            posterior_hypotheses[track] = self._get_posterior_hypotheses_level(level)
        return posterior_hypotheses

    def get_num_nodes(self):
        """
        Get the total number of nodes in the tree
        """
        return sum(len(level.nodes) for level in self.levels.values())

    def _make_tree(self, track_hypotheses, make_tree):
        """
        Build up the tree from the track hypotheses
        """
        # Build up tree from the roots
        for track, measurement_hypotheses in reversed(track_hypotheses):

            # Get measurements which might conflict
            meas = set([x.measurement for x in measurement_hypotheses if x.measurement])

            accmeas = meas
            child_tracks = []
            new_root_tracks = [track]

            # Go through each of the existing roots
            for root_track in self.root_tracks:

                if meas & self.levels[root_track].accumulated_measurements or (not make_tree):
                    # If this track's measurements intersect with the root, make the root a child
                    # of this track and gather the measurement set
                    # If make_tree is False, always place the previous root below the current track
                    child_tracks.append(root_track)
                    accmeas = accmeas.union(self.levels[root_track].accumulated_measurements)
                else:
                    # Otherwise, create another root
                    new_root_tracks.append(root_track)

            self.levels[track] = EHMLevel(accmeas, measurement_hypotheses, child_tracks)
            self.root_tracks = new_root_tracks

    def _make_nodes(self):
        # Make the tree nodes
        for root_track in self.root_tracks:
            # Create a node with no used measurements
            self.levels[root_track].nodes[frozenset()] = EHMNode()
            # Make descendent nodes of this root
            self._make_child_nodes(self.levels[root_track])

    def _make_child_nodes(self, level):
        # Make the child nodes for this level, and recursively for its children
        for child_track in level.child_tracks:

            # Add nodes to children of this level
            child_level = self.levels[child_track]

            for usedmeas, node in level.nodes.items():

                # For each hypothesis of this track:
                for hypothesis in level.measurement_hypotheses:

                    if hypothesis.measurement not in usedmeas:
                        # If the hypothesis doesn't conflict with already used measurements, get
                        # the new used measurement set
                        newusedmeas = child_level._get_new_used_measurements(usedmeas, hypothesis)
                        # If a node for this set doesn't exist, make one
                        if newusedmeas not in child_level.nodes:
                            child_level.nodes[newusedmeas] = EHMNode()

            # Recursively make nodes for child levels
            self._make_child_nodes(child_level)

    def _set_backward_weights(self):
        for root_track in self.root_tracks:
            self._set_backward_weights_level(self.levels[root_track])

    def _set_backward_weights_level(self, level):

        # Set backward weights for the descendent nodes
        for child_track in level.child_tracks:
            self._set_backward_weights_level(self.levels[child_track])

        # For each node in this level
        for usedmeas, node in level.nodes.items():

            node.backward_weight = Probability(0.0)

            # For each hypothesis of this level's track:
            for hypothesis in level.measurement_hypotheses:

                # If hypothesis doesn't conflict with previously used measurements:
                if hypothesis.measurement not in usedmeas:

                    # Get product of this hypothesis probability and the backward weights of child
                    # nodes and add on to backward weights
                    thisprob = hypothesis.probability
                    for child_track in level.child_tracks:
                        child_level = self.levels[child_track]
                        newusedmeas = child_level._get_new_used_measurements(usedmeas, hypothesis)
                        thisprob *= child_level.nodes[newusedmeas].backward_weight
                    node.backward_weight += thisprob

    def _set_forward_weights(self):
        for root_track in self.root_tracks:
            # Set forward weight of root node to 1.0
            for node in self.levels[root_track].nodes.values():
                node.forward_weight = Probability(1.0)
            # Set forward weights of descendent level
            self._set_forward_weights_level(self.levels[root_track])

    def _set_forward_weights_level(self, level):

        # For each node in this level:
        for usedmeas, node in level.nodes.items():

            # For each hypothesis for this level's track:
            for hypothesis in level.measurement_hypotheses:

                # If hypothesis doesn't conflict with the previously used measurements
                if hypothesis.measurement not in usedmeas:

                    # Get backward weights of child nodes (so we can get product of sibling nodes
                    # later)
                    child_backward_weights = {}
                    for child_track in level.child_tracks:
                        child_level = self.levels[child_track]
                        newusedmeas = child_level._get_new_used_measurements(usedmeas, hypothesis)
                        child_backward_weights[child_track] = \
                            child_level.nodes[newusedmeas].backward_weight

                    # For each child level
                    for child_track in level.child_tracks:

                        # Get product of hypothesis probability, this node's forward weight and
                        # child node's sibling weights
                        thisprob = hypothesis.probability * node.forward_weight
                        for sibling_track in level.child_tracks:
                            if sibling_track != child_track:
                                thisprob *= child_backward_weights[sibling_track]
                        child_level = self.levels[child_track]
                        newusedmeas = child_level._get_new_used_measurements(usedmeas, hypothesis)

                        # Push weight down to child node's forward weight
                        child_level.nodes[newusedmeas].forward_weight += thisprob

        # Recursively calculate forward weights for descendent levels
        for child_track in level.child_tracks:
            self._set_forward_weights_level(self.levels[child_track])

    def _get_posterior_hypotheses_level(self, level):

        posterior_hypotheses = []
        probsum = Probability(0.0)

        # For each hypothesis
        for hypothesis in level.measurement_hypotheses:
            thisprob = Probability(0.0)

            # For each node
            for usedmeas, node in level.nodes.items():

                # If the hypothesis doesn't conflict with previously used measurements:
                if hypothesis.measurement not in usedmeas:

                    # Get the product of the backward weights of the children and the forward
                    # weight and add to thisprob
                    backprod = Probability(1.0)
                    for child_track in level.child_tracks:
                        child_level = self.levels[child_track]
                        newusedmeas = child_level._get_new_used_measurements(usedmeas, hypothesis)
                        backprod *= self.levels[child_track].nodes[newusedmeas].backward_weight
                    thisprob += backprod * node.forward_weight

            # Set (unnormalised) probability for this assignment
            thisprob *= hypothesis.probability
            posterior_hypotheses.append((hypothesis, thisprob))

            # Get sum of all probabilities for normalisation
            probsum += thisprob

        # Return normalised probabilities
        return [(h, p / probsum) for h, p in posterior_hypotheses]


class EHMNode:

    def __init__(self):

        # Forward and backward weights
        self.forward_weight = Probability(0.0)
        self.backward_weight = Probability(0.0)


class EHMLevel:

    def __init__(self, accumulated_measurements, measurement_hypotheses, child_tracks):
        # Set of measurement hypotheses used by this level or a descendant
        self.accumulated_measurements = accumulated_measurements
        # List of measurement hypotheses and probabilities (None = null assignment)
        self.measurement_hypotheses = measurement_hypotheses
        # List of tracks corresponding to child levels
        self.child_tracks = child_tracks
        # Dict of (set of used measurements) -> node
        self.nodes = {}

    def _get_new_used_measurements(self, previous_used_measurements, hypothesis):
        # Return measurements which might conflict with assignments at this level or
        # descendants, given parent hypothesis and set of previously used measurements
        return previous_used_measurements.union({hypothesis.measurement}).intersection(
            self.accumulated_measurements)


class TrackClusterer:

    def __init__(self, hypotheses):

        # Get initial clusters of tracks (1 track per cluster)
        self.clusters = [((track,), set([hyp.measurement for hyp in hyps if hyp.measurement]))
                         for track, hyps in hypotheses.items()]

        # Get table of number of intersections
        nintersect_table = self._get_num_intersect_table()

        # Continue until we have only one cluster or none of them intersect
        while len(self.clusters) > 0:

            # Find maximum intersection - if no intersection, break
            maxi, maxj = np.unravel_index(np.argmax(nintersect_table), nintersect_table.shape)
            if nintersect_table[maxi, maxj] == 0:
                break

            # Merge one cluster into another and delete it
            self.clusters[maxi] = (self.clusters[maxi][0] + self.clusters[maxj][0],
                                   self.clusters[maxi][1].union(self.clusters[maxj][1]))
            del self.clusters[maxj]

            # Compute new intersection table
            nintersect_table = np.delete(
                np.delete(nintersect_table, maxj, axis=0), maxj, axis=1
            )
            for j in range(maxi):
                nintersect_table[j, maxi] = len(self.clusters[maxi][1] & self.clusters[j][1])
            for j in range(maxi+1, nintersect_table.shape[0]):
                nintersect_table[maxi, j] = len(self.clusters[maxi][1] & self.clusters[j][1])

        # Get clustered hypotheses
        self.clustered_hypotheses = []
        for cluster_tracks, _ in self.clusters:
            self.clustered_hypotheses.append([(track, hypotheses[track])
                                              for track in cluster_tracks])

    def _get_num_intersect_table(self):
        # Get table nintersect_table[i1, i2] = number of measurements in both clusters i1 and i2,
        # where i2 > i1
        nclusters = len(self.clusters)
        nintersect_table = np.zeros((nclusters, nclusters)).astype('int')
        for i1, c1 in enumerate(self.clusters):
            for i2, c2 in enumerate(self.clusters[i1+1:]):
                nintersect_table[i1][i1+i2+1] = len(c1[1].intersection(c2[1]))
        return nintersect_table
