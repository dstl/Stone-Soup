# -*- coding: utf-8 -*-
import itertools

from .base import DataAssociator
from ..base import Property
from ..hypothesiser import Hypothesiser
from ..hypothesiser.probability import PDAHypothesiser
from ..types import Probability, MissedDetection, \
    SingleMeasurementProbabilityHypothesis, ProbabilityJointHypothesis
from ..types.multimeasurementhypothesis import \
    ProbabilityMultipleMeasurementHypothesis
import numpy as np
import itertools
import time as tm
import copy


class SimplePDA(DataAssociator):
    """Simple Probabilistic Data Associatoion (PDA)

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
    """Joint Probabilistic Data Associatoion (JPDA)

    Given a set of detections and a set of tracks, each detection has a
    probability that it is associated with each specific track.  However,
    when a detection could be associated with one of several tracks, this
    must be calculated via a joint probability.  In the end, the highest-
    probability Joint Hypothesis is returned as the correct Track/Detection
    association set.

    This particular data associator has no configurable gating; therefore,
    all detections have the possibility to be associated with any track
    (although the probability of association could be very close to 0).  This
    can lead to excessive computation time due to combinatorial explosion.  To
    address this problem, some rudimentary gating is implemented.  If

    .. math::

          prob_association(Detection, Track) <
          \frac{prob_association(MissedDetection, Track);gate_ratio}

    then Detection is assumed to be outside Track's gate ('gate_ratio'
    arbitrarily set to 5).  This calculation takes place in the function
    'enumerate_JPDA_hypotheses()'.
    """

    hypothesiser = Property(
        PDAHypothesiser,
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

        # Calculate MultipleMeasurementHypothesis for each Track over all
        # available Detections
        hypotheses = {
            track: self.hypothesiser.hypothesise(track, detections, time)
            for track in tracks}

        # enumerate the Joint Hypotheses of track/detection associations
        joint_hypotheses = self.enumerate_JPDA_hypotheses(tracks, detections, hypotheses)

        # Calculate MultiMeasurementHypothesis for each Track over all
        # available Detections with probabilities drawn from JointHypotheses
        new_hypotheses = []

        for track_index, track in enumerate(tracks):

            single_measurement_hypotheses = list()

            # record the MissedDetection hypothesis for this track
            prob_misdetect = Probability(
                sum([joint_hypothesis.probability
                     for joint_hypothesis in joint_hypotheses
                     if isinstance(joint_hypothesis.hypotheses[track].
                                   measurement, MissedDetection)]))

            single_measurement_hypotheses.append(SingleMeasurementProbabilityHypothesis(
                hypotheses[track].single_measurement_hypotheses[0].prediction,
                MissedDetection(timestamp=time),
                measurement_prediction=hypotheses[track].single_measurement_hypotheses[0].measurement_prediction,
                probability=prob_misdetect))

            # record hypothesis for any given Detection being associated with
            # this track
            for detection in detections:
                pro_detect_assoc = Probability(
                    sum([joint_hypothesis.probability
                         for joint_hypothesis in joint_hypotheses
                         if joint_hypothesis.
                        hypotheses[track].measurement is detection]))

                single_measurement_hypotheses.append(SingleMeasurementProbabilityHypothesis(
                    hypotheses[track].single_measurement_hypotheses[0].prediction,
                    detection,
                    measurement_prediction=hypotheses[track].single_measurement_hypotheses[0].measurement_prediction,
                    probability=pro_detect_assoc))

            result = ProbabilityMultipleMeasurementHypothesis(
                single_measurement_hypotheses)
            result.normalize_probabilities()

            new_hypotheses.append(result)

        new_hypotheses_result = {
            track: new_hypothesis
            for track, new_hypothesis in zip(tracks, new_hypotheses)}

        return associate_highest_probability_hypotheses(tracks, new_hypotheses_result)


    # ==================================================================================
    #   JPDA METHOD 2
    #
    #   uses matrix permanents instead of enumeration to calculate the JPDA probabilities
    # ==================================================================================
    def associate_ver_2(self, tracks, detections, time):
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

        # Calculate MultipleMeasurementHypothesis for each Track over all
        # available Detections
        hypotheses = {
            track: self.hypothesiser.hypothesise(track, detections, time)
            for track in tracks}

        # --------------------------------------------------------------------
        # form the likelihood matrix C
        # C is an Nx(M+N) matrix where N=num tracks and M=num detections
        # C consists of two matrices (A and B) concatenated
        # A - NxM matrix - A_i,j is proability of association between Track i
        #       and Detection j
        # B - NxN matirx - sparse matrix of 0's except B_i,i is probability of
        #       Missed Detection for Track i
        # --------------------------------------------------------------------
        C = None
        if hypotheses:

            num_tracks = len(tracks)
            num_detect = len(detections)

            A = [[float(detection.probability) for detection in track.single_measurement_hypotheses if not isinstance(detection.measurement, MissedDetection)] for _, track in hypotheses.items()]

            B = [[float(next(iter(list(hypotheses.values())[row_index].single_measurement_hypotheses), MissedDetection).probability) if col_index==row_index else Probability(0) for col_index in range(0, num_tracks)] for row_index in range(0, num_tracks)]

            C = np.concatenate((A, B), axis=1)


        # --------------------------------------------------------------------
        # forms the 'processed_probs' likelihood matric
        # 'processed_probs' is an Nx(M+1) matrix where N=num tracks
        #       and M=num detections
        # processed_probs_i,j is the probability of association between
        #       Track i and Detection j WHEN JOINT ASSOCIATIONS ARE CONSIDERED
        # processed_probs_i,(M+1) is the probability that Track i had a
        #       Missed Detection
        # --------------------------------------------------------------------
        processed_probs = []
        if C is not None:

            processed_probs = np.array([[C[which_row][which_col]*perm(C, [True if inner_row == which_row else False for inner_row in range(0, num_tracks)], [True if inner_col == which_col else False for inner_col in range(0, num_detect+num_tracks)])/perm_whole_matrix for which_col in itertools.chain(range(0, num_detect),[num_detect+which_row])] for which_row in range(0, num_tracks)])

            # ALTERNATE METHOD TO CALCULATE 'processed_probs' INSTEAD OF LIST COMPREHENSION

            #processed_probs = np.empty([num_tracks, num_detect+1], dtype=Probability)
            #for row_index in range(0, num_tracks):

                #this_row_missed_detect_prob = C[row_index, num_detect+row_index]

                #for col_index in itertools.chain(range(0, num_detect),[num_detect+row_index]):

                #    local_prob = C[row_index, col_index]
                #    target_col = col_index if col_index < num_detect else -1

                    # simple gating
                    # if the probability of association between a given
                    # Detection and Track is suffiently smaller than the
                    # probability of a MissedDetection, then just set that
                    # probability in 'processed_probs' to 0 - don't bother
                    # calculating to save time
                    #if (this_row_missed_detect_prob/local_prob > 10):
                    #    processed_probs[row_index, target_col] = 0
                    #    continue

                    #boolRowsSkip = [True if inner_row == row_index else False for inner_row in range(0, num_tracks)]
                    #boolColsSkip = [True if inner_col == col_index else False for inner_col in range(0, num_detect + num_tracks)]
                    #local_perm = perm(C, boolRowsSkip, boolColsSkip)

                    #processed_probs[row_index, target_col] = local_prob * (local_perm/perm_whole_matrix)

        # update the hypotheses in 'hypotheses' with probabilities calculated with JPDA considerations in mind
        for track_index, track in enumerate(tracks):

            # update the probability of the 'MissedDetection' hypothesis
            for hypothesis in hypotheses[track].single_measurement_hypotheses:
                if isinstance(hypothesis.measurement, MissedDetection):
                    hypothesis.probability = Probability(processed_probs[track_index][-1])

            # update the probability of the Detection hypotheses
            detect_hypothesis_index = 0
            for hypothesis in hypotheses[track].single_measurement_hypotheses:
                if not isinstance(hypothesis.measurement, MissedDetection):
                    hypothesis.probability = Probability(processed_probs[track_index][detect_hypothesis_index])
                    detect_hypothesis_index += 1

        return associate_highest_probability_hypotheses(tracks, hypotheses)


    @classmethod
    def enumerate_JPDA_hypotheses(cls, tracks, input_detections, multihypths):

        detections = list(input_detections)
        joint_hypotheses = list()

        num_tracks = len(tracks)
        num_detections = len(detections)

        if num_detections <= 0 or num_tracks <= 0:
            return joint_hypotheses

        # perform a simple level of gating - all track/detection pairs for
        # which the probability of association is a certain multiple less
        # than the probability of missed detection - detection is outside the
        # gating region, association is impossible
        gate_ratio = 5
        possible_assoc = list()

        for track_index, track in enumerate(tracks):
            this_track_possible_assoc = list()
            this_track_missed_detection_probability = multihypths[track].\
                get_missed_detection_probability()
            for hypothesis_index, hypothesis in enumerate(
                    multihypths[track].single_measurement_hypotheses):
                if this_track_missed_detection_probability / \
                        hypothesis.probability <= gate_ratio:
                    this_track_possible_assoc.append(hypothesis_index)
            possible_assoc.append(tuple(this_track_possible_assoc))

        # enumerate all valid JPDA joint hypotheses: position in character
        # string is the track, digit is the assigned detection
        # (0 is missed detection)
        enum_JPDA_hypotheses = [joint_hypothesis
                                for joint_hypothesis in
                                list(itertools.product(*possible_assoc))
                                if cls.isvalid(joint_hypothesis)]

        # turn the valid JPDA joint hypotheses into 'JointHypothesis'
        for elem in enum_JPDA_hypotheses:
            local_hypotheses = {}

            for detection, track in zip(elem, tracks):
                source_multihypothesis = multihypths[track]
                #assoc_detection = detections[detection-1] if detection > 0 \
                #    else MissedDetection(
                #    timestamp=detections[detection].timestamp)

                #local_hypothesis = \
                #    SingleMeasurementProbabilityHypothesis(
                #        source_multihypothesis.prediction, assoc_detection,
                #        measurement_prediction=source_multihypothesis.
                #        measurement_prediction,
                #        probability=source_multihypothesis.
                #        weighted_measurements[detection]["weight"])

                local_hypothesis = source_multihypothesis.single_measurement_hypotheses[detection]

                local_hypotheses[track] = local_hypothesis

            joint_hypotheses.append(
                ProbabilityJointHypothesis(local_hypotheses))

        # normalize ProbabilityJointHypotheses relative to each other
        sum_probabilities = sum([hypothesis.probability
                                 for hypothesis in joint_hypotheses])
        for hypothesis in joint_hypotheses:
            hypothesis.probability /= sum_probabilities

        return joint_hypotheses

    @staticmethod
    def isvalid(joint_hypothesis):

        # 'joint_hypothesis' represents a valid joint hypothesis if:
        #   1) no digit is repeated (except 0)

        # check condition #1
        uniqueList = []
        for elem in joint_hypothesis:
            if elem in uniqueList and elem != 0:
                return False
            else:
                uniqueList.append(elem)

        return True

#==========================================
#
# HELPER METHODS
#
#==========================================

def associate_highest_probability_hypotheses(tracks, hypotheses):
    """Associate Detections with Tracks according to highest probability hypotheses

        Parameters
        ----------
        tracks : list of :class:`Track`
            Current tracked objects
        hypotheses : list of :class:`ProbabilityMultipleMeasurementHypothesis`
            Hypothesis containing probability each of the Detections is
            associated with the specified Track (or MissedDetection)

        Returns
        -------
        dict
            Key value pair of tracks with associated detection
    """
    associations = {}
    associated_measurements = set()
    while tracks > associations.keys():
        # Define a 'greedy' association
        highest_probability_hypothesis = None
        #highest_probability = Probability(0)
        for track in tracks - associations.keys():
            for hypothesis in \
                    hypotheses[track].single_measurement_hypotheses:
                # A measurement may only be associated with a single track
                current_probability = hypothesis.probability
                if hypothesis.measurement in \
                        associated_measurements:
                    continue
                # best_hypothesis is 'greater than' other
                if (highest_probability_hypothesis is None
                        or current_probability > highest_probability_hypothesis.probability):
                    highest_probability_hypothesis = \
                        hypothesis
                    highest_probability_track = track

        hypotheses[highest_probability_track]. \
            set_selected_hypothesis(highest_probability_hypothesis)
        associations[highest_probability_track] = \
            hypotheses[highest_probability_track]
        if not isinstance(highest_probability_hypothesis.measurement, MissedDetection):
            associated_measurements.add(highest_probability_hypothesis.measurement)

    return associations


#def associate_highest_probability_matrix_hypotheses(tracks, detections, time, prob_matrix):
#    """Associate Detections with Tracks according to highest probability hypotheses
#        - but using a probability matrix

#        Parameters
#        ----------
#        tracks : list of :class:`Track`
#            Current tracked objects
#        detections : list of :class:`Detection`
#            Detections from the current time period
#        time : datetime
#            Detection time to predict to/when 'detections' occurred
#        prob_matrix : np.array
#            an Nx(M+1) matrix where N=num tracks and M=num detections
#            prob_matrix_i,j is the probability of association between Track i and Detection j
#            prob_matrix_i,(M+1) is the probability that Track i had a Missed Detection

#        Returns
#        -------
#        dict
#            Key value pair of tracks with associated detection
#    """
#    associations = {}
#    associated_measurements = set()
#    while tracks > associations.keys():
#        # Define a 'greedy' association
#        high_prob_track = None
#        high_prob_detect = None

#        for track_index in prob_matrix.shape()[0]:

#            for detect_index in prob_matrix.shape()[1]:

                # A detection may only be associated with a single track
#                current_probability = prob_matrix[track_index][detect_index]
#                if detect_index in associated_measurements:
#                    continue

                # best_hypothesis is 'greater than' other
#                if (high_prob_track is None
#                        or current_probability > prob_matrix[high_prob_track][high_prob_detect]):
#                    high_prob_track = track_index
#                    high_prob_detect = detect_index

        # if 'detect_index' indicates that the highest-probability association is with a detection (not a 'MissedDetection'), then mark this detection as already associated - cannot be associated with another Track
#        if detect_index < prob_matrix.shape[1]:
#            associated_measurements.add(detect_index)

#        associations[track] = hypothesis

#    return associations


# -------------------------------------------------------------
#   Calculates the permanent of a matrix
#
#   Reference:
#   "Computation of Target-Measurement Association Probabilities Using the Matrix Permanent"
#       David Frederic Crouse and Peter Willett
#       IEEE Transactions on Aerospace and Electronic Systems
# -------------------------------------------------------------
def perm(A, boolRowsSkip=None, boolColsSkip=None):
    # check type of inputs
    if not isinstance(A, np.ndarray):
        raise Exception("Cannot calculate the permanent of a non-matrix!")
    if not A.ndim == 2:
        raise Exception("Cannot calculate the permanent of a matrix that is not 2D!")

    # cover trivial cases
    if any(dim == 0 for dim in A.shape):
        # Empty matrices have a permanent of 1 by definition.
        return 1
    elif all(dim == 1 for dim in A.shape):
        # a 1x1 matrix has a permanent equal to the single value
        return A[0][0]

    num_rows, num_cols = A.shape
    # number of columns should exceed or be equal to number of rows; if it does not, transpose matrix A
    if (num_rows > num_cols):
        A = np.transpose(A)
        temp = boolRowsSkip
        boolRowsSkip = boolColsSkip
        boolColsSkip = temp

        num_rows, num_cols = A.shape

    # cast "boolRowsSkip" and "boolColsSkip" to numpy.array
    if isinstance(boolRowsSkip, list):
        boolRowsSkip = np.array(boolRowsSkip, dtype=bool)
    if isinstance(boolColsSkip, list):
        boolColsSkip = np.array(boolColsSkip, dtype=bool)

    # Use a modified version of Ryser's algorithm if skip lists are provided.
    if (boolRowsSkip is not None) or (boolColsSkip is not None):

        num_rows_total = num_rows
        num_cols_total = num_cols

        if boolColsSkip is None or boolColsSkip.size == 0:
            boolColsSkip = np.full((num_cols_total), False, dtype=bool)

        if boolRowsSkip is None or boolRowsSkip.size == 0:
            boolRowsSkip = np.full((num_rows_total), False, dtype=bool)

        numRowsSkipped = sum(boolRowsSkip)
        numColsSkipped = sum(boolColsSkip)

        num_rows = num_rows_total - numRowsSkipped
        num_cols = num_cols_total - numColsSkipped

        # Empty matrices have a permanent of 1 by definition.
        if (num_rows == 0) or (num_cols == 0):
            return 1

        # number of columns should exceed or be equal to number of rows; if it does not, transpose matrix A
        if (num_rows > num_cols):
            A = np.transpose(A)
            temp = num_rows
            num_rows = num_cols
            num_cols = temp

        # Set the mapping of indices of the rows in the submatrix to indices in the full matrix.
        rows2keep = []

        for curRow in range(0, num_rows_total):
            if boolRowsSkip[curRow] != True:
                rows2keep.append(curRow)

        # Set the mapping of indices of the columns in the submatrix to indices in the full matrix.
        cols2keep = []

        for curCol in range(0, num_cols_total):
            if boolColsSkip[curCol] != True:
                cols2keep.append(curCol)

        binomTerm = 1
        return_val = 0
        for x in range(0, num_rows):
            return_val = return_val + SigmaSSkip(A, num_cols - num_rows + x, rows2keep, cols2keep) * binomTerm

            binomTerm = binomTerm * (1 - num_rows + num_cols + x) / (1 + x) * (-1);

        return return_val

    # use Ryser's algorithm if no skip lists are provided
    if num_rows != num_cols:
        if num_rows > num_cols:
            A = np.transpose(A)
            temp = num_rows
            num_rows = num_cols
            num_cols = temp

        binomTerm = 1
        return_val = 0
        for x in range(0, num_rows):
            return_val = return_val + SigmaS(A, num_cols - num_rows + x) * binomTerm

            binomTerm = binomTerm * (1 - num_rows + num_cols + x) / (1 + x) * (-1)

        return return_val

    # If the matrix is square, use the PERMAN algorithm.
    else:

        x = np.full((num_cols), 0, dtype=float)  # Temporary storage space.
        p = 0

        for i in range(0, num_cols):
            sumVal = sum(A[i, :])
            x[i] = A[i, num_cols - 1] - sumVal / 2

        sgn = -1
        code = []
        nCard = num_cols - 1
        while (1):
            sgn = -sgn
            prodVal = sgn
            code, nCard, isLast, j = getNextGrayCode(code, nCard)

            if (nCard != 0):
                z = 2 * code[j] - 1
                x = x + z * A[:, j]

            for i in range(0, num_cols):
                prodVal = prodVal * x[i]

            p = p + prodVal

            if isLast:
                break

        return_val = 2 * ((2 * (num_cols % 2)) - 1) * p
        return return_val


# This function gives us the product of the row sums of A.

def S(A):
    return np.prod(np.sum(A, axis=1))


# This adds up all of the possible values of S(A) where r columns of A
# have been replaced by zeros. We shall choose the
# n - r columns of A that are NOT zero.

def SigmaS(A, r):
    _, num_cols = A.shape
    combLen = num_cols - r
    curComb = list(range(0, combLen))
    return_val = 0

    while curComb:
        return_val = return_val + S(A[:, [col for col in curComb]])
        curComb = getNextCombo(curComb, num_cols)

    return return_val


# This adds up all of the possible values of S(A) where r columns of A
# have been replaced by zeros. We shall choose the
# n - r columns of A that are NOT zero.

def SigmaSSkip(A, r, rows2keep, cols2keep):

    #print("start")
    start = tm.clock()

    num_cols = len(cols2keep)
    combLen = num_cols - r
    curComb = list(range(0, combLen))

    combHistory = []

    return_val = 0
    count = 0
    while curComb:
        combHistory.append(copy.deepcopy(curComb))

        temp = A[rows2keep, :]
        temp = temp[:, [cols2keep[which_col] for which_col in curComb]]
        return_val = return_val + S(temp)
        curComb = getNextCombo(curComb, num_cols)
        count = count + 1

        #if count > 100:
        #    x=1

    end = tm.clock()
    #print(count)

    #print(end - start)

    return return_val


# ---------------------------------------------------------------------

def getNextGrayCode(code, nCard=None):
    # If called with no code, then return the first code in the sequence.
    if np.array(code).size == 0:
        n = nCard
        code = np.full((n, 1), 0, dtype=int)
        isLast = False
        nCard = 0
        j = []
        return code, nCard, isLast, j

    n = len(code) - 1

    # If no cardinality was passed, then compute it.
    if nCard is None:
        nCard = sum(code)

    # If the final gray code was passed, then just return empty matrices.
    if (nCard == code[n]) and (nCard != 0):
        code = []
        nCard = n
        isLast = []
        j = []
        return code, nCard, isLast, j

    j = 0
    if nCard % 2 != 0:
        while (1):
            j = j + 1
            if code[j - 1] != 0:
                break;

    code[j] = 1 - code[j];
    nCard = nCard + 2 * code[j] - 1;
    isLast = (nCard == code[n])

    return code, nCard, isLast, j


# ---------------------------------------------------------------------

def getNextCombo(I, n):

    r = len(I) - 1

    if (I[r] < n - 1):
        I[r] = I[r] + 1
        return I
    else:
        for j in range(r, 0, -1):
            if (I[j - 1] < n - r + j - 2):
                I[j - 1] = I[j - 1] + 1;
                for s in range(j, r + 1, 1):
                    I[s] = I[j - 1] + s - (j - 1)
                return I

        return []

    return I


# ---------------------------------------------------------------------

