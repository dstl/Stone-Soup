import numpy
import datetime
from ..types.association import AssociationSet
import warnings


def assign2D(C, maximize=False):
    # ASSIGN2D:
    # Solve the two-dimensional assignment problem with a rectangular
    # cost matrix C, scanning row-wise. The problem being solved can
    # be formulated as minimize (or maximize):
    #       \sum_{i=1}^{numRow}\sum_{j=1}^{numCol}C_{i,j}*x_{i,j}
    # subject to:
    #       \sum_{j=1}^{numCol}x_{i,j}<=1 for all i
    #       \sum_{i=1}^{numRow}x_{i,j}=1 for all j
    #       x_{i,j}=0 or 1.
    # Assuming that numCol<=numRow. If numCol>numRow, then the
    # inequality and inequality conditions are switched. A modified
    # Jonker-Volgenant algorithm is used.
    #
    # INPUTS:
    # C     A numRowXnumCol 2D numpy array or matrix matrix that does
    #       not contain any NaNs. Forbidden assignments can be given costs
    #       of +Inf for minimization and -Inf for maximization.
    # maximize      A boolean value. If true, the minimization problem is
    #               transformed into a maximization problem. The default
    #               if this parameter is omitted or an empty matrix
    #               is passed is false.
    #
    # OUTPUTS:
    # gain  The sum of the values of the assigned elements in C. If the
    #       problem is infeasible, this is -1.
    # col4row       A length numRow numpy array where the entry in each
    #               element is an assignment of the element in that row to
    #               a column. 0 entries signify unassigned rows. If the
    #               problem is infeasible, this is an empty matrix.
    # row4col       A length numCol numpy array where the entry in each
    #               element is an assignment of the element in that column
    #               to a row. 0 entries signify unassigned columns. If the
    #               problem is infeasible, this is an empty matrix.
    #
    # If the number of rows is <= the number of columns, then every row is
    # assigned to one column; otherwise every column is assigned to one
    # row. The assignment minimizes the sum of the assigned elements (the
    # gain). During minimization, assignments can be forbidden by placing
    # Inf in elements. During maximization, assignment can be forbidden by
    # placing -Inf in elements. The cost matrix can not contain any -Inf
    # elements during minimization nor any +Inf elements during
    # maximization to try to force an assignment. If no complete
    # assignment can be made with finite cost, then gain, col4row, and
    # row4col are returned as numpy.empty(0) values.
    #
    # The algorithm is described in detail in [1] and [2]. This is a Python
    # translation of the C and Matlab functions in the
    # Tracker Component Library of
    # https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary
    #
    # EXAMPLE 1:
    # import numpy
    # import assignAlgs
    # Inf=numpy.inf
    # C=numpy.array([[Inf,   2, Inf, Inf, 3],
    #              [   7, Inf,  23, Inf, Inf],
    #              [  17,  24, Inf, Inf, Inf],
    #              [ Inf,   6,  13,  20, Inf]])
    # maximize=False
    # gain, col4row, row4col=assignAlgs.assign2D(C,maximize)
    # print(gain)
    # One will get an optimal assignment having a gain of 47
    #
    # EXAMPLE 2:
    # This is the example used in [3]. Here, we demonstrate how to form
    # assignment tuples (index from 1, not 0) from col4row.
    # import numpy
    # import assignAlgs
    # C=numpy.array([[7,   51,  52,  87,  38,  60,  74,  66,   0,  20],
    #               [50,  12,   0,  64,   8,  53,   0,  46,  76,  42],
    #               [27,  77,   0,  18,  22,  48,  44,  13,   0,  57],
    #               [62,   0,   3,   8,   5,   6,  14,   0,  26,  39],
    #               [0,   97,   0,   5,  13,   0,  41,  31,  62,  48],
    #               [79,  68,   0,   0,  15,  12,  17,  47,  35,  43],
    #               [76,  99,  48,  27,  34,   0,   0,   0,  28,   0],
    #               [0,   20,   9,  27,  46,  15,  84,  19,   3,  24],
    #               [56,  10,  45,  39,   0,  93,  67,  79,  19,  38],
    #               [27,   0,  39,  53,  46,  24,  69,  46,  23,   1]])
    # maximize=False
    # gain, col4row, row4col=assignAlgs.assign2D(C,maximize)
    # tuples=numpy.empty((2,10),dtype=int)
    # for curRow in range(0,10):
    #    tuples[0,curRow]=curRow+1
    #    tuples[1,curRow]=col4row[curRow]+1
    # print(gain)
    # print(tuples)
    # One will see that the gain is 0 and the assigned  tuples match what
    # is in [3]. However, the assigned tuples is NOT obtained by attaching
    # col4row to row4col.
    #
    # REFERENCES:
    # [1]   D. F. Crouse, "On Implementing 2D Rectangular Assignment
    #       Algorithms," IEEE Transactions on Aerospace and Electronic
    #       Systems, vol. 52, no. 4, pp. 1679-1696, Aug. 2016.
    # [2]   D. F. Crouse, "Advances in displaying uncertain estimates of
    #       multiple targets," in Proceedings of SPIE: Signal Processing,
    #       Sensor Fusion, and Target Recognition XXII, vol. 8745,
    #       Baltimore, MD, Apr. 2013.
    # [3]   Murty, K. G. "An algorithm for ranking all the assignments in
    #       order of increasing cost," Operations Research, vol. 16, no. 3,
    #       pp. 682-687, May-Jun. 1968.
    #
    # May 2018 David F. Crouse, Naval Research Laboratory, Washington D.C.
    # (UNCLASSIFIED) DISTRIBUTION STATEMENT A. Approved for public release.
    # This work was supported by the Office of Naval Research through the
    # Naval Research Laboratory 6.1 Base Program

    numRow = C.shape[0]
    numCol = C.shape[1]
    totalNumElsInC = C.size

    didFlip = False

    if numCol > numRow:
        C = C.T
        temp = numRow
        numRow = numCol
        numCol = temp
        didFlip = True

    # The cost matrix must have all non-negative elements for the
    # assignment algorithm to work. This forces all of the elements to be
    # positive. The delta is added back in when computing the gain in the
    # end.
    if not maximize:
        CDelta = numpy.inf
        idxs = numpy.unravel_index([i for i in range(totalNumElsInC)], C.shape)
        for i in range(0, totalNumElsInC):
            idx = (idxs[0][i], idxs[1][i])
            if C[idx] < CDelta:
                CDelta = C[idx]

        # If C is all positive, do not shift.
        if CDelta > 0:
            CDelta = 0

        for i in range(0, totalNumElsInC):
            idx = (idxs[0][i], idxs[1][i])
            C[idx] = C[idx] - CDelta

    else:
        CDelta = -numpy.inf
        idxs = numpy.unravel_index([i for i in range(totalNumElsInC)], C.shape)
        for i in range(0, totalNumElsInC):
            idx = (idxs[0][i], idxs[1][i])
            if C[idx] > CDelta:
                CDelta = C[idx]

        # If C is all negative, do not shift.
        if CDelta < 0:
            CDelta = 0

        for i in range(0, totalNumElsInC):
            idx = (idxs[0][i], idxs[1][i])
            C[idx] = -C[idx] + CDelta

    CDelta = CDelta * numCol

    gain, col4row, row4col = assign2DBasic(C)

    if gain == -1:
        # The problem is infeasible
        emptyMat = numpy.empty(0)
        return emptyMat, emptyMat, emptyMat
    else:
        # The problem is feasible. Adjust for the shifting of the elements
        # in C.
        if not maximize:
            gain = gain + CDelta
        else:
            gain = -gain + CDelta

    # If a transposed matrix was used
    if didFlip:
        temp = col4row
        col4row = row4col
        row4col = temp

    return gain, col4row, row4col


def assign2DBasic(C):
    numRow = C.shape[0]
    numCol = C.shape[1]

    col4row = numpy.full(numRow, -1, dtype=int)
    row4col = numpy.full(numCol, -1, dtype=int)
    u = numpy.zeros(numCol)
    v = numpy.zeros(numRow)

    ScannedColIdx = numpy.empty(numCol, dtype=int)
    pred = numpy.empty(numRow, dtype=int)
    Row2Scan = numpy.empty(numRow, dtype=int)
    shortestPathCost = numpy.empty(numRow)

    for curUnassignedCol in range(0, numCol):
        # First, find the shortest augmenting path starting at
        # curUnassignedCol.

        # Mark everything as not yet scanned. A 1 will be placed in each
        # row entry as it is scanned.
        numColsScanned = 0
        scannedRows = numpy.zeros(numRow, dtype=bool)

        for curRow in range(0, numRow):
            Row2Scan[curRow] = curRow
            # Initially, the cost of the shortest path to each row is not
            # known and will be made infinite.
            shortestPathCost[curRow] = numpy.inf

        # All rows need to be scanned
        numRow2Scan = numRow
        # pred will be used to keep track of the shortest path.

        # sink will hold the final index of the shortest augmenting path.
        # If the problem is not feasible, then sink will remain -1.
        sink = -1
        delta = 0
        curCol = curUnassignedCol

        while sink == -1:
            # Mark the current column as having been visited.
            ScannedColIdx[numColsScanned] = curCol
            numColsScanned = numColsScanned + 1

            minVal = numpy.inf
            for curRowScan in range(0, numRow2Scan):
                curRow = Row2Scan[curRowScan]

                reducedCost = \
                    delta + C[curRow, curCol] - u[curCol] - v[curRow]
                if reducedCost < shortestPathCost[curRow]:
                    pred[curRow] = curCol
                    shortestPathCost[curRow] = reducedCost

                if shortestPathCost[curRow] < minVal:
                    minVal = shortestPathCost[curRow]
                    closestRowScan = curRowScan

            if minVal == numpy.inf:
                # If the minimum cost row is not finite, then the
                # problem is not feasible.
                return -1, col4row, row4col

            # Change the index from the relative row index to the
            # absolute row index.
            closestRow = Row2Scan[closestRowScan]

            # Add the closest row to the list of scanned rows and
            # delete it from the list of rows to scan by shifting all
            # of the items after it over by one.
            scannedRows[closestRow] = True

            numRow2Scan = numRow2Scan - 1  # One fewer rows to scan.
            for curRow in range(closestRowScan, numRow2Scan):
                Row2Scan[curRow] = Row2Scan[curRow + 1]

            delta = shortestPathCost[closestRow]
            # If we have reached an unassigned column
            if col4row[closestRow] == -1:
                sink = closestRow
            else:
                curCol = col4row[closestRow]

        # Next, update the dual variables.
        # Update the first column in the augmenting path.
        u[curUnassignedCol] = u[curUnassignedCol] + delta

        # Update the rest of the columns in the augmenting path.
        # curCol starts from 1, not zero, so that it skips
        # curUnassignedCol.
        for curCol in range(1, numColsScanned):
            curScannedIdx = ScannedColIdx[curCol]
            u[curScannedIdx] = u[curScannedIdx] + delta \
                               - shortestPathCost[row4col[curScannedIdx]]

        # Update the rows in the augmenting path.
        for curRow in range(0, numRow):
            if scannedRows[curRow]:
                v[curRow] = v[curRow] - delta + shortestPathCost[curRow]

        # Remove the current node from those that must be assigned.
        curRow = sink
        curCol = -1
        while curCol != curUnassignedCol:
            curCol = pred[curRow]
            col4row[curRow] = curCol
            h = row4col[curCol]
            row4col[curCol] = curRow
            curRow = h

    # Determine the gain to return
    gain = 0
    for curCol in range(0, numCol):
        gain = gain + C[row4col[curCol], curCol]

    return gain, col4row, row4col

# LICENSE:
#
# The source code is in the public domain and not licensed or under
# copyright. The information and software may be used freely by the public.
# As required by 17 U.S.C. 403, third parties producing copyrighted works
# consisting predominantly of the material produced by U.S. government
# agencies must provide notice with such work(s) identifying the U.S.
# Government material incorporated and stating that such material is not
# subject to copyright protection.
#
# Derived works shall not identify themselves in a manner that implies an
# endorsement by or an affiliation with the Naval Research Laboratory.
#
# RECIPIENT BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE
# SOFTWARE AND ANY RELATED MATERIALS, AND AGREES TO INDEMNIFY THE NAVAL
# RESEARCH LABORATORY FOR ALL THIRD-PARTY CLAIMS RESULTING FROM THE ACTIONS
# OF RECIPIENT IN THE USE OF THE SOFTWARE.


def multidimensional_deconfliction(association_set, low_diff_warning=None):
    """Solves the Multidimensional Assignment Problem (MAP)

    The assignment problem becomes more complex when time is added as a dimension.
    This basic solution finds all the conflicts in an association set and then creates a
    matrix of sums of conflicts in seconds, which is then passed to assign2D to solve as a
    simple 2D assignment problem.  Therefore, each object will only ever be assigned to one other
    at any one time.  In the case of an association that only partially overlaps, the "weaker" one
    (the one eliminated by assign2D) will be trimmed until there is no conflict.

    Due to the possibility of more than two conflicting associations at the same time,
    this algorithm is recursive, but it is not expected many (if any) recursions will be required
    for most uses.

    Parameters
    ----------
    association_set: The :class:`AssociationSet` to de-conflict
    low_diff_warning: If the longest association between objects minus the shortest (in seconds) is
    less than this, a warning will be given.  This may occur if the range of times covered includes
    only a low number of seconds.


    Returns
    -------
    : :class:`AssociationSet`
        The association set without contradictory associations
    """
    objects = list(association_set.object_set)
    length = len(objects)
    if length <= 1:
        return association_set
    totals = numpy.zeros((length, length))  # Time objects i and j are associated for in total
    association_on = numpy.full((length, length), False)
    association_start = numpy.full((length, length), datetime.datetime.min)
    key_times = association_set.key_times
    for time in key_times:
        associations_to_end = []
        for i in range(length):
            for j in range(length):
                if association_on[i][j]:
                    associations_to_end.append({i, j})
        time_ass_set = association_set.associations_at_timestamp(time)
        for association in time_ass_set:
            if len(association.objects) != 2:
                raise ValueError("Supplied set must only contain pairs of associated objects")
            obj_indices = [objects.index(list(association.objects)[0]),
                           objects.index(list(association.objects)[1])]
            if not association_on[obj_indices[0], obj_indices[1]]:
                association_on[obj_indices[0], obj_indices[1]] = True
                association_start[obj_indices[0], obj_indices[1]] = time
            elif time != max(association.time_range.key_times):
                associations_to_end.remove({obj_indices[0], obj_indices[1]})
        for inds in associations_to_end:
            print("loop")
            association_on[inds[0], inds[1]] = False
            totals[inds[0], inds[1]] += (time -
                                         association_start[inds[0], inds[1]]).total_seconds()
        association_start = _make_symmetric(association_start)
        totals = _make_symmetric(totals)
        association_on = _make_symmetric(association_on)

    totals = numpy.rint(totals).astype(int)
    if low_diff_warning and numpy.max(totals) - numpy.min(totals) <= low_diff_warning:
        warnings.warn(f"Difference between longest association and shortest is low after rounding"
                      f"({numpy.max(totals) - numpy.min(totals)} seconds)")
    numpy.fill_diagonal(totals, 0)  # Don't want to count associations of an object with itself

    solved_2d = assign2D(totals, maximize=True)[1]
    winning_indices = []  # Pairs that are chosen by assign2D
    for i in range(length):
        winning_indices.append([i, solved_2d[i]])
    cleaned_set = AssociationSet()
    for winner in winning_indices:
        print(winner)
        assoc = association_set.associations_including_objects({objects[winner[0]],
                                                                objects[winner[1]]})
        cleaned_set.add(assoc)
        association_set.remove(assoc)
    # Recursive step
    for assoc1 in association_set:
        for assoc2 in association_set:
            if conflicts(assoc1, assoc2):
                association_set = multidimensional_deconfliction(association_set)

    # At this point, none of association_set should conflict with one another
    for association in association_set:
        for winner in cleaned_set:
            if conflicts(association, winner):
                association.time_range.minus(winner.time_range)
        if association.time_range is not None:
            cleaned_set.add(association)

    return cleaned_set


def conflicts(assoc1, assoc2):
    if not hasattr(assoc1, 'time_range') or not hasattr(assoc2, 'time_range'):
        raise TypeError("Associations must have a time_range property")
    if assoc1.time_range.overlap(assoc2.time_range) and assoc1 != assoc2 \
            and len(assoc1.objects.intersection(assoc2.objects)) > 0:
        return True
    else:
        return False


def _make_symmetric(matrix):
    if isinstance(matrix[0, 0], datetime.datetime):
        ans = matrix
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[0]):
                if matrix[i, j] >= matrix[j, i]:
                    ans[j, i] = matrix[i, j]
                else:
                    ans[i, j] = matrix[j, i]
        return ans
    else:
        return numpy.tril(matrix) + numpy.triu(matrix.T, k=1)




