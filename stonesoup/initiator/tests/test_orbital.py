import numpy as np
import pytest

from ...types.detection import Detection
from ..preliminaryorbitdetermination import GibbsInitiator


def test_gibbsinitiator():
    """Follow example 5.1 from [1]

    Using three position vectors initialise a track using the Gibbs
    initiator. Then check with the answer in the book. Note that [1]
    has rounding issues.

    Reference
    ---------
        1. Curtis, H. D. 2010, Orbital Mechanics for Engineering Students,
        Third Edition, Elsevier"""

    # Define three position vectors
    r1 = Detection(np.array([[-294.32], [4265.1], [5986.7]]))
    r2 = Detection(np.array([[-1365.5], [3637.6], [6346.8]]))
    r3 = Detection(np.array([[-2940.3], [2473.7], [6555.8]]))

    # The solution given in [1] is:
    v2 = np.array([[-6.2174], [-4.0122], [1.599]])
    out_cart = np.concatenate((r2.state_vector, v2))

    # set up the input triple
    inlist = [(r1, r2, r3)]
    # and then initiator
    ginitiator = GibbsInitiator(grav_parameter=3.986004418e5)

    # Track initialise
    out_track = ginitiator.initiate(inlist)

    print(out_track)

    assert(np.allclose(out_cart, out_track[0][1].state_vector, rtol=1e-5))


