import pytest
import numpy as np
from datetime import timedelta

from ..orbital import stumpff_c, stumpff_s, universal_anomaly_newton, \
    lagrange_coefficients_from_universal_anomaly
from ...types.array import StateVector


@pytest.mark.parametrize(
    "z, outs, outc",
    [
        (0, 1/6, 1/2),
        (np.pi**2, 0.10132118364233778, 0.20264236728467555),
        (-(np.pi**2), 0.2711433813983066, 1.073189242960177),
    ]
)
def test_stumpff(z, outs, outc):
    """Test the Stumpf functions"""
    assert np.isclose(stumpff_s(z), outs, rtol=1e-10)
    assert np.isclose(stumpff_c(z), outc, rtol=1e-10)


def test_universal_anomaly_and_lagrange():
    """Test the computation of the universal anomaly. Also test the computation of the Lagrange
    coefficients. Follows example 3.7 in [1]_.

    References
    ----------
    .. [1] Curtis H.D. 2010, Orbital mechanics for engineering students, 3rd Ed., Elsevier

    """

    # Answers
    chi_is = 253.53  # km^0.5
    f_is = -0.54128
    g_is = 184.32  # s^{-1}
    fdot_is = -0.00055297  # s^{-1}
    gdot_is = -1.6592

    # Parameters
    bigg = 3.986004418e5  # in km^{-3} rather than m^{-3}
    start_at = StateVector([7000, -12124, 0, 2.6679, 4.6210, 0])
    deltat = timedelta(hours=1)

    assert np.isclose(universal_anomaly_newton(start_at, deltat, grav_parameter=bigg),
                      chi_is, atol=1e-2)

    f, g, fdot, gdot = lagrange_coefficients_from_universal_anomaly(start_at, deltat,
                                                                    grav_parameter=bigg)
    assert np.isclose(f, f_is, rtol=1e-4)
    assert np.isclose(g, g_is, rtol=2e-3)  # Seems a bit loose - is the textbook wrong?
    assert np.isclose(fdot, fdot_is, rtol=1e-4)
    assert np.isclose(gdot, gdot_is, rtol=1e-4)
