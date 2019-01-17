import numpy as np

from stonesoup.functions import jacobian, dayOfTheWeek


def test_jacobian():
    """ jacobian function test """

    # State related variables
    state_mean = np.array([[3.0], [1.0]])

    def f(x):
        return np.array([[1, 1], [0, 1]])@x

    jac = jacobian(f, state_mean)
    jac = jac  # Stop flake8 unused warning


def test_jacobian2():
    """ jacobian function test """

    # Sample functions to compute Jacobian on
    def fun(x):
        """ function for testing scalars i.e. scalar input, scalar output"""
        return 2*x**2

    def fun1d(vec):
        """ test function with vector input, scalar output"""
        out = 2*vec[0]+3*vec[1]
        return out

    def fun2d(vec):
        """ test function with 2d input and 2d output"""
        out = np.empty((2, 1))
        out[0] = 2*vec[0]**2 + 3*vec[1]**2
        out[1] = 2*vec[0]+3*vec[1]
        return out
        x = 3
        jac = jacobian(fun, x)
        assert jac == 4*x

    x = np.array([[1], [2]])
    # Tolerance value to use to test if arrays are equal
    tol = 1.0e-2

    jac = jacobian(fun1d, x)
    T = np.array([2.0, 3.0])

    FOM = np.where(np.abs(jac-T) > tol)
    # Check # of array elements bigger than tol
    assert len(FOM[0]) == 0

    jac = jacobian(fun2d, x)
    T = np.array([[4.0*x[0], 6*x[1]],
                  [2, 3]])
    FOM = np.where(np.abs(jac - T) > tol)
    # Check # of array elements bigger than tol
    assert len(FOM[0]) == 0

    return


def test_Friday():
    day = dayOfTheWeek(5)
    assert day == "Friday"


def test_Wednesday():
    """ Test comment """
    assert dayOfTheWeek(3) == "Wednesday"


def test_tuesday():
    assert dayOfTheWeek(2) == "Tuesday"


def test_dayOfTheWeek():
    """ Tests the day of the week liverpool-walk tutorial """
    number = 1
    day = dayOfTheWeek(number)
    assert day == "Monday"


def testdayoftheweek():
    assert dayOfTheWeek(4) == "Thursday"


def test_Saturday(input=None):
    day = dayOfTheWeek(6)
    assert day == "Saturday"


def test_dayOfTheWeek():

    assert dayOfTheWeek(7) == "Sunday"


