import numpy as np

from stonesoup.models.measurement.detectability import SigmoidDetectionModel


def test1():
    c1 = SigmoidDetectionModel.create(2, 6)

    assert np.isclose(c1.mean, 4)

    assert np.isclose(c1.probability_at_value(-20), 1)
    assert np.isclose(c1.probability_at_value(2), c1.DEFAULT_PROBABILITY_AT_A)
    assert np.isclose(c1.probability_at_value(4), 0.5)
    assert np.isclose(c1.probability_at_value(6), 1 - c1.DEFAULT_PROBABILITY_AT_A)
    assert np.isclose(c1.probability_at_value(30), 0)

    x = 5
    probability_at_x = c1.probability_at_value(x)

    n_tests = 1000
    n_successful = sum(c1.is_detected(x) for _ in range(n_tests))
    test_probability_at_x = n_successful/n_tests

    assert np.isclose(probability_at_x, test_probability_at_x, atol=0.05)


def test2():
    a = 2
    b = 100
    prob_at_a = 0.77
    prob_at_b = 0.23
    mean = (a+b)/2
    low_limit = a-(10*abs(a-b))
    high_limit = b+(10*abs(a-b))

    c1 = SigmoidDetectionModel.create(a, b, prob_at_a)

    assert np.isclose(c1.mean, mean)

    assert np.isclose(c1.probability_at_value(low_limit), 1)
    assert np.isclose(c1.probability_at_value(a), prob_at_a)
    assert np.isclose(c1.probability_at_value(mean), 0.5)
    assert np.isclose(c1.probability_at_value(b), prob_at_b)
    assert np.isclose(c1.probability_at_value(high_limit), 0)

    xs = [2, 5, 20, 60, 100]
    for x in xs:
        probability_at_x = c1.probability_at_value(x)

        n_tests = 1000
        n_successful = sum(c1.is_detected(x) for _ in range(n_tests))
        test_probability_at_x = n_successful / n_tests

        assert np.isclose(probability_at_x, test_probability_at_x, atol=0.05)


def t_x(a, b, prob_at_a, prob_at_b, low_limit, high_limit, values_to_test_probability,
        mean=None, n_tests=1000, atol=0.05):

    c1 = SigmoidDetectionModel.create(a, b, prob_at_a, prob_at_b)

    if mean is not None:
        assert np.isclose(c1.mean, mean)
        assert np.isclose(c1.probability_at_value(mean), 0.5)

    assert np.isclose(c1.probability_at_value(low_limit), 1)
    assert np.isclose(c1.probability_at_value(a), prob_at_a)
    assert np.isclose(c1.probability_at_value(b), prob_at_b)
    assert np.isclose(c1.probability_at_value(high_limit), 0)

    for x in values_to_test_probability:
        probability_at_x = c1.probability_at_value(x)

        n_successful = sum(c1.is_detected(x) for _ in range(n_tests))
        test_probability_at_x = n_successful / n_tests

        assert np.isclose(probability_at_x, test_probability_at_x, atol=atol)


def test_other():

    t_x(a=2, b=100, prob_at_a=0.77, prob_at_b=0.23, low_limit=-1000, high_limit=1000,
        values_to_test_probability=[2, 5, 20, 60, 100], mean=51, n_tests=1000, atol=0.05)

    t_x(a=10, b=200, prob_at_a=0.8, prob_at_b=0.001, low_limit=-1000, high_limit=1000,
        values_to_test_probability=[2, 5, 20, 60, 100, 150, 250], mean=None, n_tests=1000,
        atol=0.05)

    t_x(a=-10, b=15, prob_at_a=0.1, prob_at_b=0.6, low_limit=150, high_limit=-500,
        values_to_test_probability=[-15, -6, -3, 0, 12, 20], mean=None, n_tests=1000, atol=0.05)

    t_x(a=-5, b=5, prob_at_a=0.001, prob_at_b=0.999, low_limit=20, high_limit=-20,
        values_to_test_probability=[-3, -1, 0, 1, 3], mean=None, n_tests=1000, atol=0.05)