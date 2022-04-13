import pytest

from ..detection import Detection, CompositeDetection
from ..hypothesis import SingleHypothesis, CompositeHypothesis, \
    CompositeProbabilityHypothesis, SingleProbabilityHypothesis
from ..prediction import StatePrediction, CompositePrediction


@pytest.fixture()
def sub_predictions1():
    return [StatePrediction([0]), StatePrediction([1, 2])]


@pytest.fixture()
def sub_predictions2():
    return [StatePrediction([3]), StatePrediction([5, 6])]


@pytest.fixture()
def composite_prediction1(sub_predictions1):
    return CompositePrediction(sub_predictions1)


@pytest.fixture()
def composite_prediction2(sub_predictions2):
    return CompositePrediction(sub_predictions2)


@pytest.fixture()
def sub_measurements1():
    return [Detection([7]), Detection([3])]


@pytest.fixture()
def sub_measurements2():
    return [Detection([2]), Detection([8])]


@pytest.fixture()
def composite_measurement1(sub_measurements1):
    return CompositeDetection(sub_measurements1)


@pytest.fixture()
def composite_measurement2(sub_measurements2):
    return CompositeDetection(sub_measurements2)


@pytest.fixture()
def sub_hypotheses1(sub_predictions1, sub_measurements1):
    return [
        SingleHypothesis(prediction=sub_prediction, measurement=sub_measurement)
        for sub_prediction, sub_measurement in zip(sub_predictions1, sub_measurements1)
    ]


@pytest.fixture()
def sub_hypotheses2(sub_predictions2, sub_measurements2):
    return [
        SingleHypothesis(prediction=sub_prediction, measurement=sub_measurement)
        for sub_prediction, sub_measurement in zip(sub_predictions2, sub_measurements2)
    ]


@pytest.fixture()
def composite_hypothesis1(composite_prediction1, composite_measurement1, sub_hypotheses1):
    return CompositeHypothesis(sub_hypotheses=sub_hypotheses1,
                               prediction=composite_prediction1,
                               measurement=composite_measurement1)


@pytest.fixture()
def composite_hypothesis2(composite_prediction2, composite_measurement2, sub_hypotheses2):
    return CompositeHypothesis(sub_hypotheses=sub_hypotheses2,
                               prediction=composite_prediction2,
                               measurement=composite_measurement2)


@pytest.fixture()
def sub_probability_hypotheses1(sub_predictions1, sub_measurements1):
    return [
        SingleProbabilityHypothesis(prediction=sub_prediction, measurement=sub_measurement,
                                    probability=0.5)
        for sub_prediction, sub_measurement in zip(sub_predictions1, sub_measurements1)
    ]


@pytest.fixture()
def sub_probability_hypotheses2(sub_predictions2, sub_measurements2):
    return [
        SingleProbabilityHypothesis(prediction=sub_prediction, measurement=sub_measurement,
                                    probability=0.2)
        for sub_prediction, sub_measurement in zip(sub_predictions2, sub_measurements2)
    ]


@pytest.fixture()
def composite_probability_hypothesis1(composite_prediction1, composite_measurement1,
                                      sub_probability_hypotheses1):
    return CompositeProbabilityHypothesis(sub_hypotheses=sub_probability_hypotheses1,
                                          prediction=composite_prediction1,
                                          measurement=composite_measurement1)


@pytest.fixture()
def composite_probability_hypothesis2(composite_prediction2, composite_measurement2,
                                      sub_probability_hypotheses2):
    return CompositeProbabilityHypothesis(sub_hypotheses=sub_probability_hypotheses2,
                                          prediction=composite_prediction2,
                                          measurement=composite_measurement2)
