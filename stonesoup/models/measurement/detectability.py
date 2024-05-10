import matplotlib.pyplot as plt
import numpy as np

from stonesoup.base import Base, Property
from stonesoup.functions import sigmoid_function, reverse_sigmoid_function


class SigmoidDetectionModel(Base):
    DEFAULT_PROBABILITY_AT_A = 0.9

    mean: float = Property()
    deviation: float = Property()

    def probability_at_value(self, x):
        z = (x-self.mean)/self.deviation
        return sigmoid_function(z)

    def is_detected(self, x):
        return np.random.rand() <= self.probability_at_value(x)

    @classmethod
    def create(cls, a: float, b: float,
               probability_at_a=DEFAULT_PROBABILITY_AT_A, probability_at_b=None):

        if probability_at_b is None:
            probability_at_b = 1 - probability_at_a

        sigmoid_input_at_a = reverse_sigmoid_function(probability_at_a)
        sigmoid_input_at_b = reverse_sigmoid_function(probability_at_b)

        (std, mean) = np.polyfit([sigmoid_input_at_a, sigmoid_input_at_b], [a, b], 1)

        return cls(mean, std)

    def plot(self, limit=0.001, block=False):
        min_x = reverse_sigmoid_function(limit)*self.deviation + self.mean
        max_x = reverse_sigmoid_function(1-limit)*self.deviation + self.mean

        x = np.linspace(min_x, max_x)
        y = [self.probability_at_value(x_) for x_ in x]

        plt.plot(x, y)
        plt.grid(which='both')
        plt.ylabel("Probability at x")
        plt.xlabel("x")
        plt.show(block=block)
