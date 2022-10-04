from stonesoup.base import Base, Property
from stonesoup.functions import sigmoid_function, reverse_sigmoid_function
import numpy as np
import matplotlib.pyplot as plt


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
    def create(cls, a, b, probability_at_a=DEFAULT_PROBABILITY_AT_A, probability_at_b=None):
        if probability_at_b is None:
            probability_at_b = 1 - probability_at_a

        sigmoid_input_at_a = reverse_sigmoid_function(probability_at_a)
        sigmoid_input_at_b = reverse_sigmoid_function(probability_at_b)

        std = (b - a) / (sigmoid_input_at_b - sigmoid_input_at_a)

        mean_a = a - sigmoid_input_at_a * std
        mean_b = b - sigmoid_input_at_b * std

        if not np.isclose(mean_a, mean_b):
            print(mean_a, "and", mean_b, "should be the same. They aren't")

        return cls(mean_a, std)


if __name__ == '__main__':

    a = 2
    b = 6
    plot_padding = (b - a)*0.5

    c2 = SigmoidDetectionModel.create(a, b, 0.99, 0.01)
    c2 = SigmoidDetectionModel.create(2, 6)

    x = np.arange(a-plot_padding, b+plot_padding, 0.1)
    y = [c2.probability_at_value(x_) for x_ in x]

    plt.plot(x, y, label="s")
    plt.grid(which='both')
    plt.show()


    five = 5
