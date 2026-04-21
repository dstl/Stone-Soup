from .base import Reducer


class IdentityReducer(Reducer):
    def reduce(self, states, timestamp):
        states = self.calculate_likelihood(states, timestamp)
        return states
