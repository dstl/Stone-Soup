from .kernel import Kernel
from .base import Property

class GaussianProcess:

    kernel: Kernel = Property(doc='The kernel to be used for the Gaussian Process')

    def optimise(self, hyperparameters):
        """Optimises specified kernel hyperparameters on training data. Hyperparameters are 
        updated in place.

        Parameters
        ----------
        hyperparameters: Union[str, List[str]]
            List of kernel hyperparameters to optimise.

        """
        # TODO: define cost function based on kernel's __call__ method which is then minimised
        # updates kernel parameters using self.kernel.update_parameters
        pass

    def get_posterior(self, x_train, y_train, x_test):
        # uses kernel to generate covariance matrices
        # implements main predictive GP equations, returning mean and cov of posterior distribution
        pass

    def states_to_train_data(self, states, timsetamp, window):
        """Takes a collection of `:class:~State`s (e.g. Track, list of Detections) and returns 
        data in the format of training data expected by the `:meth:get_posterior` method. 

        """
        # returns:
        #   x_train (list of times within window)
        #   y_train (list of state_vectors within window)
        pass