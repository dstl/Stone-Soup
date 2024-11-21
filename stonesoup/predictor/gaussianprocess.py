from ..kernel import Kernel
from . import Predictor
from ..base import Property

class GaussianProcessPredictor(Predictor):
    
    kernel: Kernel = Property(doc='The kernel to be used for the Gaussian Process')

    def predict(self, states, timestamp, window, optimise_hyperparameters=True, **kwargs):
        """Uses a Gaussian process to generate a prediction for the specified timestamp.

        Parameters
        ----------
        states : MutableSequence[:class:`~.State`]
            A mutable sequence of previous states (e.g. `:class:~Track`, list of
            `:class:~Detection`s)
        timestamp : :class:`datetime.datetime`
            Time for which the prediction will be made (:math:`k`)
        window: int
            The maximum number of previous states to use to train the Gaussian process
        optimise_hyperparameters: Union[bool, str, List[str]]
            Whether to optimise kernel hyperparameters. A string or list of strings can be used to
            specify individual hyperparameters. Strings must match the name of a 
            corresponding kernel property.
        **kwargs : various, optional

        Returns
        -------
        : :class:`~.GaussianStatePrediction`
            The predicted state :math:`\mathbf{x}_{k|k-1}` and the predicted
            state covariance :math:`P_{k|k-1}`
        """

        gp = GaussianProcess(kernel=self.kernel)
        x_train, y_train = gp.states_to_train_data(states, timestamp, window)

        # TODO: figure out how to handle hyperparameters for optimisation

        means, covars = gp.posterior(x_train, y_train, timestamp)

        # create state prediction from means and covars
        # return state prediction

        pass