from ..types.state import StateMutableSequence


class GPPredictorWrapper:
    """
    A wrapper class for Gaussian Process (GP) transition models.

    This class is designed to handle GP transition models that
    may require track history for predictions.
    It checks whether the transition model has an attribute `requires_track` set to True,
    indicating the need for track history. If track history is required, the wrapper passes
    the track history to the transition model; otherwise, only the state is passed.

    Example usage: predictor = GPPredictorWrapper(KalmanPredictor)
    """

    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, prior, *args, **kwargs):
        requires_track = (
            getattr(self.predictor.transition_model, "requires_track", False)
            or self._check_requires_track(self.predictor.transition_model)
        )
        if requires_track:
            if not isinstance(prior, StateMutableSequence):
                raise TypeError('Prior must be StateMutableSequence')
            # Pass the track to the predictor
            return self.predictor.predict(prior.state, track=prior, *args, **kwargs)
        else:
            # Pass only the state to the predictor
            prior = prior.state if isinstance(prior, StateMutableSequence) else prior
            return self.predictor.predict(prior, *args, **kwargs)

    def _check_requires_track(self, model):
        """Recursively checks if any model in a combined transition model requires a track."""
        if hasattr(model, "model_list"):  # Check if it's a combined model
            return any(
                getattr(sub_model, "requires_track", False)
                or self._check_requires_track(sub_model) for sub_model in model.model_list
                )
        return False
