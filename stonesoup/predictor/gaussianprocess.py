from ..types.state import StateMutableSequence

class GPPredictor:
    """Wrapper class for GP transition models."""
    
    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, prior, *args, **kwargs):
        requires_track = getattr(self.predictor.transition_model, "requires_track", False) or self._check_requires_track(self.predictor.transition_model)
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
            return any(getattr(sub_model, "requires_track", False) or self._check_requires_track(sub_model) for sub_model in model.model_list)
        return False
