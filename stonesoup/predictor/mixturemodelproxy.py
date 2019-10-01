# -*- coding: utf-8 -*-
from .base import Predictor
from ..updater.base import Updater
import functools


def add_mixture_capability(cls1):
    """Adds Prepost Capability to the dosomething() method of the class cls1.
       Uses subtyping """

    if issubclass(cls1, Predictor):
        @functools.wraps(cls1.predict)
        def predict(self, prior, *args, **kwargs):
            state = self.convert2local_state(prior)
            out = cls1.predict(self, state, *args, **kwargs)
            out = self.convert2common_state(out)
            return out

        name = "Mixture" + cls1.__name__
        new_dict = {"predict": predict}
        newclass = type(name, (cls1,), new_dict)
    elif issubclass(cls1, Updater):
        @functools.wraps(cls1.predict_measurement)
        def predict_measurement(self, prior, *args, **kwargs):
            state = self.convert2local_state(prior)
            out = cls1.predict_measurement(self, state, *args, **kwargs)
            out = self.convert2common_state(out)
            return out

        name = "Mixture" + cls1.__name__
        newclass = type(name, (cls1,), {
                "predict_measurement": predict_measurement})
    else:
        raise ValueError("Class needs to be a Predictor or Updater")

    return newclass
