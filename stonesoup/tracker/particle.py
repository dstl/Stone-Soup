# -*- coding: utf-8 -*-
import copy

import numpy as np

from .base import Tracker
from ..base import Property
from ..detector import Detector
from ..mixturereducer import MixtureReducer
from ..models import MeasurementModel
from ..predictor import Predictor
from ..types import GaussianState, Track


class SamplingImportanceResamplingParticleFilter(Tracker):
    """A SIR (Sampling Importance Resampling) particle filter for one target

    This is a fairly simple particle filter as the importance sampling density (where the particles are prediced to) is
    independent of the detection.

    Track an object using StoneSoup components.
    """
    detector = Property(
        Detector, doc="Detector used to generate detection objects.")
    predictor = Property(
        Predictor, doc="Predictor used to predict new state of the particle.")
    initial_state = Property(
        GaussianState, doc="Initial mean of the particles")
    resampler = Property(
        MixtureReducer, doc="Resampler used to prevent particle degeneracy")
    measurement_model = Property(
        MeasurementModel, doc="Measurement Model")
    number_particles = Property(
        int, default=2000, doc="Number of particles")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracks = set()

    def tracks_gen(self):

        track = None

        for time, detections in self.detector.detections_gen():

            if track is None:
                state = copy.deepcopy(self.initial_state)
                if state.timestamp is None:
                    state.timestamp = time
                track = Track()
                self.tracks.add(track)
                particles = [
                    GaussianParticle(
                        state.state_vector,
                        np.zeros([state.state_vector.shape[0]] * 2),
                        1 / self.number_particles,
                        timestamp=state.timestamp)
                    for _ in range(self.number_particles)]
                # Initialise particles
                for particle in particles:
                    # Flatten then convert back to column vector because
                    # pdf only takes 1d arrays but we want our state_vector
                    # to be a column
                    particle.state_vector = np.random.multivariate_normal(
                        particle.state_vector.ravel(),
                        state.covar).reshape((-1, 1))
            else:
                particles = track.particles

                # Draw the new states of the particles
                # Should this be performed on the initial particles or not?
                for particle in particles:
                    # Find the estimated new state of the particle
                    new_state = self.predictor.predict_state(
                        particle, timestamp=time)
                    # Draw from a distribution with the covariance of the
                    # motion model
                    new_state_drawn = np.random.multivariate_normal(
                        new_state.state_vector.ravel(),
                        self.predictor.transition_model.covar(
                            time - particle.timestamp)
                        ).reshape((-1, 1))
                    particle.state_vector = new_state_drawn
                    particle.timestamp = time



            if detections:
                for detection in detections:
                    for particle in particles:
                        particle.weight += self.measurement_model.pdf(
                            detection.state_vector, particle.state_vector)

                # Normalise the weights
                sum_w = sum(i.weight for i in particles)
                if sum_w == 0:
                    raise RuntimeError(
                        'Sum of weights is equal to zero; track lost')
                for particle in particles:
                    particle.weight /= sum_w

                particles = self.resampler.resample(particles)

            # Find the average state vector for current estimate
            # Don't need to do weighted mean as the weights are normalised
            state_matrix = np.concatenate(
                [i.state_vector for i in particles], axis=1)
            track.states.append(
                GaussianState(
                    np.mean(state_matrix, axis=1, keepdims=True),
                    np.cov(state_matrix),
                    timestamp=time))
            track.particles = particles

            yield {track}
