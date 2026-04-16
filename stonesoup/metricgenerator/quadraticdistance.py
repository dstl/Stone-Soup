from itertools import chain, groupby

import scipy.stats
from ..types.time import TimeRange
from ..types.metric import SingleTimeMetric, TimeRangeMetric
from ..types.groundtruth import GroundTruthState
from ..types.state import State, StateMutableSequence, TaggedWeightedGaussianState
from ..types.track import Track
from ..base import Property
from .base import MetricGenerator
import numpy as np
from scipy.stats import multivariate_normal
from datetime import datetime, timedelta


class QuadraticDistance(MetricGenerator):
    """

    """
    kernel: str = Property(doc="kernel parametrisation of the quadratic distance.")
    kernel_parameters: dict = Property(doc="Required parameters for the kernel of the quadratic distance. Set of name-value pairs")

    generator_name: str = Property(doc="Unique identifier to use when accessing generated metrics "
                                       "from MultiManager",
                                   default='qdist_generator')
    tracks_key: str = Property(doc="Key to access set of tracks added to MetricManager",
                               default='tracks')
    truths_key: str = Property(doc="Key to access set of ground truths added to MetricManager. "
                                   "Or key to access a second set of tracks for track-to-track"
                                   " metric generation",
                               default='groundtruth_paths')
    
    def __init__(self, *args, **kwargs):
        self.tracks_type = None
        self.truths_type = None
        self.state_dim = None
        super().__init__(*args, **kwargs)

    def compute_metric(self, manager):
        tracks_states, self.tracks_type, self.state_dim = self.extract_states(manager.states_sets[self.tracks_key])
        truths_states, self.truths_type, self.state_dim = self.extract_states(manager.states_sets[self.truths_key])
        return self.compute_over_time(tracks_states, truths_states)

    @staticmethod
    def extract_states(object_with_states):
        """
        Extracts a list of states from a list of (or single) objects
        containing states. This method is defined to handle :class:`~.StateMutableSequence`
        and :class:`~.State` types.

        Parameters
        ----------
        object_with_states: object containing a list of states
            Method of state extraction depends on the type of the object

        Returns
        -------
        : list of :class:`~.State`
        """

        state_list = StateMutableSequence()
        ids = []
        for i, element in enumerate(list(object_with_states)):
            # extract state type
            for el in element:
                    state_dim_set = {np.array(getattr(el, 'mean', el.state_vector)).flatten().shape[0]}
                    state_type_set = {type(el)}
                    if len(state_type_set) == 1 and len(state_dim_set) == 1:
                        state_type = list(state_type_set)[0]
                        state_dim = list(state_dim_set)[0]
                    else:
                        raise ValueError('elements of the state sequence must share the same state type and state vectors must be of the same dimension.')
                    
            if isinstance(element, StateMutableSequence):
                states = list(element.last_timestamp_generator())
                state_list.extend(states)
                ids.extend([i]*len(states))
                
            elif isinstance(element, State):
                state_list.append(element)
                ids.extend([i])
            
            else:
                raise ValueError(
                    "{!r} has no state extraction method".format(element))

        return state_list, state_type, state_dim
    
    def compute_over_time(self, measured_states, truth_states):
        """
        Computes the qudratic distance over time between the measured and ture states passed to this function.

        Parameters
        ----------
        measured_states: list of states
        truth_states: list of states

        Returns
        -------
        :class: `.~TimeRangeMetric`
        """
        
        # Make a sorted list of all the unique timestamps used
        timestamps = sorted({
            state.timestamp
            for state in chain(measured_states, truth_states)})

        quadratic_distances = []

        for timestamp in timestamps:
            meas_points = [state
                           for state in measured_states
                           if state.timestamp == timestamp]
            truth_points = [state
                            for state in truth_states
                            if state.timestamp == timestamp]
            quadratic_distances.append(
                self.compute_Q_distance(truth_points, meas_points))

        # If only one timestamp is present then return a SingleTimeMetric
        if len(timestamps) == 1:
            return quadratic_distances[0]
        else:
            return TimeRangeMetric(
                title='Quadratic distances',
                value=quadratic_distances,
                time_range=TimeRange(min(timestamps), max(timestamps)),
                generator=self)

    def vectorised_gaussian_eval(self, ret_sum, normalise, **kwargs):
        '''
        Batch evaluates weighted gaussian densities of the form 
        $\omega_i\omega_j\mathcal{N}(m_i; m_j, R + P_i + P_j), i = \{1, \dots, N\}, , j = \{1, \dots, M\}$
        where using the keyword arguements the following parameters may be given:

        Parameters
        ----------
        w1: Corresponds to a list of weights given by $\omega_i$ in the above expression.
        w2: Corresponds to a list of weights given by $\omega_j$ in the above expression.
        m1: Corresponds to a list of means given by $m_i$ in the above expression.
        m2: Corresponds to a list of means given by $m_j$ in the above expression.
        const_cov: Corresponds to a covariance matrix given by $R$ in the above expression.
        var_cov1: Corresponds to a list of covariance matrices given by $P_i$ in the above expression.
        var_cov2: Corresponds to a list of covariance matrices given by $P_j$ in the above expression.

        dim: The dimension of the means. This must be the same for both sets.

        ret_sum: If true, this flag causes the function to return the sum of the batch results. If false, an (N, M) matrix is returned.

        normalise: if true, batch evaluations will be computed according to the normalised Gaussian density, if false the density is unnormalised.


        Returns
        -------
        values: a (N, M) matrix of evaluated gaussian densities for each combination of i, j.
        '''
        allowed_keys = {"dim", "w1", "w2", "m1", "m2","const_cov", "var_cov1", "var_cov2"}

        unknown = set(kwargs) - allowed_keys
        if unknown:
            raise ValueError(f"Unknown parameter(s) for vectorised_gaussian_eval: {', '.join(unknown)}")

        means1 = kwargs.pop("m1")
        means2 = kwargs.pop("m2")
        len1 = len(means1)
        len2 = len(means2)

        dim = kwargs.pop("dim")

        if "w1" not in kwargs:
            w1 = np.ones((len1,))
        else:
            w1 = kwargs.pop("w1")

        if "w2" not in kwargs:
            w2 = np.ones((len2,))
        else:
            w2 = kwargs.pop("w2")

        if "const_cov" not in kwargs:
            const_cov = np.eye(dim)
        else:
            const_cov = kwargs.pop("const_cov")

        if "var_cov1" not in kwargs:
            var_cov1 = np.repeat(np.zeros((dim, dim))[None, :, :], len1, axis=0)
        else:
            var_cov1 = kwargs.pop("var_cov1")

        if "var_cov2" not in kwargs:
            var_cov2 = np.repeat(np.zeros((dim, dim))[None, :, :], len2, axis=0)
        else:
            var_cov2 = kwargs.pop("var_cov2")
        

        # compute all combinations of covariance matrices
        cov = (const_cov + var_cov1[None, :, :, :] + var_cov2[:, None, :, :]).reshape(-1, dim, dim)

        diffs = (means1[None, :, :] - means2[:, None, :]).reshape(-1, dim)
        inv_covs = np.linalg.inv(cov)
        det_covs = np.linalg.det(cov)
        flat_exp = np.einsum('ik,ikj,ij->i', diffs, inv_covs, diffs)
        evals = np.exp(-0.5*flat_exp) 

        if normalise:
            evals = evals / np.sqrt((2*np.pi)**dim * det_covs)

        const = (w1[:, None] * w2[None, :])

        values = np.multiply(const, evals.reshape(len1, len2))
        if ret_sum:
            return np.sum(values) 
        else:
            return values

    def compute_Q_distance(self, truth_states, track_states):
        """
        Computes the quadratic distance at a single instant in time.

        Parameters
        ----------
        truth_states: a list of states. these states must have a state_vector attribute but may also have a covar attribute.
        track_states: a list of states. these states must have a state_vector attribute but may also have a covar attribute.

        Returns
        -------
        :class: `.~SingleTimeMetric`
        """
        timestamps = {
            state.timestamp
            for state in chain(track_states, truth_states)}
        if len(timestamps) > 1:
            raise ValueError('All states must be from the same time to perform quadratic distance calculation.')
        
        #########################################
        ########### Gaussian kernel #############
        #########################################
        if self.kernel == 'Gaussian':

            # unpack kernel parameters
            allowed_keys = {'covariance'}

            if self.kernel_parameters is not None:
                unknown = set(self.kernel_parameters) - allowed_keys
                if unknown:
                    raise ValueError(f"Unknown parameter(s) for vectorised_gaussian_eval: {', '.join(unknown)}.")
                
                R = self.kernel_parameters['covariance']

                # check dimension symmetry and positive-definiteness
                if R.shape != (self.state_dim, self.state_dim) or not np.allclose(R, R.T, rtol=1e-10, atol=1e-10) or np.any(np.linalg.eigvals(R) < 0):
                    raise ValueError(f'The {self.kernel} kernel covariance matrix must be symmetric and positive-definite with shape ({self.state_dim}, {self.state_dim}).')
            else:
                raise ValueError(f'No covariance matrix was provided for the {self.kernel} kernel.')


            ## quadratic error calculations for all combinations of point and track comparison ##
            distance = np.inf # default value

            # quadratic error between point sets
            if (hasattr(self.truths_type, 'state_vector') and not hasattr(self.truths_type, 'covar')) and (hasattr(self.tracks_type, 'state_vector') and not hasattr(self.tracks_type, 'covar')):
                def quadratic_dist_pp(trth, trck, R):

                    Phi = len(trth)
                    Psi = len(trck)
                    
                    # set weights to one for non-weighted states
                    if hasattr(trth, 'weight'):
                        trth_weights = np.array([t.weight for t in trth])
                    else:
                        trth_weights = np.ones(Phi) 

                    if hasattr(trck, 'weight'):
                        trck_weights = np.array([t.weight for t in trck])
                    else:
                        trck_weights = np.ones(Psi) 

                    # extract state vectors
                    trth_means = np.array([np.asarray(getattr(trth[x], 'mean', trth[x].state_vector)) for x in range(Phi)])
                    trck_means = np.array([np.asarray(getattr(trck[x], 'mean', trck[x].state_vector)) for x in range(Psi)])

                    term1 = 0
                    term2 = 0
                    term3 = 0
                    if Phi > 0:
                        term1 = self.vectorised_gaussian_eval(ret_sum=True, normalise=False, dim=self.state_dim, w1=trth_weights, w2=trth_weights, m1=trth_means, m2=trth_means, const_cov=R)
                    if Phi > 0 and Psi > 0:
                        term2 = self.vectorised_gaussian_eval(ret_sum=True, normalise=False, dim=self.state_dim, w1=trth_weights, w2=trck_weights, m1=trth_means, m2=trck_means, const_cov=R)
                    if Psi > 0:
                        term3 = self.vectorised_gaussian_eval(ret_sum=True, normalise=False, dim=self.state_dim, w1=trck_weights, w2=trck_weights, m1=trck_means, m2=trck_means, const_cov=R)
                    
                    d = term1 - 2*term2 + term3
                    return d.item()
                
                distance = quadratic_dist_pp(truth_states, track_states, R)

            # quadratic error between a point set and a track set
            elif (hasattr(self.truths_type, 'state_vector') and not hasattr(self.truths_type, 'covar')) and (hasattr(self.tracks_type, 'state_vector') and hasattr(self.tracks_type, 'covar')):
                def quadratic_dist_pt(trth, trck, R):

                    Phi = len(trth)
                    Psi = len(trck)

                    # set weights to one for non-weighted states
                    if hasattr(trth, 'weight'):
                        trth_weights = np.array([t.weight for t in trth])
                    else:
                        trth_weights = np.ones(Phi) 

                    if hasattr(trck, 'weight'):
                        trck_weights = np.array([t.weight for t in trck])
                    else:
                        trck_weights = np.ones(Psi)    
                    
                    # extract state vectors
                    trth_means = np.array([np.asarray(getattr(trth[x], 'mean', trth[x].state_vector)) for x in range(Phi)])
                    trck_means = np.array([np.asarray(getattr(trck[x], 'mean', trck[x].state_vector)) for x in range(Psi)])

                    # extract covariances
                    trck_covs = np.array([trck[x].covar for x in range(Psi)])

                    term1 = 0
                    term2 = 0
                    term3 = 0
                    if Phi > 0:
                        term1 = self.vectorised_gaussian_eval(ret_sum=True, normalise=False, dim=self.state_dim, w1=trth_weights, w2=trth_weights, m1=trth_means, m2=trth_means, const_cov=R)
                    if Phi > 0 and Psi > 0:
                        term2 = np.sqrt((2*np.pi)**self.state_dim * np.linalg.det(R))*self.vectorised_gaussian_eval(ret_sum=True, normalise=True, dim=self.state_dim, w1=trth_weights, w2=trck_weights, m1=trth_means, m2=trck_means, const_cov=R, var_cov2=trck_covs)
                    if Psi > 0:
                        term3 = np.sqrt((2*np.pi)**self.state_dim * np.linalg.det(R))*self.vectorised_gaussian_eval(ret_sum=True, normalise=True, dim=self.state_dim, w1=trck_weights, w2=trck_weights, m1=trck_means, m2=trck_means, const_cov=R, var_cov1=trck_covs, var_cov2=trck_covs)

                    d = term1 - 2*term2 + term3
                    return d.item()
                
                distance = quadratic_dist_pt(truth_states, track_states, R)

            # quadratic error between track sets
            elif (hasattr(self.truths_type, 'state_vector') and hasattr(self.truths_type, 'covar')) and (hasattr(self.tracks_type, 'state_vector') and hasattr(self.tracks_type, 'covar')):
                def quadratic_dist_tt(trth, trck, R):
        
                    Phi = len(trth)
                    Psi = len(trck)

                    # set weights to one for non-weighted states
                    if hasattr(trth, 'weight'):
                        trth_weights = np.array([t.weight for t in trth])
                    else:
                        trth_weights = np.ones(Phi) 

                    if hasattr(trck, 'weight'):
                        trck_weights = np.array([t.weight for t in trck])
                    else:
                        trck_weights = np.ones(Psi)  
                    
                    # extract state vectors
                    trth_means = np.array([np.asarray(getattr(trth[x], 'mean', trth[x].state_vector)) for x in range(Phi)])
                    trck_means = np.array([np.asarray(getattr(trck[x], 'mean', trck[x].state_vector)) for x in range(Psi)])

                    # extract covariances
                    trth_covs = np.array([trth[x].covar for x in range(Phi)])
                    trck_covs = np.array([trck[x].covar for x in range(Psi)])

                    term1 = 0
                    term2 = 0
                    term3 = 0
                    if Phi > 0:
                        term1 = np.sqrt((2*np.pi)**self.state_dim * np.linalg.det(R))*self.vectorised_gaussian_eval(ret_sum=True, normalise=True, dim=self.state_dim, w1=trth_weights, w2=trth_weights, m1=trth_means, m2=trth_means, const_cov=R, var_cov1=trth_covs, var_cov2=trth_covs)
                    if Phi > 0 and Psi > 0:   
                        term2 = np.sqrt((2*np.pi)**self.state_dim * np.linalg.det(R))*self.vectorised_gaussian_eval(ret_sum=True, normalise=True, dim=self.state_dim, w1=trth_weights, w2=trck_weights, m1=trth_means, m2=trck_means, const_cov=R, var_cov1=trth_covs, var_cov2=trck_covs)
                    if Psi > 0:
                        term3 = np.sqrt((2*np.pi)**self.state_dim * np.linalg.det(R))*self.vectorised_gaussian_eval(ret_sum=True, normalise=True, dim=self.state_dim, w1=trck_weights, w2=trck_weights, m1=trck_means, m2=trck_means, const_cov=R, var_cov1=trck_covs, var_cov2=trck_covs)

                    d = term1 - 2*term2 + term3
                    return d.item()
                
                distance = quadratic_dist_tt(truth_states, track_states, R)

            else:
                raise ValueError('Inputs to the quadratic error must be sets of means or sets of weighted (or non-weighted) mean-covariance pairs.')

            return SingleTimeMetric(title='Quadratic Distance', value=distance,
                                timestamp=timestamps.pop(), generator=self)
        else:
            raise NotImplementedError(f'The Quadratic Distance with the {self.kernel} kernel parametrisation is not implemented.')

        
    

class MeanQuadraticError(QuadraticDistance):
    """
    Mean quadratic error of an estimator. the implementation is different for each choice of random object
    as the covariance is different for each point process parametrisation. the default case is where we 
    compute the mqe between the posterior point process and the truth. in this case, we need the posterior
    intensity (Gaussian mixture) and the truths (Dirac mixture). 
    """
    filter_data: dict = Property(doc="Point process model of the random object.")
    generator_name: str = Property(doc="Unique identifier to use when accessing generated metrics "
                                       "from MultiManager",
                                   default='mqe_generator')
    hypotheses_key: str = Property(doc="Key to access set of hypotheses added to MetricManager",
                               default='hypotheses')
    def compute_metric(self, manager):
        tracks_states, self.tracks_type, self.state_dim = self.extract_states(manager.states_sets[self.tracks_key])
        truths_states, self.truths_type, self.state_dim = self.extract_states(manager.states_sets[self.truths_key])
        hypotheses = manager.states_sets[self.hypotheses_key]
        return self.compute_over_time(tracks_states, hypotheses, truths_states)
    
    def compute_over_time(self, measured_states, hypotheses, truth_states):
        """
        
        """
        
        # Make a sorted list of all the unique timestamps used
        timestamps = sorted({
            state.timestamp
            for state in chain(measured_states, truth_states)})

        mqes = []

        for n, timestamp in enumerate(timestamps[1:]):
            truth_points = [state
                            for state in truth_states
                            if state.timestamp == timestamp]
            meas_points = [state
                           for state in measured_states
                           if state.timestamp == timestamp]
            mqes.append(
                self.compute_MQE(truth_points, meas_points, list(hypotheses)[n]))

        # If only one timestamp is present then return a SingleTimeMetric
        if len(timestamps) == 1:
            return mqes[0]
        else:
            return TimeRangeMetric(
                title='MQE values',
                value=mqes,
                time_range=TimeRange(min(timestamps), max(timestamps)),
                generator=self)
        
    def kernel_smoothed_covariance(self, estimator_states, hypotheses, updater, detection_probability, survival_probability, clutter_rate, R):
            '''
            Computes the covariance term of the covariance bias decomposition of the mean quadratic error.
            '''
            
            # sum of posterior weights
            update_weights_sum = 0
            for track in estimator_states:
                update_weights_sum += track.weight

            update_weights = []
            update_means = []
            update_covs = []
            for z, multi_hypothesis in enumerate(hypotheses[:-1]):
                # Initialise weight sum for measurement to clutter intensity
                weight_sum = 0
                # For every valid single hypothesis, update that component with
                # measurements and calculate new weight
                update_weights.append([])
                update_means.append([])
                update_covs.append([])
                for hypothesis in multi_hypothesis:
                    measurement_prediction = \
                        updater.predict_measurement(
                                hypothesis.prediction, hypothesis.measurement.measurement_model)
                    measurement = hypothesis.measurement
                    prediction = hypothesis.prediction
                    # Calculate new weight and add to weight sum
                    q = multivariate_normal.pdf(
                        measurement.state_vector.flatten(),
                        mean=measurement_prediction.mean.flatten(),
                        cov=measurement_prediction.covar
                    )
                    new_weight = detection_probability\
                        * prediction.weight * q * survival_probability
                    weight_sum += new_weight
                    # Perform single target Kalman Update
                    temp_updated_component = updater.update(hypothesis)

                    update_weights[z].append(new_weight/(weight_sum + clutter_rate))
                    update_means[z].append(temp_updated_component.mean)
                    update_covs[z].append(temp_updated_component.covar)
            
            detection_sum = 0
            for z, multi_hypothesis in enumerate(hypotheses[:-1]):
                for i in range(len(multi_hypothesis)):
                    for j in range(len(multi_hypothesis)):
                        detection_sum += update_weights[z][i]*update_weights[z][j]*multivariate_normal.pdf(np.array(update_means[z][i]).flatten(), np.array(update_means[z][j]).flatten(), R + update_covs[z][i] + update_covs[z][j])
                
            return update_weights_sum + detection_sum
    
    def compute_MQE(self, parameter_states, estimator_states, hypotheses):

        timestamps = {
            state.timestamp
            for state in chain(estimator_states, parameter_states)}
        if len(timestamps) > 1:
            raise ValueError('All states must be from the same time to perform quadratic distance calculation.')
        
        #########################################
        ########### Gaussian kernel #############
        #########################################
        if self.kernel == 'Gaussian':

            # unpack kernel specific parameters
            allowed_keys = {'covariance'}

            if self.kernel_parameters is not None:
                unknown = set(self.kernel_parameters) - allowed_keys
                if unknown:
                    raise ValueError(f"Unknown parameter(s) for vectorised_gaussian_eval: {', '.join(unknown)}.")
                
                R = self.kernel_parameters['covariance']

                # check dimension symmetry and positive-definiteness
                if R.shape != (self.state_dim, self.state_dim) or not np.allclose(R, R.T, rtol=1e-10, atol=1e-10) or np.any(np.linalg.eigvals(R) < 0):
                    raise ValueError(f'The {self.kernel} kernel covariance matrix must be symmetric and positive-definite with shape ({self.state_dim}, {self.state_dim}).')
            else:
                raise ValueError(f'No covariance matrix was provided for the {self.kernel} kernel.')
        else: 
            raise NotImplementedError(f'The Mean Quadratic Error with the {self.kernel} kernel parametrisation is not implemented.')
            
        #################################
        ######### GM-PHD Update #########
        #################################
        if self.filter_data['filter model'] == 'GMPHD':
            
            # unpack filter data
            survival_probability = self.filter_data['survival probability']
            detection_probability = self.filter_data['detection probability']
            clutter_rate = self.filter_data['clutter rate']
            updater = self.filter_data['updater']

            bias_squared_term = self.compute_Q_distance(parameter_states, estimator_states).value
            covariance_term = self.kernel_smoothed_covariance(estimator_states, hypotheses, updater, detection_probability, survival_probability, clutter_rate, R)
            distance = bias_squared_term + covariance_term

        else: 
            raise NotImplementedError(f'The Mean Quadratic Error for the {self.filter_data['filter model']} filter is not implemented.')
            

        return SingleTimeMetric(title='MQE', value=distance,
                                timestamp=timestamps.pop(), generator=self)