import numpy as np

from .base import MetricGenerator, MetricManager
from ..base import Property
from ..types import Clutter, TrueDetection, Metric, SingleTimeMetric, TimePeriodMetric, Detection, GroundTruthPath, Track




class GOSPAMetric(MetricGenerator):
	c = Property(float, doc='1<=p<infty, exponent.')
	p = Property(float, doc='c>0, cutoff distance.')
	alpha = Property(float, doc='0<alpha<=2, factor for the cardinality penalty. Recommended value 2 => Penalty on missed & false targets',
		default=2)

	measurement_matrix_truth = Property(np.ndarray, doc='Measurement matrix for the truth states to extract parameters to calculate distance over')
	measurement_matrix_meas = Property(np.ndarray, doc='Measurement matrix for the track states to extract parameters to calculate distance over')

	def compute_metric(self, manager, **kwargs):
		metric = self.process_datasets(manager.tracks, manager.groundtruth_paths)
		return metric


	def process_datasets(self, tracks_list, truth_paths_list):

		track_states = self.extract_states(tracks_list)
		truth_states = self.extract_states(truth_paths_list)
		return self.compute_over_time(track_states, truth_states)

	def extract_states(self, object_with_states):
		"""
		Extracts a list of states from a list of (or single) object containing states
		:param object_with_states:
		:return:
		"""

		state_list = []
		for element in list(object_with_states):

			if isinstance(element, Track):
				for state in element.states:
					state_list.append(state)

			elif isinstance(element, GroundTruthPath):
				for state in element.states:
					state_list.append(state)

			elif isinstance(element, Detection):
				state_list.append(element)

			else:
				raise ValueError(type(element), ' has no state extraction method')

		return state_list

	def compute_over_time(self, measured_states, truth_states):
		"""
		Compute the GOSPA metric at every timestep from a list of measured states and truth states
		:param measured_states: List of states created by a filter
		:param truth_states: List of truth states to compare against
		:return:
		"""

		# Make a list of all the unique timestamps used

		timestamps = []
		for state in (measured_states + truth_states):
			if state.timestamp not in timestamps:
				timestamps.append(state.timestamp)

		gospa_metrics = []

		for timestamp in timestamps:
			meas_states_inst = [state for state in measured_states if state.timestamp == timestamp]
			truth_states_inst = [state for state in truth_states if state.timestamp == timestamp]

			metric, truth_to_measured_assignment =  self.compute_gospa_metric(meas_states_inst, truth_states_inst, timestamp)
			single_time_gospa_metric = SingleTimeMetric(title='GOSPA Metric', value=metric, timestamp=timestamp, generator=self)
			gospa_metrics.append(single_time_gospa_metric)

		if len(timestamps) == 1:
			return gospa_metrics[0]

		return TimePeriodMetric(title='GOSPA Metric',
								value=gospa_metrics,
								start_timestamp=min(timestamps),
								end_timestamp=max(timestamps),
								generator=self)


	def _compute_assignments(self, cost_matrix, max_iter):
		"""
		Compute assignments using Auction Algorithm.

		Inputs:
		:param cost_matrix: Matrix (size mxn) that denotes the cost of assigning
							mth truth state to each of the n measured states.
		:param max_iter: Maximum number of iterations to perform

		:return:

		measured_to_truth: Vector of size 1xn, which has indices of the
							truth objects or '-1' if unassigned.

		truth_to_measured: Vector of size 1xm, which has indices of the
							measured objects or '-1' if unassigned.

		opt_cost: Scalar value of the optimal assignment


		"""
		m_truth, n_measured = cost_matrix.shape
		# Index for objects that will be left un-assigned.
		unassigned_idx = -1

		opt_cost = 0.0
		measured_to_truth = -1*np.ones([1, m_truth], dtype=np.int64)
		truth_to_measured = -1*np.ones([1, n_measured], dtype=np.int64)

	
		if m_truth == 1:
			# Corner case 1: if there is only one truth state.
			opt_cost = np.max(cost_matrix)
			truth_to_measured[0, 0] = np.where(cost_matrix == opt_cost)[1]
			measured_to_truth[0, truth_to_measured[0, 0]] = 1

			return truth_to_measured, measured_to_truth, opt_cost
		
		if n_measured == 1:
			# Corner case 1: if there is only one measured state.
			opt_cost = np.max(cost_matrix)
			measured_to_truth[0, 0] = np.where(cost_matrix == opt_cost)[1]
			truth_to_measured[0, measured_to_truth[0, 0]] = 1

			return truth_to_measured, measured_to_truth, opt_cost

		swap_dim_flag = False
		epsil = 1./np.max([m_truth, n_measured])

		if (n_measured < m_truth):
			# The implementation only works when
			# m_truth <= n_measured
			# So swap cost matrix
			cost_matrix = cost_matrix.transpose()
			m_truth, n_measured = cost_matrix.shape
			swap_dim_flag = True


		# Initial cost for each measured state
		c_measured = np.zeros([1, n_measured])
		k_iter = 0

		while (not np.all(truth_to_measured != unassigned_idx)):
			if (k_iter > max_iter):
				# Raise max iterations reached warning.
				break
			for i in range(m_truth):
				if(truth_to_measured[0,i] == unassigned_idx):
					# Unassigned truth object 'i' bids for the best
					# measured object j_star

					# Value for each measured object for truth 'i'
					val_i_j = np.sort(cost_matrix[i, :] - c_measured)[::-1]
					j = np.argsort(cost_matrix[i, :] - c_measured)[::-1]
			
					# Best measurement for truth 'i'
					j_star = j[0,0]


					# 1st and 2nd best value for truth 'i'
					v_i_j_star = val_i_j[0, 0]
					w_i_j_star = val_i_j[0, 1]

					# Bid for measured j_star
					if (w_i_j_star != -1.*np.inf):
						c_measured[0, j_star] = c_measured[0, j_star] + v_i_j_star - w_i_j_star + epsil
					else:
						c_measured[0, j_star] = c_measured[0, j_star] + v_i_j_star  + epsil


					# If j_star is unassigned
					if (measured_to_truth[0, j_star] != unassigned_idx):

						opt_cost = opt_cost - cost_matrix[measured_to_truth[0, j_star], j_star]
						truth_to_measured[0, measured_to_truth[0, j_star]] = unassigned_idx

					measured_to_truth[0, j_star] = i
					truth_to_measured[0, i] = j_star

					# update the cost of new assignment
					opt_cost = opt_cost + cost_matrix[i, j_star]
			k_iter += 1

		if (swap_dim_flag):
			tmp = measured_to_truth
			measured_to_truth = truth_to_measured
			truth_to_measured = tmp

		return truth_to_measured, measured_to_truth, opt_cost



	def _compute_base_distance(self, truth_state, measured_state):
		# Euclidean base distance between truth state and measured_state
		# is used in this implementation.

		return np.linalg.norm(
			self.measurement_matrix_truth @ truth_state.state_vector.__array__() -\
			self.measurement_matrix_meas @ measured_state.state_vector.__array__())

	def compute_gospa_metric(self, measured_states, truth_states, tstamp=None):
		'''
		:param track_states: list of state objects to be assigned to the truth
		:param truth_states: list of state objects for the truth points
		:param timestamp: timestamp at which the states occured. If none then selected from the list of ststes
		:return: 
			gospa_metric: Dictionary containing GOSPA metric for alpha = 2. GOSPA metric is divided
						into four components: distance, localisation, missed, and false.
						Note that 
						distance = (localisation + missed + false)^1/p
			truth_to_measured_assignment:

		'''
		gospa_metric = {'distance': 0,
			'localisation': 0,
			'missed': 0,
			'false': 0}
		truth_to_measured_assignment = []

		num_truth_states = len(truth_states)
		num_measured_states = len(measured_states)
		

		cost_matrix = np.zeros([num_truth_states, num_measured_states])

		# Compute cost matrix.
		for i in range(num_truth_states):
			for j in range(num_measured_states):
				cost_matrix[i, j] = np.min([self._compute_base_distance(truth_states[i], measured_states[j]), self.c])

		# Initialise output values
		num_missed = 0
		num_false = 0
		localisation = 0.0

		truth_to_track_assignment = []
		opt_cost = 0.0;

		dummy_cost = (self.c**self.p)/self.alpha


		if (num_truth_states == 0):
			# When truth states are empty all measured states are false
			opt_cost = -1.0*num_measured_states*dummy_cost
			num_false = opt_cost
		else:
			if (num_measured_states == 0):
				# When there are measured states are empty all truth states are missed
				opt_cost = -1.*num_truth_states*dummy_cost
				if (self.alpha == 2):
					self.missed = opt_cost

			else:
				# print(cost_matrix)
				# Use auction algorithm when both truth_states and measured_states are non-empty
				cost_matrix = -1.*np.power(cost_matrix, self.p)
				truth_to_measured_assignment, measured_to_truth_assignment, opt_cost_tmp = \
				self._compute_assignments(cost_matrix, 10*(num_truth_states * num_measured_states))

				# Initialize outputs.
				# gospa_metric['localisation'] = 0
				# gospa_metric['missed'] = 0
				# gospa_metric['false'] = 0
				# gospa_metric['distance'] = 0
				# Now use assignments to compute costs
				
				for i in range(num_truth_states):
					if truth_to_measured_assignment[0, i] != -1:
						opt_cost = opt_cost + cost_matrix[i, truth_to_measured_assignment[0, i]]
						if (self.alpha == 2):
							gospa_metric['localisation'] = gospa_metric['localisation'] +\
							cost_matrix[i, truth_to_measured_assignment[0,i]] *\
							np.double(cost_matrix[i, truth_to_measured_assignment[0,i]] > -1.*self.c**self.p)
							
							gospa_metric['missed'] = gospa_metric['missed'] - dummy_cost *\
							np.double(cost_matrix[i, truth_to_measured_assignment[0,i]] == -1.*self.c**self.p)

							gospa_metric['false'] = gospa_metric['false'] - dummy_cost *\
							np.double(cost_matrix[i, truth_to_measured_assignment[0,i]] == -1.*self.c**self.p)
					else:
						opt_cost = opt_cost - dummy_cost
						if (self.alpha == 2):
							gospa_metric['missed'] = gospa_metric['missed'] - dummy_cost


				opt_cost = opt_cost - np.sum(measured_to_truth_assignment == 0) * dummy_cost
				if (self.alpha == 2):
					gospa_metric['false'] = gospa_metric['false'] - np.sum(measured_to_truth_assignment == 0)*dummy_cost


		gospa_metric['distance'] = np.power((-1.*opt_cost), 1/self.p)
		gospa_metric['localisation'] = -1.*gospa_metric['localisation']
		gospa_metric['missed'] = -1.*gospa_metric['missed']
		gospa_metric['false'] = -1.*gospa_metric['false']

		return gospa_metric, truth_to_measured_assignment
