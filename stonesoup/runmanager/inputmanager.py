from base import RunManager


class InputManager(RunManager):

    def set_stateVector(list_state_vector):
        """Get a list and return a state vector

        Parameters:
            list_state_vector: State vector list

        Returns:
            StateVector: state vector 
        """

        state_vector=list(list_state_vector)
        state_vector=np.c_[set_combinations[idx]]

        return list(state_vector)


    def set_int_float():
        raise NotImplementedError

    def set_covariance():
        raise NotImplementedError

    def set_bool():
        raise NotImplementedError

    def set_tuple():
        raise NotImplementedError

    def set_ndArray():
        raise NotImplementedError

    def set_timeDelta():
        raise NotImplementedError
    
    def set_deltaTime():
        raise NotImplementedError

    def set_coordinate_system():
        raise NotImplementedError

    def set_probability():
        raise NotImplementedError
