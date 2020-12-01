.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_Matlab_Wrapper.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_Matlab_Wrapper.py:


Matlab Wrapper Example
======================
This example looks at how the :class:`~.MatlabWrapper` class can be used to call MATLAB functions
in Stone Soup.

More specifically, we will show how to write a :class:`~.MatlabKalmanPredictor` class that
makes use of a custom MATLAB function to perform the prediction and compare its output to the
standard :class:`~.KalmanPredictor`.

Writing the MATLAB script
-------------------------
Below we show the MATLAB script :download:`kf_predict.m <../../examples/kf_predict.m>` used to
perform a standard Kalman Filter prediction step.

.. literalinclude:: ../../examples/kf_predict.m
   :language: matlab

Defining the MATLAB Kalman predictor
------------------------------------
Now that we have written our MATLAB function, we can proceed to define our custom Kalman
predictor class.

Since we want our custom Stone Soup object to interface with MATLAB we need to subclass the
:class:`~.MatlabWrapper`, so that our class inherits all the attributes and methods necessary to
interface with the `MATLAB Engine API for Python <https://uk.mathworks.com/help/matlab/matlab-engine-for-python.html>`__.

We also subclass the :class:`~.KalmanPredictor`. This is mostly done for convenience, since it
means that our class can inherit all the attributes and methods defined therein. As such, we only
need to override the ``predict()`` method so that it makes use of our MATLAB function, instead of
performing the computations in Python.


.. code-block:: default


    import numpy as np
    from functools import lru_cache

    from stonesoup.predictor.kalman import KalmanPredictor
    from stonesoup.types.prediction import GaussianStatePrediction
    from stonesoup.wrapper.matlab import MatlabWrapper


    class MatlabKalmanPredictor(KalmanPredictor, MatlabWrapper):
        """A standard Kalman predictor using MATLAB functions to prove that you can. """

        @lru_cache()
        def predict(self, prior, control_input=None, timestamp=None, **kwargs):

            # Get the prediction interval
            predict_over_interval = self._predict_over_interval(prior, timestamp)

            # Transition model parameters
            transition_matrix = self._transition_matrix(
                prior=prior, time_interval=predict_over_interval, **kwargs)
            transition_covar = self.transition_model.covar(
                time_interval=predict_over_interval, **kwargs)

            # Control model parameters
            control_matrix = self._control_matrix
            control_noise = self.control_model.control_noise
            control_input = control_input if control_input is not None else self.control_model.control_input()

            # Create MATLAB compatible arrays
            x = self.matlab_array(prior.state_vector)
            P = self.matlab_array(prior.covar)
            F = self.matlab_array(transition_matrix)
            Q = self.matlab_array(transition_covar)
            B = self.matlab_array(control_matrix)
            u = self.matlab_array(control_input)
            Qu = self.matlab_array(control_noise)

            # Call the custom kf_predict MATLAB function
            pred_mean, pred_covar = self.matlab_engine.kf_predict(x, P, F, Q, u, B, Qu, nargout=2)

            return GaussianStatePrediction(np.array(pred_mean), np.array(pred_covar), timestamp)









Using the MATLAB Kalman predictor
---------------------------------
We now proceed to make use of the predictor we have defined.

Initialise prior state and transition model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let us assume we have a target moving with nearly constant velocity in 2D. Therefore, our state
:math:`\mathrm{x}_k` is of the following following form:

.. math::
      \mathrm{x}_k = [x_k, \dot{x}_k, y_k, \dot{y}_k]

where :math:`x_k, y_k` denote the 2D positional coordinates and :math:`\dot{x}_k, \dot{y}_k`
denote the respective velocity on each dimension.


.. code-block:: default


    from datetime import datetime, timedelta
    from stonesoup.types.state import GaussianState
    from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel,
                                                    ConstantVelocity)

    # Define prior state
    timestamp_init = datetime.now()
    prior = GaussianState(state_vector=[[0.], [1.], [0.], [1.]],
                          covar=np.diag([1.5, 0.5, 1.5, 0.5]),
                          timestamp=timestamp_init)

    # Initialise a transition model
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(noise_diff_coeff=0.1), ConstantVelocity(noise_diff_coeff=0.1)])








Initialise our predictors
~~~~~~~~~~~~~~~~~~~~~~~~~
Let's now proceed to create our custom predictor object. As is standard with Stone Soup
predictors, we need to provide our predictor with a ``transition_model``. In addition, the
``dir_path`` argument specifies a directory to be added to the `MATLAB search path <https://uk.mathworks.com/help/matlab/search-path.html>`__
which contains the downloaded ``kf_predict.m`` script.


.. code-block:: default

    dir_path = '../'  # Change this to the directory where the kf_predict.m script is stored
    matlab_predictor = MatlabKalmanPredictor(transition_model=transition_model, dir_path=dir_path)








We also create a standard :class:`~.KalmanPredictor` object.


.. code-block:: default

    standard_predictor = KalmanPredictor(transition_model)








Perform prediction
~~~~~~~~~~~~~~~~~~
Finally, we proceed to perform a prediction step using both predictors and print-out the time
taken by each object to complete the operation.


.. code-block:: default


    # Assume we are predicting 2 seconds in the future
    timestamp_pred = timestamp_init + timedelta(seconds=2)

    # Matlab predictor
    matlab_prediction = matlab_predictor.predict(prior, timestamp=timestamp_pred)

    # Standard predictor
    standard_prediction = standard_predictor.predict(prior, timestamp=timestamp_pred)








Compare the results
~~~~~~~~~~~~~~~~~~~
The lines below are used to assert that the two predictors generate equivalent results.


.. code-block:: default

    assert(np.array_equal(standard_prediction.mean, matlab_prediction.mean))
    assert(np.array_equal(standard_prediction.covar, matlab_prediction.covar))








Plot the output
~~~~~~~~~~~~~~~


.. code-block:: default

    from matplotlib import pyplot as plt
    from matplotlib.patches import Ellipse


    def plot_state(state, ax, color, label=None):
        mean = state.mean
        cov = state.covar
        w, v = np.linalg.eig(cov[[0, 2], :][:, [0, 2]])
        max_ind = np.argmax(w)
        min_ind = np.argmin(w)
        orient = np.arctan2(v[1, max_ind], v[0, max_ind])
        ellipse = Ellipse(xy=(mean[0], mean[2]),
                              width=2 * np.sqrt(w[max_ind]), height=2 * np.sqrt(w[min_ind]),
                              angle=np.rad2deg(orient),
                              alpha=0.4,
                              color=color)
        ax.add_artist(ellipse)
        ax.plot(mean[0], mean[2], '.', color=color, label=label)


    fig, ax = plt.subplots()
    ax.set_ylim(-2, 5)
    ax.set_xlim(-2, 5)

    plot_state(prior, ax, 'r', 'Prior')
    plot_state(matlab_prediction, ax, 'b', 'Prediction')
    ax.legend()



.. image:: /auto_examples/images/sphx_glr_Matlab_Wrapper_001.png
    :alt: Matlab Wrapper
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <matplotlib.legend.Legend object at 0x000001BAE7068488>




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  6.455 seconds)


.. _sphx_glr_download_auto_examples_Matlab_Wrapper.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: Matlab_Wrapper.py <Matlab_Wrapper.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: Matlab_Wrapper.ipynb <Matlab_Wrapper.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
