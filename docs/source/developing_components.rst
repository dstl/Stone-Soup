Developing Stone Soup components
================================

Stone Soup is designed around small, composable components. New functionality should normally
extend the nearest existing base type rather than introducing a parallel interface. This page
summarises the contracts to preserve when adding the component families most commonly extended by
contributors.

Before starting, read :ref:`contributing:Contributing` for the project-wide code style, testing,
documentation, dependency and pull-request requirements. It is also useful to find the closest
existing component and its tests: matching established conventions usually produces a smaller and
easier-to-review change.

Common development pattern
--------------------------

For most new components:

#. Subclass the narrowest appropriate Stone Soup base class.
#. Declare user-configurable values with :class:`~.Property` rather than adding an unrelated
   constructor convention.
#. Preserve the input and output types of the base interface. Use the existing prediction, update,
   detection, track and metric types where applicable.
#. Accept and forward ``**kwargs`` when the base API supports them, so model-specific options can
   travel through composed components.
#. Make deterministic behaviour directly testable. For stochastic components, tests should use a
   fixed random state or disable noise where the interface permits it.
#. Add focused unit tests for the normal path, boundary cases and expected failures.
#. Document public interfaces with NumPy-style docstrings. Where a new feature benefits from a
   worked workflow, add a Sphinx-Gallery example as described on the contributing page.

Measurement models
------------------

Measurement models translate state space into measurement space. Start from
:class:`~.MeasurementModel` and choose the existing linear, nonlinear and/or probabilistic model
base classes that match the mathematics of the new model.

A measurement model should define the dimensional contract explicitly:

* ``ndim_state`` is the number of state dimensions accepted by the model;
* ``mapping`` identifies the state dimensions represented in the measurement where a mapping is
  meaningful;
* ``ndim_meas`` reports the number of measurement dimensions;
* ``function(state, ...)`` evaluates the measurement function and must return an array with the
  expected measurement shape.

Depending on the model family, the surrounding Stone Soup interfaces may also expect a covariance
through ``covar()``, a matrix through ``matrix()`` for linear models, or a Jacobian through
``jacobian()`` for differentiable nonlinear models. Follow the nearest existing model rather than
implementing methods that the selected base class does not require.

Tests should cover dimensions and mapping, deterministic evaluation with noise disabled, stochastic
behaviour with a controlled random state where relevant, and matrix/Jacobian behaviour if the model
exposes it. If the implementation supports vectorised states, include a vectorised regression as
well as a single-state case.

Transition models
-----------------

Transition models propagate a state through time. New transition models derive from
:class:`~.TransitionModel` or a more specific existing transition-model base class.

The core contract is:

* ``ndim_state`` reports the state dimension;
* ``function(state, ...)`` returns the transitioned state vector;
* time-varying models consume the established ``time_interval`` keyword used by Stone Soup model
  implementations.

Gaussian transition models should provide process-noise covariance in the way required by their
base class. Linear models should expose their transition matrix, while differentiable nonlinear
models should follow the established Jacobian interface when one is required by downstream
predictors.

Test zero/known-noise behaviour separately from sampled noise, check output shape, and include
representative time intervals. For models with analytical matrix or Jacobian forms, test those
quantities independently of a full tracker.

Predictors and updaters
-----------------------

Predictors and updaters form the filter-facing interface around transition and measurement models.
A new predictor normally subclasses :class:`~.Predictor` and implements::

    predict(prior, timestamp=None, **kwargs)

The result should use the appropriate Stone Soup prediction type and retain relevant information
from the prior state. Reuse the existing ``from_state`` factories when the neighbouring
implementations do so, because these preserve the state-family-specific prediction type.

A new updater normally subclasses :class:`~.Updater` and implements both::

    predict_measurement(predicted_state, measurement_model=None,
                        measurement_noise=True, **kwargs)
    update(hypothesis, **kwargs)

``predict_measurement`` should return the appropriate measurement-prediction type. ``update``
should return an update/posterior state that is compatible with the prediction family. If an
updater can take its measurement model either at construction time or from a detection, preserve
the established model-selection behaviour.

Filter tests should check returned types, timestamps, state vectors and uncertainty quantities, not
only that execution completes. Test model override paths separately when the component supports
them.

Metric generators
-----------------

Metric implementations derive from :class:`~.MetricGenerator`. A generator needs a stable,
unique ``generator_name`` and implements::

    compute_metric(manager, **kwargs)

Metric data should be obtained from the supplied metric manager using the same keys and helper
patterns as existing generators. Return Stone Soup :class:`~.Metric` objects rather than raw values
when the surrounding metric API expects them.

Tests should exercise the metric numerically on a small deterministic fixture and include important
empty or degenerate cases, such as no associations or a timestamp with no applicable state. When a
metric spans time, verify both its values and reported time range.

Readers
-------

Readers convert an external data source into Stone Soup objects over time. Choose the reader base
that matches the output:

* :class:`~.DetectionReader` for detections;
* :class:`~.GroundTruthReader` for ground-truth paths;
* :class:`~.TrackReader` for tracks;
* :class:`~.SensorDataReader` for sensor data;
* :class:`~.FrameReader` for image frames.

Reader generator methods yield ``(timestamp, set_of_objects)`` pairs and use
:func:`~.BufferedGenerator.generator_method` in the same way as the existing readers. Preserve time
ordering and group objects that belong to the same time step consistently. Keep parsing concerns
inside the reader and return standard Stone Soup types to downstream components.

Reader tests should use small local fixtures rather than network services where possible. Check
state-vector fields, timestamp conversion, grouping, IDs/metadata and malformed input. For formats
with optional third-party dependencies, follow the project's existing optional-dependency pattern.
The pandas reader examples provide one reference for adapting an external data representation into
the standard reader interface.

Sensors
-------

A sensor derives from :class:`~.Sensor` or, where its behaviour matches the existing convenience
layer, :class:`~.SimpleSensor`. A sensor exposes a ``measurement_model`` and implements::

    measure(ground_truths, noise=True, **kwargs)

Measurements should be returned as Stone Soup detection types with the correct timestamp and source
information. Sensor properties that can be changed by sensor-management actions should use the
existing actionable-property mechanism rather than a separate command interface.

For ``SimpleSensor`` implementations, keep detectability and clutter handling consistent with the
base class contracts. If a sensor has field-of-view, range or visibility constraints, test those
boundaries independently of the measurement-noise path.

Sensor tests should include ``noise=False`` for a deterministic geometric check and a controlled
random-state case when noise generation is part of the implementation. Also verify timestamps,
measurement-model attachment and any platform-relative geometry.

Initiators and deleters
-----------------------

Initiators create tracks from detections. New initiators derive from :class:`~.Initiator` or an
appropriate Gaussian/particle initiator base and implement::

    initiate(detections, timestamp, **kwargs)

The method returns a set of :class:`~.Track` objects. Initial state types should be compatible with
the tracker, predictor and updater that will consume them. Wrapper initiators should delegate to
the wrapped component and preserve its track semantics unless the wrapper explicitly documents a
transformation.

Deleters derive from :class:`~.Deleter`. In most cases only::

    check_for_deletion(track, **kwargs)

needs to be implemented; the base ``delete_tracks`` method applies the check across a set of tracks
and handles the standard ``delete_last_pred`` option.

For initiators, test empty detections, the expected initial state and timestamp, and any minimum
measurement/release criteria. For deleters, test values on both sides of each deletion threshold
and any behaviour involving the final prediction state.

Review checklist
----------------

Before opening a pull request for a new component, check that:

* the component extends the appropriate existing base type;
* its public configuration is represented using established Stone Soup properties;
* returned Stone Soup types and array dimensions are explicit and tested;
* stochastic tests are reproducible;
* edge cases and failure behaviour have tests;
* public APIs have NumPy-style documentation;
* a worked example is included when it materially helps users understand the feature;
* ``pytest --flake8 stonesoup`` passes for code changes; and
* ``sphinx-build -W docs/source docs/build`` passes for documentation changes.

These checks complement, rather than replace, the full project guidance on the
:ref:`contributing:Contributing` page.
