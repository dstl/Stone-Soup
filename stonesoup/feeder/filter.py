# -*- coding: utf-8 -*-
from operator import attrgetter
from types import FunctionType

import numpy as np

from .base import DetectionFeeder, GroundTruthFeeder
from ..base import Property
from ..buffered_generator import BufferedGenerator


class MetadataReducer(DetectionFeeder):
    """Reduce detections so unique metadata value present at each time step.

    This allows to reduce detections so a single detection is returned, based
    on a particular metadata value, for example a unique identity. The most
    recent detection will be yielded for each unique metadata value at each
    time step.

    Note
    ====
    * If :class:`~.GroundTruthPath` type is extended to have a metadata attribute, this class
    will be applicable to this type.

    """

    metadata_field = Property(
        str,
        doc="Field used to reduce set of detections")

    @BufferedGenerator.generator_method
    def data_gen(self):
        for time, detections in self.reader:
            unique_detections = set()
            sorted_detections = sorted(
                detections, key=attrgetter('timestamp'), reverse=True)
            meta_values = set()
            for detection in sorted_detections:
                meta_value = detection.metadata.get(self.metadata_field)
                if meta_value not in meta_values:
                    unique_detections.add(detection)
                    # Ignore those without meta data value
                    if meta_value is not None:
                        meta_values.add(meta_value)
            yield time, unique_detections


class MetadataValueFilter(DetectionFeeder):
    """ Reduce detections by filtering out objects based on whether the value
        of a particular metadata field conforms to a given condition.

        The MetadataValueFilter provides an easy way of reducing detections in
        cases where an informative metadata field exists (e.g. MMSI, SNR, etc.)
        , that can be used to identify and filter out unwanted detections.

        Once provided with a :py:attr:`~metadata_field` name and a suitable
        :py:attr:`~operator` function, the feeder will filter the incoming
        detections by evaluating the :py:attr:`~metadata_field` field value
        using the desired :py:attr:`~operator` function. Only detections
        that satisfy the operator condition (i.e. cause the
        :py:attr:`~operator` to return :code:`True`) are allowed through the
        filter.

        Note
        ====
        * If :class:`~.GroundTruthPath` type is extended to have a metadata attribute, this class
        will be applicable to this type.

    """

    metadata_field = Property(
        str,
        doc="Field used to reduce set of detections")

    operator = Property(
        FunctionType,
        doc="A unary operator/function of the form :code:`b = f(val)`, "
            "where :code:`val` is the value of the selected "
            ":py:attr:`~metadata_field`. The function MUST return a "
            ":py:class:`~bool` type that evaluates to :code:`True` when a "
            "particular object satisfies the condition(s) set by the operator,"
            " and thus should be allowed through the filter. Detections that "
            "cause the operator to return :code:`False` will be filtered out. "
            "Any custom function that conforms to the above specifications can"
            " be used as an operator, e.g. :code:`operator=lambda x: x < 0.1`")

    keep_unmatched = Property(
        bool,
        doc="If set to :code:`True`, any detections that do not have a "
            "metadata field matching the name :py:attr:`~metadata_field` "
            "(meaning they also cannot be processed by the "
            ":py:attr:`~operator`), will be allowed through the filter. The "
            "default is :code:`False`.",
        default=False)

    @BufferedGenerator.generator_method
    def data_gen(self):
        for time, detections in self.reader:
            filtered_detections = set()
            for detection in detections:
                value = detection.metadata.get(self.metadata_field)
                if value is None and self.keep_unmatched:
                    filtered_detections.add(detection)
                elif value is not None and self.operator(value):
                    filtered_detections.add(detection)

            yield time, filtered_detections


class BoundingBoxDetectionReducer(DetectionFeeder, GroundTruthFeeder):
    """ Reduce data by selecting only data placed within the limits of a
        n-dimensional bounding box, defined on the data coordinate space.

        When provided with the limit coordinates of a given n-dimensional
        bounding box (expressed in the form of min/max bounds on each
        dimension), the feeder will apply a filter to the incoming data,
        allowing to pass through only data that falls within the desired
        limits, and discarding the rest.

        Assuming a 2D Cartesian data space, the feeder operation is
        equivalent to drawing an imaginary bounding box tangential to the plane
        defined by the XY axes, and only feeding data whose state vector
        falls within the bounds of the box.

        Note
        ====
        * For the time being, the bounding box limits must be defined on
          the same coordinate axes as the received data, which is in
          turn assumed to all share a common coordinate frame.
        * For example, assume we are tracking some targets in Lat/Lon, but
          receive detections in the form of polar coordinates (e.g. relative to
          a single Radar sensor), then the bounding box MUST be defined
          on (a subset of) the polar coordinate system, and NOT the
          geo-spatial.
        * Thus, it is worth noting that in its current version the feeder is
          not recommended for use in filtering data coming from multiple
          sources/sensors, unless a common coordinate frame is guaranteed.
        * Finally, for simplicity purposes, the bounding box is not allowed to
          rotate around any axis.

    """

    limits = Property(
        np.ndarray,
        doc="Array of points that define the bounds of the desired bounding "
            "box. Expressed as a 2D array of min/max coordinate pairs (e.g. "
            ":code:`limits = [[x_min, x_max], [y_min, y_max], ...]`), where "
            "the n-th row corresponds to the n-th bounding box dimension "
            "limits. Points that fall ON or WITHIN the box's bounds are "
            "considered as valid and are thus forwarded through the feeder, "
            "whereas points that fall OUTSIDE the box will be filtered out.")
    mapping = Property(
        np.ndarray,
        default=None,
        doc="Mapping between the state and bounding box coordinates. "
            "Should be specified as a vector of length equal to the number of "
            "bounding box dimensions, whose elements correspond to row indices"
            " in the state vector. E.g. :code:`mapping = [2, 0]` "
            "dictates that the first bounding box dimension (i.e. row 0 in "
            ":py:attr:`~limits`), defines the limits that correspond to the "
            "element with (row) index 2 in the data state vector, while "
            "the second row (i.e. row 1) in :py:attr:`~limits` relates to the "
            "element with index 0. Default is `None`, where the dimensions of "
            "the state vector will be used in order, up to length to limits."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mapping is None:
            self.mapping = tuple(range(len(self.limits)))

    @BufferedGenerator.generator_method
    def data_gen(self):
        num_dims = len(self.limits)
        for time, states in self.reader:
            outlier_data = set()
            for state in states:
                state_vector = state.state_vector
                for i in range(num_dims):
                    min = self.limits[i][0]
                    max = self.limits[i][1]
                    value = state_vector[self.mapping[i]]
                    if value < min or value > max:
                        outlier_data.add(state)
                        break
            yield time, states - outlier_data
