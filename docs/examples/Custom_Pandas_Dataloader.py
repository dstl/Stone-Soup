#!/usr/bin/env python

"""
Use of Custom Readers that support Pandas DataFrames
====================================================
This is a demonstration of using customised readers that
support data contained within Pandas DataFrames, rather than
loading directly from a .csv file using :class:`~.CSVGroundTruthReader` or 
:class:`~.CSVDetectionReader`. 

The benefit is that this allows us to use the versatile data loading
capabilities of pandas to read from many different data source types 
as needed, including .csv, JSON, XML, Parquet, HDF5, .txt, .zip and more. 
The resulting DataFrame can then simply be fed into the defined 
`DataFrameGroundTruthReader` or `DataFrameDetectionReader` for further processing
in Stone Soup as required. 
"""

# %%
# Software dependencies
# ---------------------
# Before beginning this example, you need to ensure that Pandas is installed, 
# which is a fast, powerful and flexible open-source data analysis tool in Python.
# The easiest way to install pandas (if not done so already), is to run pip install
# from a terminal window within the desired environment:
#
# .. code::
#
#     pip install pandas

# %%
# The main dependencies and imports for this example are included below:

import numpy as np
import os
import pandas as pd

from stonesoup.base import Property
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.reader.base import GroundTruthReader, DetectionReader, Reader
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from typing import Sequence, Collection

from datetime import datetime, timedelta
from dateutil.parser import parse

# %%
# Data Frame Reader
# ^^^^^^^^^^^^^^^^^
# Similarly to Stone Soup's :class:`~._CSVFrameReader`, we'll define a `_DataFrameReader`
# class that inherits from the base :class:`~.Reader` class to read a DataFrame containing
# state vector fields, a time field, and additional metadata fields (all other columns
# by default). The only difference between this class and the :class:`~._CSVFrameReader` 
# class is that we have no path attribute (the DataFrame is already loaded in memory).

class _DataFrameReader(Reader):
    state_vector_fields: Sequence[str] = Property(
        doc='List of columns names to be used in state vector')
    time_field: str = Property(
        doc='Name of column to be used as time field')
    time_field_format: str = Property(
        default=None, doc='Optional datetime format')
    timestamp: bool = Property(
        default=False, doc='Treat time field as a timestamp from epoch')
    metadata_fields: Collection[str] = Property(
        default=None, doc='List of columns to be saved as metadata, default all')

    def _get_metadata(self, row):
        if self.metadata_fields is None:
            local_metadata = dict(row)
            for key in list(local_metadata):
                if key == self.time_field or key in self.state_vector_fields:
                    del local_metadata[key]
        else:
            local_metadata = {field: row[field]
                              for field in self.metadata_fields
                              if field in row}
        return local_metadata

    def _get_time(self, row):
        if self.time_field_format is not None:
            time_field_value = datetime.strptime(row[self.time_field], self.time_field_format)
        elif self.timestamp:
            fractional, timestamp = modf(float(row[self.time_field]))
            time_field_value = datetime.utcfromtimestamp(int(timestamp))
            time_field_value += timedelta(microseconds=fractional * 1E6)
        else:
            time_field_value = parse(row[self.time_field], ignoretz=True)
        return time_field_value


# %%
# Data Ground Truth Reader
# ^^^^^^^^^^^^^^^^^^^^^^^^
# With the help of our `_DataFrameReader` class, we can now define a custom
# `DataFrameGroundTruthReader`. This is similar to :class:`~.CSVGroundTruthReader` and 
# inherits from the base `GroundTruthReader` class. A key difference is that we 
# include an instance attribute for the dataframe containing our data.
# 
# We also define a custom generator function (groundtruth_paths_gen) that uses the decorator
# `@BufferedGenerator.generator_method`. The generator needs to return a time and a set of
# detections, like so:

class DataFrameGroundTruthReader(GroundTruthReader, _DataFrameReader):
    """A custom reader for pandas DataFrames containing truth data.

    The DataFrame must have colums containing all fields needed to generate the
    ground truth state. Those states with the same ID will be put into
    a :class:`~.GroundTruthPath` in sequence, and all paths that are updated at the same time
    are yielded together, and such assumes file is in time order.

    Parameters
    ----------
    """
    dataframe: pd.DataFrame = Property(doc="DataFrame containing the ground truth data.")
    path_id_field: str = Property(doc='Name of column to be used as path ID')

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self):
        """ Generator method for providing each row of ground truth data. """
        groundtruth_dict = {}
        updated_paths = set()
        previous_time = None
        for row in self.dataframe.to_dict(orient="records"):

            time = self._get_time(row)
            if previous_time is not None and previous_time != time:
                yield previous_time, updated_paths
                updated_paths = set()
            previous_time = time

            state = GroundTruthState(
                np.array([[row[col_name]] for col_name in self.state_vector_fields],
                        dtype=np.float_),
                timestamp=time,
                metadata=self._get_metadata(row))

            id_ = row[self.path_id_field]
            if id_ not in groundtruth_dict:
                groundtruth_dict[id_] = GroundTruthPath(id=id_)
            groundtruth_path = groundtruth_dict[id_]
            groundtruth_path.append(state)
            updated_paths.add(groundtruth_path)

            # Yield remaining
        yield previous_time, updated_paths

# %%
# With our `DataFrameGroundTruthReader` defined, we can test it on a simple example. Let's 
# do a basic 3D simulation to create an example dataframe, from which we can test our class:

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity

q_x = 0.05
q_y = 0.05
q_z = 0.05
start_time = datetime.now()
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                          ConstantVelocity(q_y),
                                                          ConstantVelocity(q_z)])
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1, 0, 1], timestamp=start_time)])

times = []
x, y, z = [], [], []
vel_x, vel_y, vel_z = [], [], []

num_steps = 25
for k in range(1, num_steps + 1):
    
    time = start_time+timedelta(seconds=k)
    
    next_state = GroundTruthState(
        transition_model.function(truth[k-1], noise=True, 
                                  time_interval=timedelta(seconds=1)),
        timestamp=time)
    truth.append(next_state)
    
    times.append(time)
    x.append(next_state.state_vector[0])
    vel_x.append(next_state.state_vector[1])
    y.append(next_state.state_vector[2])
    vel_y.append(next_state.state_vector[3])
    z.append(next_state.state_vector[4])
    vel_z.append(next_state.state_vector[5])

truth_df = pd.DataFrame({'time' : times, 
                         'x' : x, 
                         'y' : y, 
                         'z' : z, 
                         'vel_x' : vel_x, 
                         'vel_y' : vel_y,
                         'vel_z' : vel_z,
                         'track_id' : 0})

truth_df.head(5)

# %%
# Note that the process above is just an example for providing a simple dataframe to use,
# and is not generally something we would need to do (since we already have the GroundTruthPath).
# The dataframe above is just used to show the workings of our newly defined `DataFrameGroundTruthReader`.
# In practice, we can use any dataframe containing our Cartesian positions or longitude and latitude 
# co-ordinates. Note that if we are using longitude and latitude inputs, we would also need to
# transform these using :class:`~.LongLatToUTMConverter` or equivalent.
# 
# We can now initialise our DataFrameGroundTruthReader using this example DataFrame like so:

# read ground truth data from pandas dataframe
ground_truth_reader = DataFrameGroundTruthReader(
    dataframe=truth_df,
    state_vector_fields=['x', 'vel_x', 'y', 'vel_y', 'z', 'vel_z'],
    time_field='time',
    path_id_field='track_id')


# %%
# Let's demonstrate the ground truth reader generating output for one iteration:

next(iter(ground_truth_reader))



# %%
# Another benefit of this ground truth reader is that we now have convenient access to the original
# dataframe, using the .dataframe attribute, like so:

ground_truth_reader.dataframe.head(3)

# %%
# DataFrame Detection Reader
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# Similarly to our `DataFrameGroundTruthReader`, we can develop a custom `DataFrameDetectionReader`
# that can read in DataFrames containing detections through subclassing from Stone Soup's
# `DetectionReader` class, along with our custom `_DataFrameReader` class above.
# Again, this closely resembles the existing `CSVDetectionReader` class within the Stone Soup
# library, except we include a instance attribute 'dataframe', and modify our detections_gen
# function to work with dataframes rather than .csv files. This can be seen below:

class DataFrameDetectionReader(DetectionReader, _DataFrameReader):
    """A custom detection reader for DataFrames containing detections.

    DataFrame must have headers with the appropriate fields needed to generate
    the detection. Detections at the same time are yielded together, and such assume file is in
    time order.

    Parameters
    ----------
    """
    dataframe: pd.DataFrame = Property(doc="DataFrame containing the ground truth data.")
    
    @BufferedGenerator.generator_method
    def detections_gen(self):
        detections = set()
        previous_time = None
        for row in self.dataframe.to_dict(orient="records"):
            
            time = self._get_time(row)
            if previous_time is not None and previous_time != time:
                yield previous_time, detections
                detections = set()
            previous_time = time

            detections.add(Detection(
                np.array([[row[col_name]] for col_name in self.state_vector_fields],
                        dtype=np.float_),
                timestamp=time,
                metadata=self._get_metadata(row)))

        # Yield remaining
        yield previous_time, detections


# %%
# We can instantiate this using our example DataFrame above like so:

detection_reader = DataFrameDetectionReader(
    dataframe=truth_df,
    state_vector_fields=['x', 'vel_x', 'y', 'vel_y', 'z', 'vel_z'],
    time_field='time')

# %%
# Following this, we can now perform any desired follow-up task such as simulation or tracking
# as covered in the other Stone Soup examples, tutorials and demonstrations. As discussed previously,
# the huge benefits of using a custom DataFrame reader like this is that we can read any type of data
# supported by the pandas library, which gives us a huge range of options. This strategy also saves
# us the overhead of manually specifying custom Stone Soup Reader classes for each format of data.