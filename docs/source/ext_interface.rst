External Interface
==================

This sections describes how the interface to an external tracking library might look.

Introduction
------------
The external tracking library will define the structures that it accepts in terms of:

* Initialization parameters
* State update information
* Measurement information
* Format of returned track list

Interface
---------
The library functions generally take array of structures as parameters and the tracklist returned is an array of track structures. Examples of possible formats for these structures are given below:

.. code-block:: c

	struct measurement {
		double range;
		double bearing;
		double elevation;
		double timestamp;
	};
	
	struct track_cartesian {
		double x;
		double y;
		double z;
		double vx;
		double vy;
		double vz;
		double **covar
	};
	
	struct track_latlong {
		double lat;
		double long;
		double alt;
		double vlat;
		double vlong;
		double valt;
		double **covar
	};
	
Example Python code to interface to the external library is given below:

.. code-block:: python

	import TrackerLibrary as TL
	TL.init(param_list_xml)
	while True:
		TL.add_sensor_state()
		TL.add_measurement(measurement_list)
		tracklist = TL.get_tracks()
	TL.reset()
	TL.free()



TBD
---
These items may require further investigation.

* Memory allocation & management
* Specifics on Windows / Linux DLL handling

