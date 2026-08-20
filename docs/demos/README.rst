Demonstrations
==============

Here are a selection of demonstrations of Stone Soup against real data sets.

Running downloaded demonstrations
---------------------------------

The Sphinx-Gallery download links are useful for taking a demonstration into a
local notebook, but some demonstrations also depend on data files, generated
assets, optional Python packages, or external downloads that are not contained
in every generated archive. Before running a downloaded demonstration, check
its documentation page for the complete setup instructions.

The repository data files used by the demonstrations can be downloaded directly:

* `Solent AIS data <https://raw.githubusercontent.com/dstl/Stone-Soup/main/docs/demos/SolentAIS_20160112_130211.csv>`_
  for ``AIS_Solent_Tracker``;
* `OpenSky plane-state data <https://raw.githubusercontent.com/dstl/Stone-Soup/main/docs/demos/OpenSky_Plane_States.csv>`_
  for ``OpenSky_Demo``; and
* `UAV rotation data <https://raw.githubusercontent.com/dstl/Stone-Soup/main/docs/demos/UAV_Rot.csv>`_
  for ``UAV_tutorial``.

``AIS_Solent_Tracker`` and ``OpenSky_Demo`` use ``folium``. It is included in
Stone Soup's ``dev`` extra for a development checkout. ``Video_Processing`` has
additional multimedia and object-detection requirements; follow that demo's
setup instructions and install the ``video`` and ``ultralytics`` extras. The
video demonstration also downloads its example video, and Ultralytics may
download model weights on first use, so those steps require network access.

For the most reproducible setup, particularly on an isolated or restricted
network, clone the repository so the demonstration sources and repository data
files are available together, and obtain any externally downloaded video or
model assets before moving into the restricted environment.
