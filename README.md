<h1><img valign="middle" alt="Stone Soup Logo" src="https://raw.githubusercontent.com/dstl/Stone-Soup/main/docs/source/_static/stone_soup_logo.svg" height="100"> Stone Soup</h1>

[![PyPI](https://img.shields.io/pypi/v/stonesoup?style=flat)](https://pypi.org/project/stonesoup)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/stonesoup.svg)](https://anaconda.org/conda-forge/stonesoup)
[![CircleCI branch](https://img.shields.io/circleci/project/github/dstl/Stone-Soup/main.svg?label=tests&style=flat)](https://circleci.com/gh/dstl/Stone-Soup)
[![Codecov](https://img.shields.io/codecov/c/github/dstl/Stone-Soup.svg)](https://codecov.io/gh/dstl/Stone-Soup)
[![Read the Docs](https://img.shields.io/readthedocs/stonesoup.svg?style=flat)](https://stonesoup.readthedocs.io/en/latest/?badge=latest)
[![Gitter](https://img.shields.io/gitter/room/dstl/Stone-Soup.svg?color=informational&style=flat)](https://gitter.im/dstl/Stone-Soup?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.4663993-informational)](https://doi.org/10.5281/zenodo.4663993)

## Background
Stone Soup is a software project to provide the target tracking and state estimation
community with a framework for the development and testing of tracking and state
estimation algorithms.

An article is [available](https://www.gov.uk/government/news/dstl-shares-new-open-source-framework-initiative) that details the background to the project, and contains links to sample data.

Please see the
[Stone Soup documentation](https://stonesoup.readthedocs.org/) for more
information.

Please see the [tutorials](https://stonesoup.readthedocs.io/en/latest/auto_tutorials/index.html),
[examples](https://stonesoup.readthedocs.io/en/latest/auto_examples/index.html),
and [demonstrations](https://stonesoup.readthedocs.io/en/latest/auto_demos/index.html),
which you can also try out on Binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dstl/Stone-Soup/main?filepath=notebooks)

## Dependencies
Stone Soup uses the following dependencies:

| Name | License |
| ---- | ------- |
| [Python](https://www.python.org/) (v3.8+) | PSFL |
| [numpy](https://numpy.org/) | BSD |
| [SciPy](https://www.scipy.org/) | BSD |
| [matplotlib](https://matplotlib.org/) | [PSF/BSD-compatible](https://matplotlib.org/users/license.html) |
| [ruamel.yaml](https://yaml.readthedocs.io/) | MIT |
| [pymap3d](https://github.com/scivision/pymap3d) | MIT |
| [utm](https://github.com/Turbo87/utm) | MIT |
| [ordered-set](https://github.com/LuminosoInsight/ordered-set) | MIT |
| [setuptools](https://github.com/pypa/setuptools) | MIT |
| [rtree](https://github.com/Toblerity/rtree) | MIT |

### Development

#### Testing
These dependencies are required for running Stone Soup tests.

| Name | License |
| ---- | ------- |
| [pytest](https://docs.pytest.org/) | MIT |
| [Flake8](https://flake8.pycqa.org/) | MIT |
| [Coverage.py](https://coverage.readthedocs.io/) | Apache 2.0 |

#### Documentation
These dependencies are required for building Stone Soup documentation.

| Name | License |
| ---- | ------- |
| [Sphinx](https://www.sphinx-doc.org/) | BSD |
| [sphinx-gallery](https://sphinx-gallery.github.io/) | BSD |
| [pillow](https://pillow.readthedocs.io/en/stable/index.html) | [PIL Software License](https://pillow.readthedocs.io/en/stable/about.html#license) |
| [folium](https://python-visualization.github.io/folium/) | MIT |

## License
Stone Soup is released under MIT License. Please see [License](LICENSE) for details.
