"""Stone Soup framework: development and assessment of tracking algorithms."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("stonesoup")
except PackageNotFoundError:
    # package is not installed
    pass

__copyright__ = '''\
© Crown Copyright 2017-2025 Defence Science and Technology Laboratory UK
© Crown Copyright 2018-2025 Defence Research and Development Canada / Recherche et développement pour la défense Canada
© Copyright 2018-2025 University of Liverpool UK
© Copyright 2020-2025 Fraunhofer FKIE
© Copyright 2020-2025 John Hiles
© Copyright 2021-2025 Roke Manor Research Ltd UK
© Copyright 2023-2025 Loughborough University UK
'''  # noqa: E501
__license__ = 'MIT'
