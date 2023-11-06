"""Stone Soup framework: development and assessment of tracking algorithms."""

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution("stonesoup").version
except DistributionNotFound:
    # package is not installed
    pass

__copyright__ = '''\
© Crown Copyright 2017-2023 Defence Science and Technology Laboratory UK
© Crown Copyright 2018-2023 Defence Research and Development Canada / Recherche et développement pour la défense Canada
© Copyright 2018-2023 University of Liverpool UK
© Copyright 2020-2023 Fraunhofer FKIE
© Copyright 2020-2023 John Hiles
© Copyright 2021-2023 Roke Manor Research Ltd UK
'''  # noqa: E501
__license__ = 'MIT'
