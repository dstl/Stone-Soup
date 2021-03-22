# -*- coding: utf-8 -*-
"""Stone Soup framework: development and assessment of tracking algorithms."""

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution("stonesoup").version
except DistributionNotFound:
    # package is not installed
    pass

__copyright__ = '''\
© Crown Copyright 2017-2021 Defence Science and Technology Laboratory UK
© Crown Copyright 2018-2021 Defence Research and Development Canada / Recherche et développement pour la défense Canada
© Copyright 2018-2021 University of Liverpool UK
'''  # noqa: E501
__license__ = 'MIT'
