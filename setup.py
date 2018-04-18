#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

from stonesoup import __version__ as version

setup(name='stonesoup',
      version=version,
      description='A target tracking development/testing framework',
      url='https://github.com/dstl/Stone-Soup',
      packages=find_packages(exclude=('docs', '*.tests')),
      install_requires=['numpy', 'ruamel.yaml', 'scipy', 'matplotlib'],
      extras_require={
          'dev': ['pytest', 'Sphinx', 'flake8', 'coverage'],
      },
      )
