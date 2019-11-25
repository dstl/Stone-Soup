#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

from stonesoup import __version__ as version

with open('README.md') as f:
    long_description = f.read()

setup(name='stonesoup',
      version=version,
      maintainer='Defence Science and Technology Laboratory UK',
      maintainer_email='oss@dstl.gov.uk',
      url='https://github.com/dstl/Stone-Soup',
      description='A tracking and state estimation framework',
      long_description=long_description,
      long_description_content_type='text/markdown',
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3 :: Only',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
      ],
      packages=find_packages(exclude=('docs', '*.tests')),
      install_requires=[
          'ruamel.yaml>=0.15.45', 'scipy', 'matplotlib', 'utm', 'pymap3d',
          'ffmpeg-python', 'moviepy'],
      extras_require={
          'dev': [
              'pytest-flake8', 'pytest-cov', 'Sphinx', 'sphinx_rtd_theme',
              'setuptools>=30'],
      },
      )
