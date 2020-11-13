#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(name='stonesoup',
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
      python_requires='>=3.6',
      setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
      use_scm_version={'write_to': 'stonesoup/_version.py'},
      install_requires=[
          'ruamel.yaml>=0.15.45', 'numpy>=1.17', 'scipy', 'matplotlib', 'utm', 'pymap3d'],
      extras_require={
          'dev': [
              'pytest-flake8', 'pytest-cov', 'Sphinx', 'sphinx_rtd_theme',
              'setuptools>=42', 'sphinx-gallery>=0.8', 'pillow', 'folium'],
          'video': ['ffmpeg-python', 'moviepy'],
          'tensorflow': ['tensorflow>=2.2.0']
      },
      )
