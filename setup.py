#!/usr/bin/env python

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
      python_requires='>=3.7',
      setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
      use_scm_version=True,
      install_requires=[
          'ruamel.yaml>=0.16.5', 'numpy>=1.17', 'scipy', 'matplotlib', 'utm', 'pymap3d', 'ordered-set',
          'setuptools>=42', 'rtree',
      ],
      extras_require={
          'dev': [
              'pytest-flake8', 'pytest-cov', 'pytest-remotedata', 'flake8<5',
              'Sphinx', 'sphinx_rtd_theme', 'sphinx-gallery>=0.10.1', 'pillow', 'folium', 'plotly',
          ],
          'video': ['ffmpeg-python', 'moviepy', 'opencv-python'],
          'tensorflow': ['tensorflow>=2.2.0'],
          'tensornets': ['tensorflow>=2.2.0', 'tensornets'],
      },
      )
