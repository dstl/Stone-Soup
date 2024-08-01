from setuptools import find_packages, setup

setup(
    name="dstl-mast",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3.9.13",
    entry_points={"stonesoup.plugins": "RL = ReinforcementLearning"},
)
