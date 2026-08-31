import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal

from stonesoup.plotter import Plotter
from stonesoup.types.mixture import GaussianMixture
from stonesoup.types.state import WeightedGaussianState


def test_plot_density_gaussian_mixture(monkeypatch):
    covariance_1 = np.array([
        [1.0, 0.0, 0.3, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.3, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    covariance_2 = np.diag([2.0, 1.0, 1.5, 1.0])
    mixture = GaussianMixture([
        WeightedGaussianState([0, 0, 0, 0], covariance_1, weight=0.4),
        WeightedGaussianState([5, 0, 4, 0], covariance_2, weight=0.6),
    ])

    plotter = Plotter()
    plotted = {}

    def capture_density(x, y, density, **kwargs):
        plotted.update(x=x, y=y, density=density, kwargs=kwargs)

    monkeypatch.setattr(plotter.ax, 'pcolormesh', capture_density)
    plotter.plot_density(mixture, mapping=(0, 2), n_bins=30, cmap='viridis')

    positions = np.dstack((plotted['x'], plotted['y']))
    expected = sum(
        component.weight * multivariate_normal.pdf(
            positions,
            mean=component.mean[(0, 2), 0],
            cov=component.covar[np.ix_((0, 2), (0, 2))],
            allow_singular=True,
        )
        for component in mixture
    )

    assert plotted['density'].shape == (30, 30)
    assert plotted['kwargs']['shading'] == 'auto'
    assert plotted['kwargs']['cmap'] == 'viridis'
    assert_allclose(plotted['density'], expected)
