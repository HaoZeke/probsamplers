import probsamplers as pbs
import numpy as np
import scipy.stats
import pytest

TOL_TEST = 1e-02

def test_mvn_samples():
    mu = [0.1, 0.2, 0.3]
    covMat = np.array([2, -1, 0, -1, 2, -1, 0, -1, 2]).reshape(3,3)
    a_pbs = pbs.distributions.mvn.MultivariateNormal(mu, covMat)
    a_scpy = scipy.stats.multivariate_normal(mu, covMat)
    assert a_pbs.logDensity(1) == pytest.approx(a_scpy.logpdf(1), TOL_TEST)