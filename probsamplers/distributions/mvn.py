import numpy as np
from probsamplers import aux
import scipy.stats


class MultivariateNormal:
    # Static member
    rng = np.random.default_rng()

    def __init__(self, mu, covMat):
        self.mean = np.asarray(mu)
        self.dim = self.mean.shape[0]
        self.constant = -0.5 * np.log(2.0 * np.pi) * self.dim
        assert covMat.shape == (self.dim, self.dim)
        self.covMat = covMat
        self.covL = np.linalg.cholesky(covMat)
        self.logDet = np.log(self.covL.diagonal()).sum()

    @staticmethod
    def getNormal():
        def draw():
            pv = lambda: MultivariateNormal.rng.standard_normal() * 2 - 1
            x = pv()
            y = pv()
            return aux.structs.drawvals(res=(x**2 + y**2), x=x, y=y)

        w = draw()
        while w.res >= 1.0:
            w = draw()
        return w.x * np.sqrt((-2 * np.log(w.res)) / w.res)

    @staticmethod
    def getMVNSample(dim):
        dist = MultivariateNormal(np.zeros(dim), np.eye(dim))
        return dist.getSample()

    def getSample(self):
        return MultivariateNormal.rng.multivariate_normal(
            self.mean, self.covMat
        )

    def logDensity(self, x):
        diff = self.mean - x
        eqsolv = np.linalg.solve(self.covL, diff)
        return self.constant - np.linalg.norm(
            self.logDet - 0.5 * np.linalg.solve(self.covL, eqsolv)
        )

    def gradLogDensity(self, x):
        diff = self.mean - x
        eqsolv2 = np.linalg.solve(self.covL, diff)
        return -1 * np.linalg.solve(self.covL, eqsolv2)

    def __repr__(self):
        return f"mu: {self.mean} \t logDet: {self.logDet} \n cov: {self.covMat}\n covL: {self.covL}\n"
