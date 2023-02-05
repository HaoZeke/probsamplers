# Common
import abc
import numpy as np
import scipy
from collections import namedtuple

# Internal libraries
import probsamplers.distributions as pbsd
from probsamplers import aux


class baseTargetDistrib(metaclass=abc.ABCMeta):
    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "logDensity")
            and callable(subclass.logDensity)
            and hasattr(subclass, "gradLogDensity")
            and callable(subclass.gradLogDensity)
            or NotImplemented
        )

    @abc.abstractmethod
    def logDensity(self, x):
        """Return the log of the probability density"""
        raise NotImplementedError

    @abc.abstractmethod
    def gradLogDensity(self, x):
        """Returns the gradient of the probability density"""
        raise NotImplementedError

    @abc.abstractmethod
    def gradLogDensity(self, x):
        """Returns the gradient of the probability density"""
        raise NotImplementedError

    def plotDensity(
        self,
        xlim={"low": -1.5, "high": 1.5},
        ylim={"low": -1.5, "high": 1.5},
        nstep=200,
    ):
        x = np.linspace(xlim["low"], xlim["high"], nstep)
        y = np.linspace(ylim["low"], ylim["high"], nstep)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros((nstep, nstep))
        for i in np.arange(0, nstep):
            for j in np.arange(0, nstep):
                zz[i, j] = np.exp(self.logDensity([xx[i, j], yy[i, j]]))
        return aux.structs.plotvals(xx=xx, yy=yy, zz=zz)


class rosenbrockBanana(baseTargetDistrib):
    def __init__(
        self,
        xmin=-6,
        xmax=6,
        alpha=2,
        beta=0.2,
        mu=np.array([0, 4]),
        cov=np.array([1, 0.5, 0.5, 1]).reshape(2, 2),
    ):
        super().__init__(xmin, xmax)
        self.alpha = alpha
        self.beta = beta
        self.pdistrib = pbsd.mvn.MultivariateNormal(mu, cov)

    def getYvec(self, x):
        assert len(x) == 2
        yvec = np.zeros(2)
        yvec[0] = x[0] / self.alpha
        yvec[1] = x[1] * self.alpha + self.alpha * self.beta * (
            x[0] ** 2 + self.alpha**2
        )
        return yvec

    def logDensity(self, x):
        """Return the log probability of the Banana function"""
        return self.pdistrib.logDensity(self.getYvec(x))

    def gradLogDensity(self, x):
        y = self.getYvec(x)  # Uses x[1]
        grad = self.pdistrib.gradLogDensity(y)
        gradx0 = (
            grad[0] / self.alpha + grad[1] * self.alpha * self.beta * 2 * x[0]
        )
        gradx1 = grad[1] * self.alpha
        return np.array([gradx0, gradx1])


class donutDistrib(baseTargetDistrib):
    def __init__(self, xmin=-6, xmax=6, radius=2.6, sigma2=0.033):
        super().__init__(xmin, xmax)
        self.radius = radius
        self.sigma2 = sigma2

    def logDensity(self, x):
        rval = np.linalg.norm(x)
        return -1 * np.power(rval - self.radius, 2) / self.sigma2

    def gradLogDensity(self, x):
        rval = np.linalg.norm(x)
        return np.array([])
        gradx0 = x[0] * (self.radius / (rval - 1)) / self.sigma2
        gradx1 = x[1] * (self.radius / (rval - 1)) / self.sigma2
        return np.array([gradx0, gradx1])


class funnelDistrib(baseTargetDistrib):
    def __init__(self, xmin=-6, xmax=6):
        super().__init__(xmin, xmax)

    def f(self, x, m, s):
        return (
            -0.5 * np.log(2.0 * np.pi)
            - np.log(s)
            - 0.5 * np.power((np.array(x) - m) / s, 2)
        )

    def dfdx(self, x, m, s):
        return -(x - m) / np.power(s, 2)

    def dfds(self, x, m, s):
        return np.power(x - m, 2) - np.power(s, 2) / np.power(s, 3)

    def logDensity(self, x_):
        x = np.array([x_[1] - 2, x_[0]])
        m0 = m1 = 0
        s0 = 3
        s1 = np.exp(x[0] / 2)
        return self.f(x[0], m0, s0) + self.f([1], m1, s1)

    def gradLogDensity(self, x_):
        x = np.array(x_[1] - 2, x_[0])
        m0 = m1 = 0
        s0 = 3
        s1 = np.exp(x[0] / 2)
        return np.array(
            [
                self.dfdx(x[1], m1, np.exp(x[0] / 2)),
                self.dfds(x[0], m0, s0) + np.exp(x[0], 2),
            ]
            * self.dfds(x[1], m1, np.exp(x[0] / 2))
        )
