---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Libraries

```{code-cell} ipython3
import abc
import numpy as np
import scipy
```

## Internal dependencies

```{code-cell} ipython3
import probsamplers.distributions as pbsd
```

# MCMC

```{code-cell} ipython3
class baseTargetDistrib(metaclass=abc.ABCMeta):
    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.rng = np.random.default_rng()
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'logDensity') and
                callable(subclass.logDensity) and
                hasattr(subclass, 'gradLogDensity') and
                callable(subclass.gradLogDensity) or
                NotImplemented)
    
    @abc.abstractmethod
    def logDensity(self, x):
        """Return the log of the probability density"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def gradLogDensity(self, x):
        """Returns the gradient of the probability density"""
        raise NotImplementedError
    
    def computeMean(self, chain):
        """Computes the chain mean"""
        return np.mean(chain) # TODO: figure this out
        
    def computeAutocorrelation(self, chain, lag):
        """Get the autocorrelation"""
        mean = self.computeMean(chain)
        autocov = np.zeros([lag, 1])
        for k in range(0, lag+1):
            for i in range(k, len(chain)):
                autocov[k] += np.dot((chain[i] - mean), (chain[i-k] - mean))
        return (autocov / autocov[0])
```

+++ {"tags": []}

# Rosenbrock function
Rosenbrock 'Banana' Function:

$ P(X) \propto {\rm exp} ( - \frac{1}{2a^2} (\sqrt{x_1^2 + x_2^2} -1 )^2  - \frac{1}{2b^2} ( x_2 - 1 )^2)$

where $a=0.1$ and $b=1.0$

```{code-cell} ipython3
class rosenbrockBanana(baseTargetDistrib):
    def __init__(self, xmin = -6, xmax = 6,
                 alpha=2, beta=0.2,
                mu = np.array([0, 4]),
                 cov = np.array([1, 0.5, 0.5, 1]).reshape(2, 2)
                ):
        super().__init__(xmin, xmax)
        self.alpha = alpha
        self.beta = beta
        self.pdistrib = scipy.stats.multivariate_normal(mu, cov)
        
    def getYvec(self, x):
        assert len(x) == 2
        yvec = np.zeros(2)
        yvec[0] = x[0] / self.alpha
        yvec[1] = x[1] * self.alpha + self.alpha * self.beta * (x[0]**2 + self.alpha**2)
        return yvec

    def logDensity(self, x):
        """ Return the log probability of the Banana function """
        return self.pdistrib.logpdf(self.getYvec(x))
    
    def gradLogDensity(self, x):
        y = self.getYvec(x) # Uses x[1]
        grad = self.gradLogDensity(y)
        gradx0 = grad[0] / self.alpha + grad[1] * self.alpha * self.beta * 2 * x[0]
        gradx1 = grad[1] * self.alpha
        return np.array([gradx0, gradx1])
        
```

```{code-cell} ipython3
a = rosenbrockBanana()
```

```{code-cell} ipython3
a.gradLogDensity([1,2])
```

```{code-cell} ipython3
rng.multivariate_normal([0,4], np.array([1, 0.5, 0.5, 1]).reshape(2,2))
```

```{code-cell} ipython3
scipy.stats.multivariate_normal([0,4], np.array([1, 0.5, 0.5, 1]).reshape(2,2)).rvs()
```

```{code-cell} ipython3

```

```{code-cell} ipython3
import probsamplers.distributions as pbs
```

```{code-cell} ipython3
dir(pbs)
```

```{code-cell} ipython3
pbs.mvn.MultivariateNormal()
```

```{code-cell} ipython3

```
