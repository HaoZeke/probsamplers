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

+++ {"tags": []}

# Libraries

```{code-cell} ipython3
import abc
import numpy as np
import scipy
from collections import namedtuple
```

```{code-cell} ipython3
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
```

## Internal dependencies

```{code-cell} ipython3
import probsamplers.distributions as pbsd
```

# MCMC Base Class

```{code-cell} ipython3
class baseChains(metaclass=abc.ABCMeta):
    def __init__(self):
        self.rng = np.random.default_rng()
        
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'reset') and
                callable(subclass.reset) and
                hasattr(subclass, 'step') and
                callable(subclass.step) or
                NotImplemented)
    
    @abc.abstractmethod
    def reset(self, x):
        """Return to initial MVN"""
        raise NotImplementedError
        
    @abc.abstractmethod
    def step(self, x):
        """Take a sampling step"""
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

# Random Walk Monte Carlo

```{code-cell} ipython3
class RandomWalkMetropolisHastingsMC(baseChains):
    def __init__(self, sigma, dimensions, targetDist):
        self.sigma = sigma
        self.dim = dimensions
        self.targetDist = targetDist
        self.chain = [pbsd.mvn.MultivariateNormal.getSample(self.dim)]
        
    def reset(self):
        self.chain = [pbsd.mvn.MultivariateNormal.getSample(self.dim)]
        
    def step(self):
        proposalDist = pbsd.mvn.MultivariateNormal(self.chain[-1], np.eye(self.dim) * self.sigma**2)
        proposal = proposalDist.getSample()
        logAcceptRatio = self.targetDist.logDensity(proposal)  - self.targetDist.logDensity(self.chain[-1])
        
        
```

```{code-cell} ipython3

```
