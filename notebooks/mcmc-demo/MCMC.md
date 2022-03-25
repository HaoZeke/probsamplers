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
import scipy.stats
from collections import namedtuple
import pandas as pd
```

```{code-cell} ipython3
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
```

## Internal dependencies

```{code-cell} ipython3
import probsamplers.distributions as pbsd
from probsamplers import aux
```

# MCMC Base Class

```{code-cell} ipython3
class baseChains(metaclass=abc.ABCMeta):        
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
    def __init__(self, dimensionsLike, targetDist, sigma = 1,):
        super().__init__()
        self.sigma = sigma
        self.dim = dimensionsLike.shape[0] # np.zeros(dim)
        self.targetDist = targetDist
        self.stepNum = 0
        self.chain = [pbsd.mvn.MultivariateNormal.getMVNSample(self.dim)]
        self.traj = []
        
    def reset(self):
        self.chain = [pbsd.mvn.MultivariateNormal.getMVNSample(self.dim)]
        self.stepNum = 0
        
    def step(self):
        isaccept = None
        proposalDist = pbsd.mvn.MultivariateNormal(self.chain[-1], np.eye(self.dim) * self.sigma**2)
        proposal = proposalDist.getSample()
        logAcceptRatio = self.targetDist.logDensity(proposal)  - self.targetDist.logDensity(self.chain[-1])
        if (np.random.default_rng().standard_normal() < np.exp(logAcceptRatio)):
            self.chain.append(proposal)
            isaccept = True
        else:
            self.chain.append(self.chain[-1])
            isaccept = False
        self.traj.append(aux.structs.mcmcData(step = self.stepNum,
                            acceptance = isaccept,
                            proposal = proposal,
                            proposalDistCovariance = proposalDist.covMat))
        self.stepNum += 1
        
```

```{code-cell} ipython3
banana = pbsd.targets.rosenbrockBanana()
```

```{code-cell} ipython3
mhSampler = RandomWalkMetropolisHastingsMC(targetDist = banana, dimensionsLike = np.zeros(2))
```

```{code-cell} ipython3
while (mhSampler.stepNum < 100000):
    mhSampler.step()
```

```{code-cell} ipython3
scipy.stats.multivariate_normal([0.1, 0.2, 0.3], np.array([2, -1, 0, -1, 2, -1, 0, -1, 2]).reshape(3,3)).logpdf(1)
```

```{code-cell} ipython3
paccept=pd.DataFrame([(x.proposal[0], x.proposal[1]) for x in mhSampler.traj[3000:-1] if x.acceptance])
paccept.columns=["x", "y"]
```

```{code-cell} ipython3
prej=pd.DataFrame([(x.proposal[0], x.proposal[1]) for x in mhSampler.traj[3000:-1] if not x.acceptance])
prej.columns=["x", "y"]
```

```{code-cell} ipython3
#res = banana.plotDensity(xlim={"low": -8, "high": 10},
#                   ylim = {"low": -15, "high": 4})
plt.figure()
#plt.contour(res.xx, res.yy, res.zz)
plt.plot(prej.x, prej.y, 'ro')
plt.plot(paccept.x, paccept.y, 'bo')
```

```{code-cell} ipython3

```
