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
import copy
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
        self.traj = []
        
    def step(self):
        isaccept = None
        proposalDist = pbsd.mvn.MultivariateNormal(self.chain[-1], np.eye(self.dim) * self.sigma**2)
        proposal = proposalDist.getSample()
        logAcceptRatio = self.targetDist.logDensity(proposal)  - self.targetDist.logDensity(self.chain[-1])
        if (np.random.default_rng().uniform() < np.exp(logAcceptRatio)):
            self.chain.append(proposal)
            isaccept = True
        else:
            self.chain.append(self.chain[-1])
            isaccept = False
        self.traj.append(aux.structs.mcmcData(step = self.stepNum,
                            acceptance = isaccept,
                            proposal = copy.deepcopy(proposal),
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
while (mhSampler.stepNum < 1000):
    mhSampler.step()
```

```{code-cell} ipython3
paccept=pd.DataFrame([(x.proposal[0], x.proposal[1]) for x in mhSampler.traj[500:-1] if x.acceptance])
paccept.columns=["x", "y"]
```

```{code-cell} ipython3
prej=pd.DataFrame([(x.proposal[0], x.proposal[1]) for x in mhSampler.traj[500:-1] if not x.acceptance])
prej.columns=["x", "y"]
```

```{code-cell} ipython3
res = banana.plotDensity(xlim={"low": -8, "high": 10},
                   ylim = {"low": -15, "high": 4})
plt.figure()
plt.contour(res.xx, res.yy, res.zz)
plt.plot(prej.x, prej.y, 'ro')
plt.plot(paccept.x, paccept.y, 'bo')
```

# Hamiltonian Monte Carlo

```{code-cell} ipython3
class HamiltonianMC(baseChains):
    def __init__(self, dimensionsLike, targetDist, leapfrogSteps = 37, dt = 0.1):
        super().__init__()
        self.leapfrogSteps = 37
        self.dt = 0.1
        self.dim = dimensionsLike.shape[0] # np.zeros(dim)
        self.targetDist = targetDist
        self.stepNum = 0
        self.petmvn = pbsd.mvn.MultivariateNormal(np.zeros(2), np.eye(2))
        self.chain = [self.petmvn.getSample()]
        self.traj = []
        
    def reset(self):
        self.petmvn = pbsd.mvn.MultivariateNormal(np.zeros(2), np.eye(2))
        self.chain = [self.petmvn.getSample()]
        self.stepNum = 0
        self.traj = []
        
    def step(self):
        isaccept = None        
        q0 = self.chain[-1]
        p0 = pbsd.mvn.MultivariateNormal.getMVNSample(self.dim)
        if self.stepNum == 0:
            self.traj.append(aux.structs.mcmcData(step = self.stepNum,
                                             acceptance = True, 
                                             proposal = q0,
                                             proposalDistCovariance = None))
            self.stepNum += 1
            return
        # Otherwise..
        p = p0
        q = q0
        # Leapfrog
        # Half step
        p -= self.targetDist.gradLogDensity(q)*self.dt/2.0
        # Full steps for position and momentum
        for istep in range(self.leapfrogSteps):
            q += p * self.dt
            if (istep != self.leapfrogSteps-1):
                p -= self.targetDist.gradLogDensity(q)*self.dt
        # Half step
        p -= self.targetDist.gradLogDensity(q)*self.dt/2.0
        
        # Acceptance ratio
        H0 = self.targetDist.logDensity(q0) + self.targetDist.logDensity(p0)
        H = self.targetDist.logDensity(q) + self.targetDist.logDensity(p)
        logAcceptRatio = H - H0
        # Determine outcome
        if (np.random.default_rng().standard_normal() < np.exp(logAcceptRatio)):
            self.chain.append(q)
            isaccept = True
        else:
            self.chain.append(q0)
            isaccept = False
        self.traj.append(aux.structs.mcmcData(step = self.stepNum,
                            acceptance = isaccept,
                            proposal = copy.deepcopy(q),
                            proposalDistCovariance = None))
        self.stepNum += 1        
```

```{code-cell} ipython3
banana = pbsd.targets.rosenbrockBanana()
```

```{code-cell} ipython3
hmcSampler = HamiltonianMC(targetDist = banana, dimensionsLike = np.zeros(2))
```

```{code-cell} ipython3
hmcSampler.step()
hmcSampler.traj
```

```{code-cell} ipython3
while (hmcSampler.stepNum < 400):
    hmcSampler.step()
```

```{code-cell} ipython3
paccept=pd.DataFrame([(x.proposal[0], x.proposal[1]) for x in hmcSampler.traj[300:-1] if x.acceptance])
paccept.columns=["x", "y"]
```

```{code-cell} ipython3
prej=pd.DataFrame([(x.proposal[0], x.proposal[1]) for x in hmcSampler.traj[300:-1] if not x.acceptance])
prej.columns=["x", "y"]
```

```{code-cell} ipython3
res = banana.plotDensity(xlim={"low": -8, "high": 10},
                   ylim = {"low": -15, "high": 4})
plt.figure()
plt.contour(res.xx, res.yy, res.zz)
plt.plot(prej.x, prej.y, 'ro')
plt.plot(paccept.x, paccept.y, 'bo')
```

```{code-cell} ipython3

```
