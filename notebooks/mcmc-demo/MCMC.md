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
import functools
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
    def __init__(self, targetDist):
        self.stepNum = 0
        self.targetDist = targetDist
        
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
    
    @functools.cached_property
    def plotDensity(self, xlim={"low": -8, "high": 10},
                   ylim = {"low": -15, "high": 4}):
        return self.targetDist.plotDensity
    
    def extractXYSamples(self, accepted=True, burnin=500):
        assert self.stepNum > burnin, f"Step must be greater than {burnin}"
        assert self.dim == 2, f"Samples only for dim == 2, got {self.dim}"
        if accepted==True:
            dat = pd.DataFrame([(x.proposal[0], x.proposal[1]) for x in self.traj[burnin:-1] if x.acceptance])
        else:
            dat = pd.DataFrame([(x.proposal[0], x.proposal[1]) for x in self.traj[burnin:-1] if not x.acceptance])
        dat.columns=["x", "y"]
        return dat
    
    def plotTargetSamples(self, xlim={"low": -8, "high": 10},
                   ylim = {"low": -15, "high": 4}):
        paccepted = self.extractXYSamples(accepted=True)
        prejected = self.extractXYSamples(accepted=False)
        res = self.plotDensity(xlim = xlim, ylim = ylim)
        fig = plt.figure()
        plt.contour(res.xx, res.yy, res.zz)
        plt.plot(prejected.x, prejected.y, 'ro')
        plt.plot(paccepted.x, paccepted.y, 'bo')
        return fig    
```

# Random Walk Monte Carlo

```{code-cell} ipython3
class RandomWalkMetropolisHastingsMC(baseChains):
    def __init__(self, dimensionsLike, targetDist, sigma = 1,):
        super().__init__(targetDist = targetDist)
        self.sigma = sigma
        self.dim = dimensionsLike.shape[0] # np.zeros(dim)
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
fig = mhSampler.plotTargetSamples()
fig.show()
```

# Hamiltonian Monte Carlo

```{code-cell} ipython3
class HamiltonianMC(baseChains):
    def __init__(self, dimensionsLike, targetDist, leapfrogSteps = 37, dt = 0.1):
        super().__init__(targetDist = targetDist)
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
        currentPositions = self.chain[-1]
        currentMomenta = pbsd.mvn.MultivariateNormal.getMVNSample(self.dim)
        if self.stepNum == 0:
            self.traj.append(aux.structs.mcmcData(step = self.stepNum,
                                             acceptance = True, 
                                             proposal = currentPositions,
                                             proposalDistCovariance = None))
            self.stepNum += 1
            return
        # Otherwise..
        propMomenta = currentMomenta
        propPositions = currentPositions
        # Leapfrog
        # Half step
        propMomenta -= self.targetDist.gradLogDensity(propPositions)*self.dt/2.0
        # Full steps for position and momentum
        for istep in range(self.leapfrogSteps):
            propPositions += propMomenta * self.dt
            if (istep != self.leapfrogSteps-1):
                propMomenta -= self.targetDist.gradLogDensity(propPositions)*self.dt
        # Half step
        propMomenta -= self.targetDist.gradLogDensity(propPositions)*self.dt/2.0
        
        # Acceptance ratio
        currentHamiltonian = self.targetDist.logDensity(currentPositions) + self.targetDist.logDensity(currentMomenta)
        propHamiltonian = self.targetDist.logDensity(propPositions) + self.targetDist.logDensity(propMomenta)
        logAcceptRatio = propHamiltonian - currentHamiltonian
        # Determine outcome
        if (np.random.default_rng().standard_normal() < np.exp(logAcceptRatio)):
            self.chain.append(propPositions)
            isaccept = True
        else:
            self.chain.append(currentPositions)
            isaccept = False
        self.traj.append(aux.structs.mcmcData(step = self.stepNum,
                            acceptance = isaccept,
                            proposal = copy.deepcopy(propPositions),
                            proposalDistCovariance = None))
        self.stepNum += 1        
```

```{code-cell} ipython3
hmcSampler = HamiltonianMC(targetDist =  pbsd.targets.rosenbrockBanana(),
                           dimensionsLike = np.zeros(2))
```

```{code-cell} ipython3
hmcSampler.step()
hmcSampler.traj
```

```{code-cell} ipython3
while (hmcSampler.stepNum < 600):
    hmcSampler.step()
```

```{code-cell} ipython3
fig = hmcSampler.plotTargetSamples()
fig.show()
```

```{code-cell} ipython3

```
