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
import logging
import numpy as np
import scipy.stats
from collections import namedtuple
import pandas as pd
from codetiming import Timer
```

```{code-cell} ipython3
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib as mpl
```

## Internal dependencies

```{code-cell} ipython3
import probsamplers.distributions as pbsd
from probsamplers import _aux
```

# MCMC Base Class
We need an abstract base class where the methods can be defined for plotting and also to extract relevant information.

```{code-cell} ipython3
class baseChains(metaclass=abc.ABCMeta):      
    def __init__(self, targetDist):
        self.stepNum = 0
        self.targetDist = targetDist
        self.clock = Timer("steps", text="Time spent: {:.8f}", logger=logging.info)
        self.tryplot()

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
        
    def tryplot(self):
        """Determine if there is any point setting up plotting environments"""
        import sys
        import importlib
        if "matplotlib" in sys.modules or (spec := importlib.util.find_spec("matplotlib")) is not None:
            self.plotPrepareDefaults()
        return
        
    def plotPrepareDefaults(self):
        """Better defaults than normal"""
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        mpl.rcParams['figure.figsize'] = [12, 12]
        mpl.rcParams['figure.dpi'] = 72
        
        plt.style.use('fivethirtyeight')
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['figure.titlesize'] = 16
        
        width, height = plt.figaspect(1.68)
        fig = plt.figure(figsize=(width,height), dpi=400)
        
    def computeMean(self, chain):
        """Computes the chain mean"""
        return np.mean(chain) # TODO: figure this out
        
    def computeAutocorrelation(self, lagRange=201):
        """Get the autocorrelation"""
        def acf(x, lagRange):
            return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  for i in range(1, lagRange)])
        dat = self.extractAllXY()
        return (acf(dat.x, lagRange), acf(dat.y, lagRange))
    
    @functools.cached_property
    def extractDensity(self, xlim={"low": -8, "high": 10},
                   ylim = {"low": -15, "high": 4}):
        return self.targetDist.plotDensity
    
    def extractXYSamples(self, accepted=True, burnin=500):
        """Extract a pandas dataframe of the x,y pairs"""
        assert self.stepNum > burnin, f"Step must be greater than {burnin}"
        assert self.dim == 2, f"Samples only for dim == 2, got {self.dim}"
        if accepted==True:
            dat = pd.DataFrame([(x.proposal[0], x.proposal[1]) for x in self.traj[burnin:-1] if x.acceptance])
        else:
            dat = pd.DataFrame([(x.proposal[0], x.proposal[1]) for x in self.traj[burnin:-1] if not x.acceptance])
        dat.columns=["x", "y"]
        return dat
    
    def extractAllXY(self, burnin=500):
        dat = pd.DataFrame([(x.proposal[0], x.proposal[1]) for x in self.traj[burnin:-1]])
        dat.columns = ["x", "y"]
        return dat
    
    def plotTargetSamples(self, xlim={"low": -8, "high": 10},
                   ylim = {"low": -15, "high": 4}, title = ""):
        """Overlay samples on a target distribution"""
        paccepted = self.extractXYSamples(accepted=True)
        prejected = self.extractXYSamples(accepted=False)
        res = self.extractDensity(xlim = xlim, ylim = ylim)
        fig = plt.figure()
        plt.contour(res.xx, res.yy, res.zz)
        plt.plot(prejected.x, prejected.y, 'ro', alpha = 0.4, label = r'rejected')
        plt.plot(paccepted.x, paccepted.y, 'bo', alpha = 0.5, label = r'accepted')
        plt.legend()
        plt.title(title + '- Samples')
        plt.xlabel('X')
        plt.ylabel('Y')
        return fig
    
    def plotTrace(self, burnin=500, title = ""):
        """Generate a trace plot"""
        dat = self.extractAllXY(burnin=burnin)
        fig = plt.figure()
        plt.plot(np.arange(len(dat.x)), dat.x, label = r'$x$')
        plt.plot(np.arange(len(dat.y)), dat.y, label = r'$y$')
        plt.legend()
        plt.title(title + '- Trace Plot')
        plt.xlabel('Step')
        plt.ylabel('Trace')
        plt.xlim([0, self.stepNum-burnin]);
        return fig
    
    def plotAutocorrelations(self, lagRange=301, title = ""):
        acx, acy = self.computeAutocorrelation(lagRange)
        fig = plt.figure()
        kplot = np.arange(lagRange)
        plt.plot(kplot[1:], acx[1:], label=r'$x$')
        plt.plot(kplot[1:], acy[1:], label=r'$y$')
        plt.legend()
        plt.title(title + '- Samples')
        plt.xlabel('X')
        plt.ylabel('Y')
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
        
    @Timer(name="Metropolis-Hastings Steps", logger=None)
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
        self.traj.append(_aux.structs.mcmcData(step = self.stepNum,
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
mhSampler.clock.timers
```

```{code-cell} ipython3
fig = mhSampler.plotAutocorrelations(title="Metropolis Hastings")
```

```{code-cell} ipython3
fig = mhSampler.plotTargetSamples(title="Metropolis-Hastings")
#fig.show()
```

```{code-cell} ipython3
fig = mhSampler.plotTrace(title="Metropolis-Hastings")
# fig.show()
```

# Hamiltonian Monte Carlo

```{code-cell} ipython3
class HamiltonianMC(baseChains):
    def __init__(self, dimensionsLike, targetDist, leapfrogSteps = 37, dt = 0.1):
        super().__init__(targetDist = targetDist)
        self.leapfrogSteps = leapfrogSteps
        self.dt = dt
        self.dim = dimensionsLike.shape[0] # np.zeros(dim)
        self.targetDist = targetDist
        self.petmvn = pbsd.mvn.MultivariateNormal(np.zeros(2), np.eye(2))
        self.chain = [self.petmvn.getSample()]
        self.traj = []
        
    def reset(self):
        self.petmvn = pbsd.mvn.MultivariateNormal(np.zeros(2), np.eye(2))
        self.chain = [self.petmvn.getSample()]
        self.stepNum = 0
        self.traj = []
        
    @Timer(name="HMC Steps", logger=None)    
    def step(self):
        isaccept = None        
        currentPositions = self.chain[-1]
        currentMomenta = pbsd.mvn.MultivariateNormal.getMVNSample(self.dim)
        if self.stepNum == 0:
            self.traj.append(_aux.structs.mcmcData(step = self.stepNum,
                                             acceptance = True, 
                                             proposal = currentPositions,
                                             proposalDistCovariance = None))
            self.stepNum += 1
            return
        # Otherwise..
        propMomenta = currentMomenta
        propPositions = currentPositions
        # Leapfrog
        # Full steps for position and momentum
        with Timer(name="HMC LeapFrog", logger=None):
            # Half step
            propMomenta -= self.targetDist.gradLogDensity(propPositions)*self.dt/2.0
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
        self.traj.append(_aux.structs.mcmcData(step = self.stepNum,
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
#fig.show()fig = hmcSampler.plotTargetSamples()
```

```{code-cell} ipython3
fig = hmcSampler.plotTrace()
# fig.show()
```

```{code-cell} ipython3
fig = hmcSampler.plotAutocorrelations(lagRange=40, title="Hamiltonian Monte Carlo")
```

```{code-cell} ipython3
hmcSampler.clock.timers
```

# HMC-NUTS

```{code-cell} ipython3
class hmcNUTS(baseChains):
    def __init__(self, dimensionsLike, targetDist, dt = 0.1, deltaMax = 1):
        super().__init__(targetDist = targetDist)
        self.deltaMax = deltaMax
        self.leapfrogSteps = 37
        self.dt = dt
        self.dim = dimensionsLike.shape[0] # np.zeros(dim)
        self.targetDist = targetDist
        self.petmvn = pbsd.mvn.MultivariateNormal(np.zeros(2), np.eye(2))
        self.chain = [self.petmvn.getSample()]
        self.traj = []
        
    def reset(self):
        self.petmvn = pbsd.mvn.MultivariateNormal(np.zeros(2), np.eye(2))
        self.chain = [self.petmvn.getSample()]
        self.stepNum = 0
        self.traj = []
        
   # @Timer(name="HMC-NUTS TreeBuilding", logger=None)           
    def buildTree(self, positions, momenta, cutoff, direction, j):        
        currentPositions = self.chain[-1]
        currentMomenta = pbsd.mvn.MultivariateNormal.getMVNSample(self.dim)
        propMomenta = currentMomenta
        propPositions = currentPositions
        if (j==0):
            # Leapfrog
            # Full steps for position and momentum
            # Half step
            propMomenta -= self.targetDist.gradLogDensity(propPositions)*self.dt/2.0
            for istep in range(self.leapfrogSteps):
                propPositions += propMomenta * self.dt
                if (istep != self.leapfrogSteps-1):
                    propMomenta -= self.targetDist.gradLogDensity(propPositions)*self.dt
            # Half step
            propMomenta -= self.targetDist.gradLogDensity(propPositions)*self.dt/2.0
            # Decider
            tmp_n = np.exp(self.targetDist.gradLogDensity(currentPositions) * (direction*self.dt)/2.)
            n_ = 1 if np.any(cutoff < tmp_n) else 0
            tmp_s = cutoff < np.exp(self.deltaMax + self.targetDist.logDensity(currentPositions) - np.linalg.norm(propMomenta))
            s_ = 1 if np.any(cutoff < tmp_s)  else 0
            if n_:
                return ({'q_p': propPositions,
                                  'p_p': propMomenta,
                                  'q_m': propPositions,
                                  'p_m': propMomenta,
                                  'q_': propPositions,
                                  'n_': n_,
                                  's_': s_})
        else:
            result = self.buildTree(propPositions, propMomenta, cutoff, direction, j-1)
            q_m = result.get("q_m")
            p_m = result.get("p_m")
            p_p = result.get("p_p")
            q_p = result.get("q_p")
            q_ = result.get("q_")
            n_ = result.get("n_")
            s_ = result.get("s_")
            if s_==1:
                if (direction==-1):
                    result = self.buildTree(q_m, p_m, cutoff, direction, j-1)
                    q_m = result.get("q_m")
                    p_m = result.get("p_m")
                    q__ = result.get("q_")
                    n__ = result.get("n_")
                    s__ = result.get("s_")
                else:
                    result = self.buildTree(q_p, p_p, cutoff, direction, j-1)
                    q_p = result.get("q_p")
                    p_p = result.get("p_p")
                    q__ = result.get("q_")
                    n__ = result.get("n_")
                    s__ = result.get("s_")
                if (np.random.default_rng().uniform() < n__ / (n_ + n__)):
                    q_ = q__
                s_ = s_ * s__ * (1 if np.dot((q_p - q_m), p_m) >=0 else 0) * (1 if np.dot((q_p - q_m), p_p) >= 0 else 0)
                n_ = n_ + n__
                if n_:
                    isaccept = True
                else:
                    isaccept = False
                self.traj.append(_aux.structs.mcmcData(step = self.stepNum, acceptance = True, proposal = copy.deepcopy(currentPositions), proposalDistCovariance = None))
            blah = {'q_p': q_p,
                     'p_p': p_p,
                     'q_m': q_m,
                     'p_m': p_m,
                     'q_': q_,
                     'n_': n_,
                     's_': s_}
            return blah
           
            
    @Timer(name="HMC-NUTS Steps", logger=None)    
    def step(self):
        isaccept = None
        p0 = pbsd.mvn.MultivariateNormal.getMVNSample(self.dim)
        u = np.random.default_rng().uniform() * np.exp(self.targetDist.logDensity(self.chain[-1])) - np.linalg.norm(p0) * 0.5
        q = copy.deepcopy(self.chain[-1])
        q_m = copy.deepcopy(self.chain[-1])
        q_p = copy.deepcopy(self.chain[-1])
        p_m = copy.deepcopy(p0)
        p_p = copy.deepcopy(p0)
        j = 0
        n = 1
        s = 1
        
        while (s==1):
            v = -1 * np.random.default_rng().uniform() - 0.5
            if (v == -1):
                result = self.buildTree(q_m, p_m, u, v, j)
                q_m = result.get("q_m")
                p_m = result.get("p_m")
                q_ = result.get("q_")
                n_ = result.get("n_")
                s_ = result.get("s_")
            else:
                result = self.buildTree(q_m, p_m, u, v, j)
                q_p = result.get("q_p")
                p_p = result.get("p_p")
                q_ = result.get("q_")
                n_ = result.get("n_")
                s_ = result.get("s_")
            if (s_ == 1 and np.random.default_rng().uniform() < n_ / n):
                q = copy.deepcopy(q_)
            s = s_ * (1 if np.dot((q_p - q_m), p_m) >=0 else 0) * (1 if np.dot((q_p - q_m), p_p) >= 0 else 0)
            n = n + n_
            j = j+1
            
            isaccept = True
            self.traj.append(_aux.structs.mcmcData(step = self.stepNum, acceptance = True, proposal = copy.deepcopy(q), proposalDistCovariance = None))
            self.stepNum += 1
        
            
    def plotTargetSamples(self, xlim={"low": -8, "high": 10},
                       ylim = {"low": -15, "high": 4}):
        """Efficient enough to never fail"""
        paccepted = self.extractXYSamples(accepted=True)
        res = self.extractDensity(xlim = xlim, ylim = ylim)
        fig = plt.figure()
        plt.contour(res.xx, res.yy, res.zz)
        plt.plot(paccepted.x, paccepted.y, 'bo', alpha = 0.5, label = r'accepted')
        plt.legend()
        plt.title("Efficient HMC" + '- Samples')
        plt.xlabel('X')
        plt.ylabel('Y')
        return fig
```

```{code-cell} ipython3
hmcSamplerNUTS = hmcNUTS(targetDist =  pbsd.targets.rosenbrockBanana(),
                           dimensionsLike = np.zeros(2))
```

```{code-cell} ipython3
hmcSamplerNUTS.step()
hmcSamplerNUTS.traj
```

```{code-cell} ipython3
while (hmcSamplerNUTS.stepNum < 6000):
    hmcSamplerNUTS.step()
```

```{code-cell} ipython3
fig = hmcSamplerNUTS.plotTrace()
```

```{code-cell} ipython3
fig = hmcSamplerNUTS.plotTargetSamples()
```

```{code-cell} ipython3
fig = hmcSamplerNUTS.plotAutocorrelations()
```

```{code-cell} ipython3
hmcSamplerNUTS.clock.timers
```

```{code-cell} ipython3

```
