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
import numpy as np
import scipy.stats
```

```{code-cell} ipython3
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
```

## Internal dependencies

```{code-cell} ipython3
import probsamplers._aux as aux
```

```{code-cell} ipython3
class MultivariateNormal():
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
        while (w.res >= 1.0):
            w = draw()
        return w.x * np.sqrt((-2 * np.log(w.res)) / w.res)
    
    @staticmethod
    def getMVNSample(dim):
        dist = MultivariateNormal(np.zeros(dim), np.eye(dim))
        return dist.getSample()
    
    def getSample(self):
        z = np.zeros(self.dim)
        for i in range(0, len(z)):
            z[i] = MultivariateNormal.getNormal()
        return self.mean + (self.covL @ z)
    
    def logDensity(self, x):
        diff = self.mean - x
        eqsolv =  np.linalg.solve(self.covL, diff)
        return self.constant - np.linalg.norm(self.logDet - 0.5 * np.linalg.solve(self.covL.T, eqsolv))
    
    def gradLogDensity(self, x):
        diff = self.mean - x
        eqsolv2 = np.linalg.solve(self.covL, diff)
        return -1 * np.linalg.solve(self.covL, eqsolv2)
    
    def __repr__(self):
        return f"mu: {self.mean} \t logDet: {self.logDet} \n cov: {self.covMat}\n covL: {self.covL}\n"
```

```{code-cell} ipython3
np.log(np.linalg.cholesky(np.array([2, -1, -1, 2]).reshape(2,2)).diagonal()).sum()
```

```{code-cell} ipython3
#covL = np.linalg.cholesky(np.array([2, -1, -1, 2, -1, 2]).reshape(3,3))
```

```{code-cell} ipython3
pv = lambda: a.standard_normal() * 2 - 1
```

```{code-cell} ipython3
MultivariateNormal.getNormal()
```

```{code-cell} ipython3
#np.mean(covL @ np.array([MultivariateNormal.getNormal(), MultivariateNormal.getNormal()]))
```

```{code-cell} ipython3
np.array([MultivariateNormal.getNormal(), MultivariateNormal.getNormal()]).reshape(2,1)
```

```{code-cell} ipython3
a = MultivariateNormal(mu = [0.1, 0.2, 0.3], covMat = np.array([2, -1, 0, -1, 2, -1, 0, -1, 2]).reshape(3,3))
```

```{code-cell} ipython3
a.logDensity(1)
```

```{code-cell} ipython3
scipy.stats.multivariate_normal([0.1, 0.2, 0.3], np.array([2, -1, 0, -1, 2, -1, 0, -1, 2]).reshape(3,3)).logpdf(1)
```

```{code-cell} ipython3
a.gradLogDensity(1)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
x = np.linspace(-1, 3, 100)
y = np.linspace(0, 4, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
mu = np.array([1, 2])
cov = np.array([[.5, .25],[.25, .5]])
rv = MultivariateNormal(mu, covMat = cov)
Z = np.exp(rv.logDensity(pos))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
fig.show()
```

```{code-cell} ipython3
z = []
for p in pos:
    z.append(np.exp(rv.logDensity(p)))
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
