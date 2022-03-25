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
class baseChains():
    def __init__(self):
        self.rng = np.random.default_rng()
        
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

```{code-cell} ipython3

```
