class baseChains(metaclass=abc.ABCMeta):
    def __init__(self, targetDist):
        self.stepNum = 0
        self.targetDist = targetDist
        self.tryplot()

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "reset")
            and callable(subclass.reset)
            and hasattr(subclass, "step")
            and callable(subclass.step)
            or NotImplemented
        )

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

        if (
            "matplotlib" in sys.modules
            or (spec := importlib.util.find_spec("matplotlib")) is not None
        ):
            self.plotPrepareDefaults()
        return

    def plotPrepareDefaults(self):
        """Better defaults than normal"""
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        mpl.rcParams["figure.figsize"] = [12, 12]
        mpl.rcParams["figure.dpi"] = 72

        plt.style.use("fivethirtyeight")
        plt.rcParams["font.size"] = 14
        plt.rcParams["axes.labelsize"] = 14
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["axes.titlesize"] = 14
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10
        plt.rcParams["legend.fontsize"] = 14
        plt.rcParams["figure.titlesize"] = 16

        width, height = plt.figaspect(1.68)
        fig = plt.figure(figsize=(width, height), dpi=400)

    def computeMean(self, chain):
        """Computes the chain mean"""
        return np.mean(chain)  # TODO: figure this out

    def computeAutocorrelation(self, lagRange=201):
        """Get the autocorrelation"""

        def acf(x, lagRange):
            return np.array(
                [1]
                + [np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, lagRange)]
            )

        dat = self.extractAllXY()
        return (acf(dat.x, lagRange), acf(dat.y, lagRange))

    @functools.cached_property
    def extractDensity(
        self, xlim={"low": -8, "high": 10}, ylim={"low": -15, "high": 4}
    ):
        return self.targetDist.plotDensity

    def extractXYSamples(self, accepted=True, burnin=500):
        """Extract a pandas dataframe of the x,y pairs"""
        assert self.stepNum > burnin, f"Step must be greater than {burnin}"
        assert self.dim == 2, f"Samples only for dim == 2, got {self.dim}"
        if accepted == True:
            dat = pd.DataFrame(
                [
                    (x.proposal[0], x.proposal[1])
                    for x in self.traj[burnin:-1]
                    if x.acceptance
                ]
            )
        else:
            dat = pd.DataFrame(
                [
                    (x.proposal[0], x.proposal[1])
                    for x in self.traj[burnin:-1]
                    if not x.acceptance
                ]
            )
        dat.columns = ["x", "y"]
        return dat

    def extractAllXY(self, burnin=500):
        dat = pd.DataFrame(
            [(x.proposal[0], x.proposal[1]) for x in self.traj[burnin:-1]]
        )
        dat.columns = ["x", "y"]
        return dat

    def plotTargetSamples(
        self,
        xlim={"low": -8, "high": 10},
        ylim={"low": -15, "high": 4},
        title="",
    ):
        """Overlay samples on a target distribution"""
        paccepted = self.extractXYSamples(accepted=True)
        prejected = self.extractXYSamples(accepted=False)
        res = self.extractDensity(xlim=xlim, ylim=ylim)
        fig = plt.figure()
        plt.contour(res.xx, res.yy, res.zz)
        plt.plot(prejected.x, prejected.y, "ro", alpha=0.4, label=r"rejected")
        plt.plot(paccepted.x, paccepted.y, "bo", alpha=0.5, label=r"accepted")
        plt.legend()
        plt.title(title + "- Samples")
        plt.xlabel("X")
        plt.ylabel("Y")
        return fig

    def plotTrace(self, burnin=500, title=""):
        """Generate a trace plot"""
        dat = self.extractAllXY(burnin=burnin)
        fig = plt.figure()
        plt.plot(np.arange(len(dat.x)), dat.x, label=r"$x$")
        plt.plot(np.arange(len(dat.y)), dat.y, label=r"$y$")
        plt.legend()
        plt.title(title + "- Trace Plot")
        plt.xlabel("Step")
        plt.ylabel("Trace")
        plt.xlim([0, self.stepNum - burnin])
        return fig

    def plotAutocorrelations(self, lagRange=301, title=""):
        acx, acy = self.computeAutocorrelation(lagRange)
        fig = plt.figure()
        kplot = np.arange(lagRange)
        plt.plot(kplot[1:], acx[1:], label=r"$x$")
        plt.plot(kplot[1:], acy[1:], label=r"$y$")
        plt.legend()
        plt.title(title + "- Samples")
        plt.xlabel("X")
        plt.ylabel("Y")
        return fig


class RandomWalkMetropolisHastingsMC(baseChains):
    def __init__(
        self,
        dimensionsLike,
        targetDist,
        sigma=1,
    ):
        super().__init__(targetDist=targetDist)
        self.sigma = sigma
        self.dim = dimensionsLike.shape[0]  # np.zeros(dim)
        self.chain = [pbsd.mvn.MultivariateNormal.getMVNSample(self.dim)]
        self.traj = []

    def reset(self):
        self.chain = [pbsd.mvn.MultivariateNormal.getMVNSample(self.dim)]
        self.stepNum = 0
        self.traj = []

    def step(self):
        isaccept = None
        proposalDist = pbsd.mvn.MultivariateNormal(
            self.chain[-1], np.eye(self.dim) * self.sigma**2
        )
        proposal = proposalDist.getSample()
        logAcceptRatio = self.targetDist.logDensity(
            proposal
        ) - self.targetDist.logDensity(self.chain[-1])
        if np.random.default_rng().uniform() < np.exp(logAcceptRatio):
            self.chain.append(proposal)
            isaccept = True
        else:
            self.chain.append(self.chain[-1])
            isaccept = False
        self.traj.append(
            aux.structs.mcmcData(
                step=self.stepNum,
                acceptance=isaccept,
                proposal=copy.deepcopy(proposal),
                proposalDistCovariance=proposalDist.covMat,
            )
        )
        self.stepNum += 1


class HamiltonianMC(baseChains):
    def __init__(self, dimensionsLike, targetDist, leapfrogSteps=37, dt=0.1):
        super().__init__(targetDist=targetDist)
        self.leapfrogSteps = 37
        self.dt = 0.1
        self.dim = dimensionsLike.shape[0]  # np.zeros(dim)
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
            self.traj.append(
                aux.structs.mcmcData(
                    step=self.stepNum,
                    acceptance=True,
                    proposal=currentPositions,
                    proposalDistCovariance=None,
                )
            )
            self.stepNum += 1
            return
        # Otherwise..
        propMomenta = currentMomenta
        propPositions = currentPositions
        # Leapfrog
        # Half step
        propMomenta -= (
            self.targetDist.gradLogDensity(propPositions) * self.dt / 2.0
        )
        # Full steps for position and momentum
        for istep in range(self.leapfrogSteps):
            propPositions += propMomenta * self.dt
            if istep != self.leapfrogSteps - 1:
                propMomenta -= (
                    self.targetDist.gradLogDensity(propPositions) * self.dt
                )
        # Half step
        propMomenta -= (
            self.targetDist.gradLogDensity(propPositions) * self.dt / 2.0
        )

        # Acceptance ratio
        currentHamiltonian = self.targetDist.logDensity(
            currentPositions
        ) + self.targetDist.logDensity(currentMomenta)
        propHamiltonian = self.targetDist.logDensity(
            propPositions
        ) + self.targetDist.logDensity(propMomenta)
        logAcceptRatio = propHamiltonian - currentHamiltonian
        # Determine outcome
        if np.random.default_rng().standard_normal() < np.exp(logAcceptRatio):
            self.chain.append(propPositions)
            isaccept = True
        else:
            self.chain.append(currentPositions)
            isaccept = False
        self.traj.append(
            aux.structs.mcmcData(
                step=self.stepNum,
                acceptance=isaccept,
                proposal=copy.deepcopy(propPositions),
                proposalDistCovariance=None,
            )
        )
        self.stepNum += 1
