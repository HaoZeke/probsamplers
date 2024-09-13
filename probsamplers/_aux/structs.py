from collections import namedtuple

drawvals = namedtuple("drawvals", ["res", "x", "y"])
plotvals = namedtuple("plotvals", ["xx", "yy", "zz"])
mcmcData = namedtuple(
    "mcmcData", ["step", "acceptance", "proposal", "proposalDistCovariance"]
)
mcmcSamples = namedtuple("mcmcSamples", ["acceptedVals", "rejectedVals"])
