# TODO(epidag): the entire sims_pars.data subpackage is dead. Every module still
# imports from `epidag` (the predecessor package), which is not a dependency, so
# `import sims_pars.data` fails outright. Either port these modules onto
# sims_pars (frame / distribution / reg) or delete the subpackage.
from epidag.data.frame import *
from epidag.data.static import *
from epidag.data.timeseries import *
from epidag.data.datafunction import *
from epidag.data.reg import *

__author__ = 'TimeWz667'
