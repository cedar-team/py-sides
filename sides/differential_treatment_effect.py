from __future__ import division

import numpy as np
import pandas as pd
import math

import scipy.stats

class DifferentialTreatmentEffect():
    """
     his function computes the differential treatment effect score associated with each split on the targets values.

    Inputs :
    y         = full pandas dataset y values
    yleft     = split left group y dataframe
    yright     = split right group y dataframe


    Outputs :
    scores    = differential treatment effect associated with each split


    Attributes:
        score: The computed score
    """

    def __init__(self):
        """Initialize differential treatment effect"""

    def compute_score(self, y, yleft, yright):
        """Computes differential treatment effects for split."""
        # z-value and p-value for left and right subgroup
        zleft, pvleft =  scipy.stats.ttest_1samp(yleft, y.mean())
        zright, pvright =  scipy.stats.ttest_1samp(yright, y.mean())

        #differential treatment effect coefficient
        w1 = 2* ( 1 - scipy.stats.norm.cdf( abs(zleft-zright)/math.sqrt(2) ) )
        w2 = 2*min([1 - scipy.stats.norm.cdf(zleft),1 - scipy.stats.norm.cdf(zright)])
        w3 = max([w1,w2])
        return pvleft, pvright, w3