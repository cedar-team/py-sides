from __future__ import division
import pandas as pd
import numpy as np
import logging
import math
import pprint

from sklearn.linear_model import ElasticNet, Ridge

import scipy
from scipy.stats import rankdata

import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.sandbox.stats

import patsy

from variance_reduction import VarianceReduction
from differential_treatment_effect import DifferentialTreatmentEffect as DTE
from subgroup_eval_scores import SubgroupEvaluationComparisons as SEC

from sklearn.metrics import recall_score, precision_score


class SIDES():

    """
     This module aims to implement the SIDES recursive tree subgroup identification for clinical trials. It will try to predict and return the most likely subgroups in the provided set.

    Inputs :
    y         = full pandas dataset y values
    yleft     = split left group y dataframe
    yright     = split right group y dataframe


    Outputs :
    scores    = differential treatment effect associated with each split


    Attributes:
        score: The computed score
    """

    def __init__(self, min_group_size=25, max_group_size=140, no_of_groups=3, max_depth=2, rel_improve=0.5, verbose=False, multiplicity='bonferroni', binning=False, response='y'):
        """Initialize differential treatment effect"""
        self.verbose = verbose
        if self.verbose:
            logging.basicConfig(level=logging.DEBUG)

        self.response = response
        self.min_group_size = min_group_size
        self.no_of_groups = no_of_groups
        self.max_group_size = max_group_size
        self.max_depth = max_depth
        self.verbose = verbose
        self.rel_improve = rel_improve
        self.multiplicity = multiplicity

        self.return_subgroups = []

        # Initialize dependent modules
        self.vr = VarianceReduction()
        self.dte = DTE()
        self.sec = SEC()

    def get_best_groups(self, dataset, covariate_columns, parent=None):
        '''
        data, the dataframe rows in the group you wish to split, if the first step send in a full dataframe,
        covariate_columns, an array of your feature columns x1 - x40, might also include treatment interactions
        '''
        self.best_subgroups = []
        self.w = []
        self.mse = []
        self.mean = []
        self.meandiff = []
        self.pvalue = []
        self.x = []
        self.z = []
        self.splits = []
        self.pval = []
        self.effectsize = []
        self.subgroups = []
        self.rank = []
        self.depth = 0

        if parent != None:
            #Log if parent group exists
            logging.debug("Parent group exists")

        if parent == None:
            # save old index if no parent
            dataset['orgidx'] = dataset.index


        dataset.reset_index(inplace=True)

        # loop through all covariates and find best split
        logging.debug("Covariate columns to loop through %s", covariate_columns)
        for xcol in covariate_columns:
            if self.verbose:
                print "\n"
            logging.debug("Start split %s", xcol)
            # define temporary lists
            wx = []
            subgroupx = []
            meanx = []
            pvaluex = []
            effectsizex = []
            zx = []
            splitx = []
            msex = []
            sizex = []
            rankx = []
            meandiffx = []

            # define temporary dataframe per split
            tempDF = pd.DataFrame()

            # loop through all possible splits
            for val in dataset[xcol].unique():

                #split on x
                left = dataset[xcol] <= val
                right = dataset[xcol] > val

                grp_index = np.array([dataset[self.response][left].mean(), dataset[self.response][right].mean()]).argmin()

                # compute group sizes
                grp_size_cond = [len(dataset[self.response][left].index), len(dataset[self.response][right].index)][grp_index]

                # Groupsize, restrict smaller groups
                if(grp_size_cond < self.min_group_size):
                     continue

                # Groupsize, restrict bigger groups
                if(grp_size_cond > self.max_group_size):
                     continue

                # split subgroups mean
                mean_left = dataset[self.response][left].mean()
                mean_right = dataset[self.response][right].mean()

                subgroup = [dataset[left],dataset[right]][grp_index]
                parentgroup =[dataset[left],dataset[right]][1-grp_index]

                # Remove all groups which have a mean treatment effect difference of more than -0.6
                _, meandiff = self.sec.compare("two", dataset, subgroup, parentgroup)
                if(meandiff > -0.6):
                    continue
                #add mean diff as comparison parameter
                meandiffx.append(meandiff)

                #add group size
                sizex.append(grp_size_cond)

                # add split value
                splitx.append(val)

                # compute differential treatment effect score
                pvleft, pvright, w = self.dte.compute_score(dataset[self.response], dataset[self.response][left], dataset[self.response][right])
                wx.append(w)

                # split subgroups mean
                mean_left = dataset[self.response][left].mean()
                mean_right = dataset[self.response][right].mean()

                # split effect size
                cohens_d = (dataset[self.response][left].mean() - dataset[self.response][right].mean()) / (math.sqrt((dataset[self.response][left].std() ** 2 + dataset[self.response][right].std() ** 2) / 2))
                effectsizex.append(cohens_d)

                # compute split variance reduction score
                mse_score = self.vr.compute_score(dataset[self.response], dataset[self.response][left], dataset[self.response][right])
                msex.append(mse_score)

                # pvalue for the groups tested against each other (two-tailed)
                zstats, pvalue = scipy.stats.ttest_ind(dataset[self.response][left], dataset[self.response][right])
                zx.append(zstats)

                # select group with lowest mean for the split
                meanx.append([dataset[self.response][left].mean(), dataset[self.response][right].mean()][grp_index])

                # compute pvalue for chosen subgroup (two-tailed)
                pvalue = [pvleft, pvright][grp_index]
                pvaluex.append(pvalue)

                if parent != None:
                    # relative improvement in treatment effect (TE) in child
                    # versus parent
                    ypsilon = pvalue / parent['pvalue']
                    if ypsilon > self.rel_improve:
                        continue

                if parent != None:
                    # translate to old indexes
                    original_groups = (dataset['orgidx'].iloc[[dataset[self.response][left].index, dataset[self.response][right].index][grp_index]].values)
                    subgroupx.append(original_groups)
                else:
                    subgroupx.append([dataset[self.response][left].index, dataset[self.response][right].index][grp_index])

            # Bonferroni correction of multiple tests
            #logging.debug("Multiplicity p-value correction using %s", self.multiplicity)
            # if(len(pvaluex) > 0):
            #     # return corrected pvalues
            #     _, pvaluex, _, _ = statsmodels.sandbox.stats.multicomp.multipletests(
            #         pvaluex, alpha=0.05, method=self.multiplicity, is_sorted=False, returnsorted=False)

            # compute mean rank of split evaluation scores
            if(len(subgroupx) > 0):
                # add each score to a seperate column
                tempDF['mse'] = rankdata(msex)
                tempDF['w'] = rankdata(wx)
                #tempDF['pvalue'] = rankdata(pvaluex)
                #tempDF['mean'] = rankdata(meanx)
                #tempDF['effectsize'] = len(effectsizex) - rankdata(effectsizex).astype(int)

                # compute mean rank for each split
                for xrow in range(0, len(tempDF.index)):
                    rankx.append(tempDF.iloc[xrow].mean())

                #self.rank.append(rankx[np.argsort(rankx)[0]])
                self.x.append(xcol)
                logging.debug("Splits for this X %s: %s", xcol, splitx)
                idx = np.argsort(rankx)[0]

                self.subgroups.append(subgroupx[idx])
                self.z.append(zx[idx])
                self.pvalue.append(pvaluex[idx])
                self.effectsize.append(effectsizex[idx])
                self.mean.append(meanx[idx])
                self.meandiff.append(meandiffx[idx])

                self.w.append(wx[idx])
                self.splits.append(splitx[idx])
                self.mse.append(msex[idx])
        print "last mse and w"
        print len(self.mse)
        print len(self.w)

        if (self.depth < self.max_depth & len(self.best_subgroups)>0):
            for group in np.argsort(self.w)[:self.no_of_groups]:
                print "in for loop"
                print group
                #self.return_subgroups.append({'w': self.w[group], "mse": self.mse[group], 'groups': self.subgroups[group], 'mean': self.mean[group], 'xsplit': self.x[group], 'groupsize': len(self.subgroups[group]), 'ztest': self.z[group], 'pvalue': self.pvalue[group], 'effectsize': self.effectsize[group], 'splitvalue': self.splits[group]})
                self.best_subgroups.append({'w': self.w[group], "mse": self.mse[group], 'group': self.subgroups[group], 'mean': self.mean[group], 'xsplit': self.x[group], 'groupsize': len(self.subgroups[group]), 'ztest': self.z[group], 'pvalue': self.pvalue[group], 'effectsize': self.effectsize[group], 'splitvalue': self.splits[group]})
                print self.best_subgroups
                self.depth += 1
                if (self.depth < self.max_depth & len(self.best_subgroups)>0):
                    for subgroup in self.best_subgroups:
                        #print subgroup
                        #for group in np.argsort(self.w)[:self.no_of_groups]:
                            #self.return_subgroups.append({'w': self.w[group], "mse": self.mse[group], 'groups': self.subgroups[group], 'mean': self.mean[group], 'xsplit': self.x[group], 'groupsize': len(self.subgroups[group]), 'ztest': self.z[group], 'pvalue': self.pvalue[group], 'effectsize': self.effectsize[group], 'splitvalue': self.splits[group]})
                        #subgroup = {'w': self.w[group], "mse": self.mse[group], 'groups': self.subgroups[group], 'mean': self.mean[group], 'xsplit': self.x[group], 'groupsize': len(self.subgroups[group]), 'ztest': self.z[group], 'pvalue': self.pvalue[group], 'effectsize': self.effectsize[group], 'splitvalue': self.splits[group]}
                        self.return_subgroups.append(subgroup)
                        print "return groups", self.return_subgroups
                        print "depth", self.depth
                        remove_x_index = covariate_columns.index(subgroup['xsplit'])
                        del covariate_columns[remove_x_index]
                        self.get_best_groups(dataset.iloc[subgroup['group']], covariate_columns, parent=subgroup)
        else:
            if (self.depth==1):
                print "depth 1"
                self.return_subgroups = [{'w': self.w[group], "mse": self.mse[group], 'group': self.subgroups[group], 'mean': self.mean[group], 'xsplit': self.x[group], 'groupsize': len(self.subgroups[group]), 'ztest': self.z[group], 'pvalue': self.pvalue[group], 'effectsize': self.effectsize[group], 'splitvalue': self.splits[group]}]

            return self.return_subgroups

if __name__ == "__main__":
    submission_df = pd.DataFrame()

    for i in range(1, 5):
        print "Dataset: ", i
        df = pd.read_pickle('dataset_' + str(i) + '.df')
        df = df.reset_index(drop=True)

        feature_cols = [x for x in df.columns[4:]]
        orig_feature_cols = [x for x in df.columns[4:]]
        new_feature_cols = [x for x in df.columns[4:]]
        feature_cols_cont = [x for x in df.columns[24:]]
        feature_cols_cat = [x for x in df.columns[4:24]]

        # normalize contionuos X min/max scaling
        for k in feature_cols_cont:
            df[k] = (df[k] - df[k].min()) / (df[k].max() - df[k].min())

        # create a mask
        trt1 = df['trt'] == 1
        trt0 = df['trt'] == 0

        alpha = 0.1
        enet = ElasticNet(alpha=alpha, l1_ratio=0.5)
        enet.fit(df[trt1][feature_cols], df[trt1]['y'])

        ridge = Ridge(alpha=alpha)
        ridge.fit(df[trt0][feature_cols], df[trt0]['y'])

        # creating predicted Y when trt = 0
        for s in range(0, 240):
            if(df['trt'].iloc[s] == 0):
                X1 = np.reshape(
                    np.ravel(df.loc[s, feature_cols].as_matrix()), (1, -1))
                df.set_value(s, 'y1', enet.predict(X1))
            else:
                df.set_value(s, 'y1', df['y'].iloc[s])

        # creating predicted Y when trt = 0
        for s in range(0, 240):
            if(df['trt'].iloc[s] == 1):
                X1 = np.reshape(
                    np.ravel(df.loc[s, feature_cols].as_matrix()), (1, -1))
                df.set_value(s, 'y0', ridge.predict(X1))
            else:
                df.set_value(s, 'y0', df['y'].iloc[s])

        for s in range(0, 240):
            df.set_value(s, 'z', df['y1'].iloc[s] - df['y0'].iloc[s])

        selected_x = []
        for idu, k in enumerate(feature_cols):
            formula = 'y ~ trt + '
            if(idu < 20):
                formula += 'trt:C(' + k + ')'
            else:
                formula += 'trt:' + k

            mod1 = smf.glm(formula=formula, data=df,
                           family=sm.families.Gaussian()).fit()
            params = mod1.params
            del params['trt']
            del params['Intercept']

            for param in params:
                selected_x.append(param)

        formula = 'y ~ trt:' + \
            '+trt:'.join('C(%s)' % s for s in feature_cols_cat) + \
            '+trt:' + '+trt:'.join(feature_cols_cont) + '-1'

        y, X = patsy.dmatrices(formula, df, return_type='dataframe')
        feature_cols = X.columns
        feature_cols = feature_cols[1:]
        selected_features = list(np.argsort(selected_x)[:4]) + list(np.argsort(selected_x)[-4:])
        original_features = []
        #print "sorted x-interactions", sorted(selected_x)
        for feat in feature_cols[selected_features]:
            try:
                original_features.append(feat[feat.index('trt:x') + 4:])
            except Exception, e:
                original_features.append(
                    feat[feat.index('trt:C(x') + 6:feat.index(')')])

            original_features.append(
                [feat[feat.index('x'):feat.index('x') + 2]][0])

        original_features = set(original_features)
        original_features = list(original_features)
        #print "best features", feature_cols[selected_features]
        sides = SIDES(verbose=False, min_group_size=20, max_group_size=140, no_of_groups=1)
        subgroups = sides.get_best_groups(df, original_features)

        result =np.zeros(240)

        if subgroups != None:
            print subgroups
            subgroup = subgroups[0]
            for pid in subgroup['group']:
                result[pid] = 1

        submission_df.insert(i-1, 'dataset_'+ str(i), result)
