# Differential treatment effects for Decision Trees Regression split evaluation

This function computes the differential treatment effect score associated with each split on the targets values. It is implemented to evaluate subgroups for exploratory subgroup analysis in clinical trials.

It expects the input to be panda dataframes (y, yleft and yright). Returns score as a float value.

Inputs :
y         = full pandas dataset y values
yleft     = split left group y dataframe
yright     = split right group y dataframe

Outputs :
scores    = differential treatment effect score for each split


For any improvements of this module feel free to do a pull request.