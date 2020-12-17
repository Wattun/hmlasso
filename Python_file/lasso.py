import pandas.Dataframe as pd
import tiem
import numpy as np


def cal_sd(X):

    X_sum = X.sum(axis = 0, skipna = True)
    X_sum = np.array(X_sum)
    X_sum_minus1 = X_sum - 1
    X_var = X.var(axis = 1, skipna = True)
    X_var = np.array(X_var)
    X_denominator = X_sum * X_var
    X_caled = X_sum_minus1 / X_denominator

    X_sqrt = np.sqrt(X_caled)

def hmlasso(X, y, family="gaussian", impl="cpp",
            lambda_min_ration=1e-2, nlambda=100,
            n_lambda=None, min_eig_th=1e-6,
            use="pairwise.complete.obs",
            positify="diag", weight_power=1,
            eig_tol=1e-8, eig_maxitr=1e+8,
            mu=1, verbose=False
            ):

    n = len(X)
    eigen_Gamma = None

    t1 = time.time()
    if (X.isnull().values.sum() + np.count_nonzero(np.isnan(y))) > 0:
        # R apply(X, 2, ...) 2は列方向，1は行方向
        # na.rm nanを無視した計算を行う
        X_mean = X.mean(axis = 0, skipna = True)
        X_sd = cal_sd(X)
        