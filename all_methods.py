import numpy as np
import pandas as pd
from scipy.stats import chi2
import scipy.stats as stats

from scipy.fft import *

import statistics
import scipy

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

def simpleKW(y, freq = 12):
    """
    SEASONALITY analysis with value: Kruskal_Wallis.
    """
    
    
    Rank = np.array(pd.Series(y).rank(method='average', na_option='keep'))
    extra = freq - len(Rank)%freq
    dat = np.concatenate((np.repeat(np.nan, extra), Rank))
    yMAT = dat.reshape((int(len(dat)/freq), freq))
    Nobs = np.apply_along_axis(lambda x: np.count_nonzero(~np.isnan(x)), 0, yMAT)
    R2n = np.power(np.apply_along_axis(np.nansum, 0, yMAT), 2)/Nobs
    H = 12/(sum(Nobs) * (sum(Nobs) + 1)) * sum(R2n) - 3 * (sum(Nobs) + 1)

    if sum(np.unique(Rank, return_counts=True)[1]>1) > 0:
        valor = np.unique(Rank, return_counts=True)[1]
        valor = valor[valor > 1]
        sumT = sum(np.power(valor, 3) - valor)
        Correction = 1 - sumT/(np.power(len(y),3) - len(y))
        H = H/Correction

    return 1 - chi2.cdf(H, freq-1)




def simpleCS (x, trend_type = "l"):
    """
    TREND analysis with pvalue: Cox_Stuart
    """
    
    #trend_type = "l" --> "decreasing trend"
    #trend_type = "r" --> "increasing trend"
    n0 = len(x)%2
    if n0 == 1:
        remover = int((len(x))/2)
        x = np.delete(x, int((len(x))/2))

    half = len(x)/2
    
    x1 = x[np.arange(0, half, dtype=int)]
    x2 = x[np.arange(half, len(x), dtype=int)]

    n = np.sum((x2 - x1) != 0)
    t = np.sum(x1 < x2)

    if trend_type == "l":
        pvalue = stats.binom.cdf(t, n, 0.5)
    else:
        pvalue = 1 - stats.binom.cdf(t - 1, n, 0.5)

    return pvalue

def simpleWW(x):
    """
    TREND analysis with pvalue: Wald_WolfWitz
    """
    threshold = statistics.median(x)

    if threshold in x:
        x = np.delete(x, np.where(x == threshold))

    z  = np.sign(x - threshold)
    n1 = np.sum(z > 0)
    n2 = np.sum(z < 0)
    N  = n1 + n2
    mu = 1 + 2 * n1 * n2/(n1 + n2)
    stdev = np.sqrt(2 * n1 * n2 * (2 * n1 * n2 - N)/(N * N * (N - 1)))

    rtam, rval = runsSeq(z)
    r1 = np.sum(np.array(rval) > 0)
    r2 = np.sum(np.array(rval) < 0)
    R = r1 + r2

    pv = scipy.stats.norm.cdf((R - mu)/stdev)

    return len(rval), n1, n2, N, pv


def runsSeq(x):
    # simple WW uses this method.
    tamanhos = []
    valores  = []
    temp = x[0]
    conta = 1

    for i in np.arange(1, len(x)):
        if x[i] == temp:
            conta += 1
        else:
            valores.append(temp)
            tamanhos.append(conta)
            temp = x[i]
            conta = 1

    valores.append(temp)
    tamanhos.append(conta)

    return tamanhos, valores