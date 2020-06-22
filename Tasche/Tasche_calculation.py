import pandas as pan
import numpy as np
import math
from scipy.optimize import fsolve,minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import csv
import json
import os
import sys
import time

num_sector = 2
num_cbf = 2

exposure1 = np.array([1, 1, 0.5])*0.01
exposure2 = np.array([5, 8, 10, 15, 20, 30, 100])*0.01
obligor1 = np.array([10000, 10000, 10000])
obligor2 = np.array([1000, 500, 100, 10, 2, 2, 1])
pd1 = np.array([0.005, 0.01, 0.01])
pd2 = np.array([0.01, 0.0175,0.0175,0.0125, 0.007, 0.003, 0.001])

exposure = np.array([exposure1, exposure2])
obligor = np.array([obligor1, obligor2])
pd = np.array([pd1, pd2])

# for MC
all_pd = np.array([np.repeat(pd1, obligor1), np.repeat(pd2, obligor2)])
all_exposure = np.array( [np.repeat(exposure1, obligor1), np.repeat(exposure2, obligor2) ])

EL = sum( exposure1*obligor1*pd1) + sum(exposure2*obligor2*pd2)
TE = sum(exposure1 * obligor1) + sum(exposure2 * obligor2)
mu1 = sum(obligor1*pd1)
mu2 = sum(obligor2*pd2)

sigma1hat = 0.16
sigma2hat = 0.36
sigma1 = 0.25
sigma2 = 0.04

alpha1 = (1/sigma1hat)**2 * 0.6 # thetak
alpha2 = (1/sigma2hat)**2 * 0.6
beta1 = sigma1hat ** 2           # delta
beta2 = sigma2hat ** 2
theta1 = 1/(sigma1**2) * 0.3
gamma11 = sigma1 ** 2
gamma12 = sigma1 ** 2
theta2 =  1/(sigma2**2) * 0.1
gamma21 = sigma2**2
gamma22 = sigma2**2

theta = np.array([alpha1, alpha2, theta1, theta2])
delta = np.array([beta1, beta2])
gamma = np.array( [[gamma11, gamma12], [gamma21, gamma22]] )

# alpha1 = (1/sigma1hat)**2
# alpha2 = (1/sigma2hat)**2
# beta1 = sigma1hat ** 2
# beta2 = sigma2hat ** 2
#
# theta = np.array([alpha1, alpha2])
# delta = np.array(([beta1, beta2]))
# gamma = np.array([])
# sigma1hat = 0.16
# sigma2hat = 0.56
# sigma1 = 0.32
# sigma2 = 0.32
#
# alpha1 = (1/sigma1hat)**2 * 0.3 # thetak
# alpha2 = (1/sigma2hat)**2 * 0.3
# beta1 = sigma1hat ** 2           # delta
# beta2 = sigma2hat ** 2
# theta1 = 1/(sigma1**2) * 0.7
# gamma11 = sigma1**2
# gamma12 = sigma1**2


est_spt = 1.6
est_var = 4.8
tailprob = 0.01


def Pk(t, k):
    """
    :param t:  the saddlepoint
    :param k:  the sector: 1 or 2
    """
    idx = k - 1
    ans = sum((np.exp(t*exposure[idx]) - 1) * obligor[idx] * pd[idx])
    return ans
def Pl(t, l):
    """
    :param l: the common beckground factors: 1, 2
    """
    idx = l - 1
    ans = gamma[idx][0] * Pk(t, 1) + gamma[idx][1] * Pk(t, 2)
    return ans

def Pk_div(t, k, n):
    """
    :param k: the sector: 1, 2
    :param n: the order of derivative
    """
    idx = k - 1
    ans = sum( (exposure[idx] ** n) * np.exp(t * exposure[idx]) * obligor[idx] * pd[idx])
    return ans
def Pl_div(t, l, n):
    """
    :param l: the common background factors: 1, 2
    :param n: the order of derivative
    """
    idx = l - 1
    ans = gamma[idx][0] * Pk_div(t, 1, n) + gamma[idx][1] * Pk_div(t, 2, n)
    return ans

def CBV_CGF(t):
    ans = 0
    for sector in np.arange(1, 3):
        # for the first two sector
        idx1 = sector -1  # 0, 1
        ans += - theta[idx1] * math.log( 1 - delta[idx1] * Pk(t, sector))

    for common_sector in np.arange(3, len(theta)+1):
        idx2 = common_sector -1  # 2, 3
        com_idx = common_sector - 2
        ans += - theta[idx2] * math.log( 1 - Pl(t, com_idx))
    return ans
def CBV_CGF_contri(t, contri_sector):
    """
    :param contri_sector: the sector need to add 1: 1, 2, 3, 4
    :return:
    """
    ans = 0
    for sector in np.arange(1, 3):
        # for the first two sector
        idx1 = sector - 1  # 0, 1
        if sector == contri_sector:
            ans += - (theta[idx1] +1) * math.log( 1 - delta[idx1] * Pk(t, sector))
        else:
            ans += - theta[idx1] * math.log(1 - delta[idx1] * Pk(t, sector))

    for common_sector in np.arange(3, len(theta)+1):
        idx2 = common_sector -1  # 2, 3
        com_idx = common_sector - 2
        if common_sector == contri_sector:
            ans += - (theta[idx2]+1) * math.log(1 - Pl(t, com_idx))
        else:
            ans += - theta[idx2] * math.log(1 - Pl(t, com_idx))
    return ans
def CBV_CGF_fir(t):
    ans = 0
    for sector in np.arange(1, 3):
        idx1 = sector - 1  # 0, 1
        Nk1 = Pk_div(t, sector, 1) / ( 1 - delta[idx1] * Pk(t, sector))
        ans += theta[idx1] * delta[idx1] * Nk1

    for common_sector in np.arange(3, len(theta) +1):
        idx2 = common_sector - 1 # 2, 3
        com_idx = common_sector - 2
        Nl1 = Pl_div(t, com_idx, 1) / ( 1 - Pl(t, com_idx))
        ans += theta[idx2] * Nl1
    return ans
def CBV_CGF_sed(t):
    ans = 0
    for sector in np.arange(1, 3):
        idx1 = sector - 1 # 0, 1
        Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx1] * Pk(t, sector))
        Nk2 = Pk_div(t, sector, 2) / (1 - delta[idx1] * Pk(t, sector))
        ans += theta[idx1] * delta[idx1] * (Nk2 + delta[idx1] * ( Nk1**2 ))

    for common_sector in np.arange(3, len(theta) + 1):
        idx2 = common_sector - 1 # 2, 3
        com_idx = common_sector - 2
        Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))
        Nl2 = Pl_div(t, com_idx, 2) / (1 - Pl(t, com_idx))
        ans += theta[idx2] * (Nl2 + (Nl1**2))
    return ans
def CBV_CGF_sed_contri(t, contri_sector):
    ans = 0
    for sector in np.arange(1, 3):
        idx1 = sector - 1 # 0，1
        Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx1] * Pk(t, sector))
        Nk2 = Pk_div(t, sector, 2) / (1 - delta[idx1] * Pk(t, sector))
        if sector == contri_sector:
            ans += (theta[idx1] + 1 ) * delta[idx1] * (Nk2 + delta[idx1]*( Nk1**2 ))
        else:
            ans += theta[idx1] * delta[idx1] * (Nk2 + delta[idx1]*(Nk1**2 ))

    for common_sector in np.arange(3, len(theta) + 1):
        idx2 = common_sector - 1  # 2， 3
        com_idx = common_sector - 2
        Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))
        Nl2 = Pl_div(t, com_idx, 2) / (1 - Pl(t, com_idx))
        if common_sector == contri_sector:
            ans += (theta[idx2] + 1) * (Nl2 + (Nl1**2))
        else:
            ans += theta[idx2] * (Nl2 + (Nl1**2))
    return ans
def CBV_CGF_thi(t):
    ans = 0
    for sector in np.arange(1, 3):
        idx1 = sector - 1 # 0， 1
        Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx1] * Pk(t, sector))
        Nk2 = Pk_div(t, sector, 2) / (1 - delta[idx1] * Pk(t, sector))
        Nk3 = Pk_div(t, sector, 3) / (1 - delta[idx1] * Pk(t, sector))
        ans += theta[idx1] * delta[idx1] * (Nk3 + 3*delta[idx1]*Nk1*Nk2 + 2*(delta[idx1]**2)*(Nk1**3))

    for common_sector in np.arange(3, len(theta) + 1):
        idx2 = common_sector - 1 # 2， 3
        com_idx = common_sector - 2
        Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))
        Nl2 = Pl_div(t, com_idx, 2) / (1 - Pl(t, com_idx))
        Nl3 = Pl_div(t, com_idx, 3) / (1 - Pl(t, com_idx))
        ans += theta[idx2] * (Nl3 + 3*Nl1*Nl2 + 2*(Nl1**3))
    return ans
def CBV_CGF_thi_contri(t, contri_sector):
    ans = 0
    for sector in np.arange(1, 3):
        idx1 = sector - 1 # 0， 1
        Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx1] * Pk(t, sector))
        Nk2 = Pk_div(t, sector, 2) / (1 - delta[idx1] * Pk(t, sector))
        Nk3 = Pk_div(t, sector, 3) / (1 - delta[idx1] * Pk(t, sector))
        if sector == contri_sector:
            ans += (theta[idx1]+1) * delta[idx1] * (Nk3 + 3*delta[idx1]*Nk1*Nk2 + 2*(delta[idx1]**2)*(Nk1**3))
        else:
            ans += theta[idx1] * delta[idx1] * (Nk3 + 3 * delta[idx1] * Nk1 * Nk2 + 2 * (delta[idx1] ** 2) * (Nk1 ** 3))

    for common_sector in np.arange(3, len(theta) + 1):
        idx2 = common_sector - 1 # 1， 2
        com_idx = common_sector - 2
        Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))
        Nl2 = Pl_div(t, com_idx, 2) / (1 - Pl(t, com_idx))
        Nl3 = Pl_div(t, com_idx, 3) / (1 - Pl(t, com_idx))
        if common_sector == contri_sector:
            ans += (theta[idx2]+1) * (Nl3 + 3*Nl1*Nl2 + 2*(Nl1**3))
        else:
            ans += theta[idx2] * (Nl3 + 3*Nl1*Nl2 + 2*(Nl1**3))
    return ans
def CBV_CGF_for(t):
    ans = 0
    for sector in np.arange(1, 3):
        idx1 = sector - 1
        Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx1] * Pk(t, sector))
        Nk2 = Pk_div(t, sector, 2) / (1 - delta[idx1] * Pk(t, sector))
        Nk3 = Pk_div(t, sector, 3) / (1 - delta[idx1] * Pk(t, sector))
        Nk4 = Pk_div(t, sector, 4) / (1 - delta[idx1] * Pk(t, sector))
        ans += theta[idx1]*delta[idx1]*(Nk4 + 3*delta[idx1]*(Nk2**2) + 4*delta[idx1]*Nk3*Nk1 +
                                    12*(delta[idx1]**2)*(Nk1**2)*Nk2 + 6*(delta[idx1]**3)*(Nk1**4))

    for common_sector in np.arange(3, len(theta) + 1):
        idx2 = common_sector - 1
        com_idx = common_sector - 2
        Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))
        Nl2 = Pl_div(t, com_idx, 2) / (1 - Pl(t, com_idx))
        Nl3 = Pl_div(t, com_idx, 3) / (1 - Pl(t, com_idx))
        Nl4 = Pl_div(t, com_idx, 4) / (1 - Pl(t, com_idx))
        ans += theta[idx2] * (Nl4 + 3*(Nl2**2) + 4*Nl3*Nl1 + 12*(Nl1**2)*Nl2 + 6*(Nl1**4))
    return ans
def CBV_CGF_for_contri(t, contri_sector):
    ans = 0
    for sector in np.arange(1, 3):
        idx1 = sector - 1 # 0， 1
        Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx1] * Pk(t, sector))
        Nk2 = Pk_div(t, sector, 2) / (1 - delta[idx1] * Pk(t, sector))
        Nk3 = Pk_div(t, sector, 3) / (1 - delta[idx1] * Pk(t, sector))
        Nk4 = Pk_div(t, sector, 4) / (1 - delta[idx1] * Pk(t, sector))
        if sector == contri_sector:
            ans += (theta[idx1]+1) * delta[idx1] * (Nk4 + 3 * delta[idx1] * (Nk2 ** 2) + 4 * delta[idx1] * Nk3 * Nk1 +
                        12 * (delta[idx1] ** 2) * (Nk1 ** 2) * Nk2 + 6 * (delta[idx1] ** 3) * (Nk1 ** 4))
        else:
            ans += theta[idx1]*delta[idx1]*(Nk4 + 3*delta[idx1]*(Nk2**2) + 4*delta[idx1]*Nk3*Nk1 +
                                    12*(delta[idx1]**2)*(Nk1**2)*Nk2 + 6*(delta[idx1]**3)*(Nk1**4))

    for common_sector in np.arange(3, len(theta) + 1):
        idx2 = common_sector - 1 # 2， 3
        com_idx = common_sector - 2
        Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))
        Nl2 = Pl_div(t, com_idx, 2) / (1 - Pl(t, com_idx))
        Nl3 = Pl_div(t, com_idx, 3) / (1 - Pl(t, com_idx))
        Nl4 = Pl_div(t, com_idx, 4) / (1 - Pl(t, com_idx))
        if common_sector == contri_sector:
            ans += (theta[idx2]+1) * (Nl4 + 3 * (Nl2 ** 2) + 4 * Nl3 * Nl1 + 12 * (Nl1 ** 2) * Nl2 + 6 * (Nl1 ** 4))
        else:
            ans += theta[idx2] * (Nl4 + 3*(Nl2**2) + 4*Nl3*Nl1 + 12*(Nl1**2)*Nl2 + 6*(Nl1**4))
    return ans
def CBV_CGF_fiv(t):
    ans = 0
    for sector in np.arange(1, 3):
        idx = sector - 1
        Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx] * Pk(t, sector))
        Nk2 = Pk_div(t, sector, 2) / (1 - delta[idx] * Pk(t, sector))
        Nk3 = Pk_div(t, sector, 3) / (1 - delta[idx] * Pk(t, sector))
        Nk4 = Pk_div(t, sector, 4) / (1 - delta[idx] * Pk(t, sector))
        Nk5 = Pk_div(t, sector, 5) / (1 - delta[idx] * Pk(t, sector))
        ans += theta[idx] * delta[idx] * (Nk5 + 5 * delta[idx] * Nk1 * Nk4 +
                                          10 * delta[idx] * Nk2 * Nk3 +
                                          30 * (delta[idx] ** 2) * Nk1 * (Nk2**2) +
                                          20 * (delta[idx] ** 2) * (Nk1 ** 2) * Nk3 +
                                          60 * (delta[idx] ** 3) * (Nk1 ** 3) * Nk2 +
                                          24 * (delta[idx] ** 4) * (Nk1 ** 5))

    for common_sector in np.arange(3, len(theta) + 1):
        idx = common_sector - 1
        com_idx = common_sector - 2
        Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))
        Nl2 = Pl_div(t, com_idx, 2) / (1 - Pl(t, com_idx))
        Nl3 = Pl_div(t, com_idx, 3) / (1 - Pl(t, com_idx))
        Nl4 = Pl_div(t, com_idx, 4) / (1 - Pl(t, com_idx))
        Nl5 = Pl_div(t, com_idx, 5) / (1 - Pl(t, com_idx))
        ans += theta[idx] * (Nl5 + 5 * Nl1 * Nl4 + 10 * Nl2 * Nl3 + 30 * Nl1 * (Nl2**2) +
                             20 * (Nl1 ** 2) * Nl3 + 60 * (Nl1 ** 3) * Nl2 + 24 * (Nl1 **5))
    return ans

def skewness():
    ans = CBV_CGF_thi(0) / (CBV_CGF_sed(0))**1.5
    return ans
def kurtosis():
    ans = CBV_CGF_for(0) / (CBV_CGF_sed(0))**2 + 3
    return ans

# VaR
def CBV_tailprob_to_vaR_first_order(var):
    """
    :return: given the var, return the corresponding spt, then return the CDF
    """

    def CBV_CGF_fir_changing(t):
        """
            this function is for given an x0, solve out t.
        """
        ans = 0
        for sector in np.arange(1, 3):
            idx1 = sector - 1 # 0, 1
            Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx1] * Pk(t, sector))
            ans += theta[idx1] * delta[idx1] * Nk1
        for common_sector in np.arange(3, len(theta) + 1):
            idx2 = common_sector - 1 # 2, 3
            com_idx = common_sector - 2
            Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))
            ans += theta[idx2] * Nl1
        return ans - var
    spt = fsolve(CBV_CGF_fir_changing, est_spt)

    uhat = spt * np.sqrt(CBV_CGF_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CBV_CGF(spt)))
    ans = norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat)
    return (1 - ans) - tailprob
def CBV_tailprob_to_VaR_second_order(var):
    """
    :return: given the var, return the corresponding spt, then return the CDF
    """
    def CBV_CGF_fir_changing(t):
        """
            this function is for given an x0, solve out t.
        """
        ans = 0
        for sector in np.arange(1, 3):
            idx1 = sector - 1
            Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx1] * Pk(t, sector))
            ans += theta[idx1] * delta[idx1] * Nk1
        for common_sector in np.arange(3, len(theta) + 1):
            idx2 = common_sector - 1
            com_idx = common_sector - 2
            Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))
            ans += theta[idx2] * Nl1
        return ans - var
    spt = fsolve(CBV_CGF_fir_changing, est_spt)

    uhat = spt * np.sqrt(CBV_CGF_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CBV_CGF(spt)))

    lambda3 = CBV_CGF_thi(spt) / (CBV_CGF_sed(spt) ** 1.5)
    lambda4 = CBV_CGF_for(spt) / (CBV_CGF_sed(spt) ** 2)
    temp = (1 / (what ** 3)) - (1 / (uhat ** 3)) - (lambda3 / (2 * (uhat ** 2))) + (1 / uhat) * (
            (lambda4 / 8) - (5 * (lambda3 ** 2) / 24))
    tp = 1 - (norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat - temp))
    return tp - tailprob
def CBV_CGF_fir_changing(t):
    """
        this function is for given an x0, solve out t.
    """
    ans = 0
    for sector in np.arange(1, 3):
        idx1 = sector - 1
        Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx1] * Pk(t, sector))
        ans += theta[idx1] * delta[idx1] * Nk1

    for common_sector in np.arange(3, len(theta) + 1):
        idx2 = common_sector - 1
        com_idx = common_sector - 2
        Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))
        ans += theta[idx2] * Nl1
    return ans - var

# ES
def CBV_ES1_first_order(var):

    def new_CGF_fir_changing(t):
        ans = CBV_CGF_fir(t) + CBV_CGF_sed(t) / CBV_CGF_fir(t)
        return ans - var

    t_ = fsolve(new_CGF_fir_changing, est_spt)
    KL_ = CBV_CGF(t_) + np.log(CBV_CGF_fir(t_)) - np.log(CBV_CGF_fir(0))
    KLsec_ = CBV_CGF_sed(t_) + (CBV_CGF_thi(t_) * CBV_CGF_fir(t_) - (CBV_CGF_sed(t_) ** 2)) / (CBV_CGF_fir(t_) ** 2)

    uhat = t_ * np.sqrt(KLsec_)
    what = np.sign(t_) * np.sqrt(2 * (t_ * var - KL_))
    cdf_ = norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat)
    ans = EL / tailprob * (1 - cdf_)
    return ans
def CBV_ES1_second_order(var):

    def new_CGF_fir_changing(t):
        ans = CBV_CGF_fir(t) + CBV_CGF_sed(t) / CBV_CGF_fir(t)
        return ans - var

    troot = fsolve(new_CGF_fir_changing, est_spt)
    KL_ = CBV_CGF(troot) + np.log(CBV_CGF_fir(troot)) - np.log(CBV_CGF_fir(0))
    KLsec_ = CBV_CGF_sed(troot) + (CBV_CGF_thi(troot) * CBV_CGF_fir(troot) - (CBV_CGF_sed(troot) ** 2)) / (CBV_CGF_fir(troot) ** 2)
    KLthi_ = CBV_CGF_thi(troot) + CBV_CGF_for(troot) / CBV_CGF_fir(troot) - \
             3 * CBV_CGF_sed(troot) *CBV_CGF_thi(spt) / (CBV_CGF_fir(troot)**2) + \
             2 * (CBV_CGF_sed(troot)**3) / (CBV_CGF_fir(troot)**3)
    KLfor_ = CBV_CGF_for(troot) + CBV_CGF_fiv(troot) / CBV_CGF_fir(troot) - \
             (4 * CBV_CGF_for(troot) * CBV_CGF_sed(troot) + 3 * (CBV_CGF_thi(troot)**2) ) / ( CBV_CGF_fir(troot)**2) + \
             12 * (CBV_CGF_sed(troot)**2) * CBV_CGF_thi(troot) / (CBV_CGF_fir(troot)**3) - \
             6 * (CBV_CGF_sed(troot)**4) / (CBV_CGF_fir(troot)**4)


    uhat = troot * np.sqrt(KLsec_)
    what = np.sign(troot) * np.sqrt(2 * (troot * var - KL_))
    lambda3 = KLthi_ / (KLsec_ ** 1.5)
    lambda4 = KLfor_ / (KLsec_ ** 2)
    temp = (1 / (what ** 3)) - (1 / (uhat ** 3)) - (lambda3 / (2 * (uhat ** 2))) + (1 / uhat) * (
            (lambda4 / 8) - (5 * (lambda3 ** 2) / 24))
    tp_ = 1 - (norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat - temp))

    ans = EL / tailprob * tp_
    return ans
def CBV_ES2(var, spt):
    uhat = spt * np.sqrt(CBV_CGF_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CBV_CGF(spt)))
    ans = EL * (1 - norm.cdf(what)) + norm.pdf(what) * ((var / uhat) - (EL / what))
    ans = ans / tailprob
    return ans
def CBV_ES3(var, spt):
    uhat = spt * np.sqrt(CBV_CGF_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CBV_CGF(spt)))
    ans = EL * (1 - norm.cdf(what)) + norm.pdf(what) * (
            (var / uhat) - (EL / what) + (EL - var) / what ** 3 + 1 / (spt * uhat))
    ans = ans / tailprob
    return ans

# check function
def CBV_check_function(var):
    """
    using the optimilization to find the minimum as the ES, and the root as VaR
    :param var:
    :return:
    """
    def CBV_CGF_fir_changing(t):
        """
            this function is for given an x0, solve out t.
        """
        ans = 0
        for sector in np.arange(1, 3):
            idx1 = sector - 1
            Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx1] * Pk(t, sector))
            ans += theta[idx1] * delta[idx1] * Nk1
        for common_sector in np.arange(3, len(theta) + 1):
            idx2 = common_sector - 1
            com_idx = common_sector - 2
            Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))
            ans += theta[idx2] * Nl1
        return ans - var
    spt = fsolve(CBV_CGF_fir_changing, est_spt)

    DEL = EL - var
    uhat = spt * np.sqrt(CBV_CGF_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CBV_CGF(spt)))
    ans = DEL * (1 - norm.cdf(what)) - norm.pdf(what) * ((DEL / what) - (DEL / (what ** 3)) - (1 / (spt * uhat)))
    ans = var + 1 / tailprob * ans
    return ans

# contribution
def CBV_VaRC_Tasche_first_order(var, est_spt):

    def CBV_CGF_fir_changing(t):
        """
            this function is for given an x0, solve out t.
        """
        ans = 0
        for sector in np.arange(1, 3):
            idx = sector - 1
            Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx] * Pk(t, sector))
            ans += theta[idx] * delta[idx] * Nk1

        for common_sector in np.arange(3, len(theta) + 1):
            idx = common_sector - 1
            com_idx = common_sector - 2
            Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))
            ans += theta[idx] * Nl1

        return ans - var
    spt = fsolve(CBV_CGF_fir_changing, est_spt)
    pdf = np.exp(CBV_CGF(spt) - spt * var) / np.sqrt(2 * math.pi * CBV_CGF_sed(spt))

    set_contribution = []
    # for this sector: one_sector
    for one_sector in np.arange(1, 3):
        # one_sector: the number of sector
        idx_sector = one_sector - 1 # 0， 1
        exposure_ = exposure[idx_sector]
        pd_ = pd[idx_sector]
        obligor_ = obligor[idx_sector]

        # for all obligors in this sector
        for one_idx in np.arange(0, len(exposure_)):
            num_pdf = 0
            one_exposure = exposure_[one_idx]
            one_pd = pd_[one_idx]
            one_obligor = obligor_[one_idx]

            def CBV_CGF_fir_changing1(t):
                """
                    this function is for given an x0, solve out t.
                """
                ans = 0
                for sector in np.arange(1, 3):
                    idx1 = sector - 1
                    Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx1] * Pk(t, sector))
                    if sector == one_sector:
                        ans += (theta[idx1] + 1) * delta[idx1] * Nk1
                    else:
                        ans += theta[idx1] * delta[idx1] * Nk1
                for common_sector in np.arange(3, len(theta) + 1):
                    idx2 = common_sector - 1
                    com_idx = common_sector - 2
                    Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))
                    ans += theta[idx2] * Nl1
                return ans - (var - one_exposure)
            spta = fsolve(CBV_CGF_fir_changing1, est_spt)
            num_pdf +=  np.exp(CBV_CGF_contri(spta, one_sector) - spta * (var - one_exposure)) / \
                   np.sqrt(2 * math.pi * CBV_CGF_sed_contri(spta, one_sector))

            for one_cbf in np.arange(3, len(theta) + 1):
                idx_cbf = one_cbf - 3
                def CBV_CGF_fir_changing2(t):
                    ans = 0
                    for sector in np.arange(1, 3):
                        idx1 = sector - 1
                        Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx1] * Pk(t, sector))
                        ans += theta[idx1] * delta[idx1] * Nk1
                    for common_sector in np.arange(3, len(theta) + 1):
                        idx2 = common_sector - 1
                        com_idx = common_sector - 2
                        Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))
                        if common_sector == one_cbf:
                            ans += (theta[idx2] + 1) * Nl1
                        else:
                            ans += theta[idx2] * Nl1
                    return ans - (var - one_exposure)
                sptb = fsolve(CBV_CGF_fir_changing2, est_spt)
                pdfb = np.exp(CBV_CGF_contri(sptb, one_cbf) - sptb * (var - one_exposure)) / \
                       np.sqrt(2 * math.pi * CBV_CGF_sed_contri(sptb, one_cbf))
                num_pdf += gamma[idx_cbf][idx_sector] * pdfb

            one_contri = num_pdf / pdf * one_exposure * one_pd
            set_contribution.append( (one_contri * one_obligor)[0])

    return set_contribution
def CBV_VaRC_Tasche_second_order(var, est_spt):

    def CBV_CGF_fir_changing(t):
        ans = 0
        for sector in np.arange(1, 3):
            idx = sector - 1
            Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx] * Pk(t, sector))
            ans += theta[idx] * delta[idx] * Nk1
        for common_sector in np.arange(3, len(theta) + 1):
            idx = common_sector - 1
            com_idx = common_sector - 2
            Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))
            ans += theta[idx] * Nl1
        return ans - var
    spt = fsolve(CBV_CGF_fir_changing, est_spt)
    rho3 = CBV_CGF_thi(spt) / (CBV_CGF_sed(spt) ** 1.5)
    rho4 = CBV_CGF_for(spt) / (CBV_CGF_sed(spt) ** 2)
    pdf = np.exp(CBV_CGF(spt) - spt * var) / np.sqrt(2 * math.pi * CBV_CGF_sed(spt)) * (
            1 + 1 / 8 * (rho4 - (5 / 3) * (rho3 ** 2)))

    set_contribution = []
    # for this sector: one_sector
    for one_sector in np.arange(1, num_sector+1):
        # one_sector: the number of sector
        idx_sector = one_sector - 1
        exposure_ = exposure[idx_sector]
        pd_ = pd[idx_sector]
        obligor_ = obligor[idx_sector]

        # for all obligors in this sector
        for one_idx in np.arange(0, len(exposure_)):
            num_pdf = 0
            one_exposure = exposure_[one_idx]
            one_pd = pd_[one_idx]
            one_obligor = obligor_[one_idx]

            def CBV_CGF_fir_changing1(t):
                ans = 0
                for sector in np.arange(1, 3):
                    idx = sector - 1
                    Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx] * Pk(t, sector))
                    if sector == one_sector:
                        ans += (theta[idx] + 1) * delta[idx] * Nk1
                    else:
                        ans += theta[idx] * delta[idx] * Nk1
                for common_sector in np.arange(3, len(theta) + 1):
                    idx = common_sector - 1
                    com_idx = common_sector - 2
                    Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))
                    ans += theta[idx] * Nl1
                return ans - (var - one_exposure)
            spta = fsolve(CBV_CGF_fir_changing1, est_spt)
            rho3a = CBV_CGF_thi_contri(spta, one_sector) / (CBV_CGF_sed_contri(spta, one_sector) ** 1.5)
            rho4a = CBV_CGF_for_contri(spta, one_sector) / (CBV_CGF_sed_contri(spta, one_sector) ** 2)
            num_pdf +=  np.exp(CBV_CGF_contri(spta, one_sector) - spta * (var - one_exposure)) / \
                        np.sqrt(2 * math.pi * CBV_CGF_sed_contri(spta, one_sector)) * (
                        1 + 1 / 8 * (rho4a - (5 / 3) * (rho3a ** 2)))

            for one_cbf in np.arange(3, len(theta) + 1):
                idx_cbf = one_cbf - 3

                def CBV_CGF_fir_changing2(t):
                    ans = 0
                    for sector in np.arange(1, 3):
                        idx = sector - 1
                        Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx] * Pk(t, sector))
                        ans += theta[idx] * delta[idx] * Nk1
                    for common_sector in np.arange(3, len(theta) + 1):
                        idx = common_sector - 1
                        com_idx = common_sector - 2
                        Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))
                        if common_sector == one_cbf:
                            ans += (theta[idx] + 1) * Nl1
                        else:
                            ans += theta[idx] * Nl1
                    return ans - (var - one_exposure)
                sptb = fsolve(CBV_CGF_fir_changing2, est_spt)
                rho3b = CBV_CGF_thi_contri(sptb, one_cbf) / (CBV_CGF_sed_contri(sptb, one_cbf) ** 1.5)
                rho4b = CBV_CGF_for_contri(sptb, one_cbf) / (CBV_CGF_sed_contri(sptb, one_cbf) ** 2)
                pdfb = np.exp(CBV_CGF_contri(sptb, one_cbf) - sptb * (var - one_exposure)) / \
                       np.sqrt(2 * math.pi * CBV_CGF_sed_contri(sptb, one_cbf))* (
                       1 + 1 / 8 * (rho4b - (5 / 3) * (rho3b ** 2)))
                num_pdf += gamma[idx_cbf][idx_sector] * pdfb

            one_contri = num_pdf / pdf * one_exposure * one_pd
            set_contribution.append( (one_contri * one_obligor)[0])
    return set_contribution
def CBV_ESC_Tasche_first_order(var, est_spt):

    def CBV_CGF_fir_changing(t):
        """
            this function is for given an x0, solve out t.
        """
        ans = 0
        for sector in np.arange(1, 3):
            idx = sector - 1
            Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx] * Pk(t, sector))
            ans += theta[idx] * delta[idx] * Nk1

        for common_sector in np.arange(3, len(theta) + 1):
            idx = common_sector - 1
            com_idx = common_sector - 2
            Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))
            ans += theta[idx] * Nl1

        return ans - var
    spt = fsolve(CBV_CGF_fir_changing, est_spt)
    uhat = spt * np.sqrt(CBV_CGF_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CBV_CGF(spt)))
    tp = 1 - (norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat))

    set_contribution = []
    # for this sector: one_sector
    for one_sector in np.arange(1, num_sector + 1):
        # one_sector: the number of sector
        idx_sector = one_sector - 1
        exposure_ = exposure[idx_sector]
        pd_ = pd[idx_sector]
        obligor_ = obligor[idx_sector]

        # for all obligors in this sector
        for one_idx in np.arange(0, len(exposure_)):
            num_tp = 0
            one_exposure = exposure_[one_idx]
            one_pd = pd_[one_idx]
            one_obligor = obligor_[one_idx]

            def CBV_CGF_fir_changing1(t):
                """
                    this function is for given an x0, solve out t.
                """
                ans = 0
                for sector in np.arange(1, 3):
                    idx = sector - 1
                    Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx] * Pk(t, sector))
                    if sector == one_sector:
                        ans += (theta[idx] + 1) * delta[idx] * Nk1
                    else:
                        ans += theta[idx] * delta[idx] * Nk1

                for common_sector in np.arange(3, len(theta) + 1):
                    idx = common_sector - 1
                    com_idx = common_sector - 2
                    Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))
                    ans += theta[idx] * Nl1

                return ans - (var - one_exposure)
            spta = fsolve(CBV_CGF_fir_changing1, est_spt)
            uhata = spta * np.sqrt(CBV_CGF_sed_contri(spta, one_sector))
            whata = np.sign(spta) * np.sqrt(2 * (spta * (var - one_exposure) - CBV_CGF_contri(spta, one_sector)))
            num_tp += 1 - (norm.cdf(whata) + norm.pdf(whata) * (1 / whata - 1 / uhata))

            for one_cbf in np.arange(3, len(theta) + 1):
                idx_cbf = one_cbf - 3

                def CBV_CGF_fir_changing2(t):
                    """
                        this function is for given an x0, solve out t.
                    """
                    ans = 0
                    for sector in np.arange(1, 3):
                        idx = sector - 1
                        Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx] * Pk(t, sector))
                        ans += theta[idx] * delta[idx] * Nk1

                    for common_sector in np.arange(3, len(theta) + 1):
                        idx = common_sector - 1
                        com_idx = common_sector - 2
                        Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))

                        if common_sector == one_cbf:
                            ans += (theta[idx] + 1) * Nl1
                        else:
                            ans += theta[idx] * Nl1
                    return ans - var
                sptb = fsolve(CBV_CGF_fir_changing2, est_spt)
                uhatb = sptb * np.sqrt(CBV_CGF_sed_contri(sptb, one_cbf))
                whatb = np.sign(sptb) * np.sqrt(2 * (sptb * (var - one_exposure) - CBV_CGF_contri(sptb, one_cbf)))
                tpb = 1 - (norm.cdf(whatb) + norm.pdf(whatb) * (1 / whatb - 1 / uhatb))
                num_tp += gamma[idx_cbf][idx_sector] * tpb

            one_contri = num_tp / tp * one_exposure * one_pd
            set_contribution.append((one_contri * one_obligor)[0])
    return set_contribution
def CBV_ESC_Tasche_second_order(var, est_spt):

    def CBV_CGF_fir_changing(t):
        """
            this function is for given an x0, solve out t.
        """
        ans = 0
        for sector in np.arange(1, 3):
            idx = sector - 1
            Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx] * Pk(t, sector))
            ans += theta[idx] * delta[idx] * Nk1

        for common_sector in np.arange(3, len(theta) + 1):
            idx = common_sector - 1
            com_idx = common_sector - 2
            Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))
            ans += theta[idx] * Nl1

        return ans - var
    spt = fsolve(CBV_CGF_fir_changing, est_spt)
    uhat = spt * np.sqrt(CBV_CGF_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CBV_CGF(spt)))
    lambda3 = CBV_CGF_thi(spt) / (CBV_CGF_sed(spt) ** 1.5)
    lambda4 = CBV_CGF_for(spt) / (CBV_CGF_sed(spt) ** 2)
    temp = (1 / (what ** 3)) - (1 / (uhat ** 3)) - (lambda3 / (2 * (uhat ** 2))) + (1 / uhat) * (
            (lambda4 / 8) - (5 * (lambda3 ** 2) / 24))
    tp = 1 - (norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat - temp))

    set_contribution = []
    # for this sector: one_sector
    for one_sector in np.arange(1, num_sector + 1):
        # one_sector: the number of sector
        idx_sector = one_sector - 1
        exposure_ = exposure[idx_sector]
        pd_ = pd[idx_sector]
        obligor_ = obligor[idx_sector]

        # for all obligors in this sector
        for one_idx in np.arange(0, len(exposure_)):
            num_tp = 0
            one_exposure = exposure_[one_idx]
            one_pd = pd_[one_idx]
            one_obligor = obligor_[one_idx]

            def CBV_CGF_fir_changing1(t):
                """
                    this function is for given an x0, solve out t.
                """
                ans = 0
                for sector in np.arange(1, 3):
                    idx = sector - 1
                    Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx] * Pk(t, sector))
                    if sector == one_sector:
                        ans += (theta[idx] + 1) * delta[idx] * Nk1
                    else:
                        ans += theta[idx] * delta[idx] * Nk1

                for common_sector in np.arange(3, len(theta) + 1):
                    idx = common_sector - 1
                    com_idx = common_sector - 2
                    Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))
                    ans += theta[idx] * Nl1

                return ans - (var - one_exposure)
            spta = fsolve(CBV_CGF_fir_changing1, est_spt)
            lambda3a = CBV_CGF_thi_contri(spta, one_sector) / (CBV_CGF_sed_contri(spta, one_sector) ** 1.5)
            lambda4a = CBV_CGF_for_contri(spta, one_sector) / (CBV_CGF_sed_contri(spta, one_sector) ** 2)
            uhata = spta * np.sqrt(CBV_CGF_sed_contri(spta, one_sector))
            whata = np.sign(spta) * np.sqrt(2 * (spta * (var - one_exposure) - CBV_CGF_contri(spta, one_sector)))
            tempa = (1 / (whata ** 3)) - (1 / (uhata ** 3)) - (lambda3a / (2 * (uhata ** 2))) + (1 / uhata) * (
                    (lambda4a / 8) - (5 * (lambda3a ** 2) / 24))
            num_tp += 1 - (norm.cdf(whata) + norm.pdf(whata) * (1 / whata - 1 / uhata - tempa))

            for one_cbf in np.arange(3, len(theta) + 1):
                idx_cbf = one_cbf - 3
                def CBV_CGF_fir_changing2(t):
                    """
                        this function is for given an x0, solve out t.
                    """
                    ans = 0
                    for sector in np.arange(1, 3):
                        idx = sector - 1
                        Nk1 = Pk_div(t, sector, 1) / (1 - delta[idx] * Pk(t, sector))
                        ans += theta[idx] * delta[idx] * Nk1

                    for common_sector in np.arange(3, len(theta) + 1):
                        idx = common_sector - 1
                        com_idx = common_sector - 2
                        Nl1 = Pl_div(t, com_idx, 1) / (1 - Pl(t, com_idx))

                        if common_sector == one_cbf:
                            ans += (theta[idx] + 1) * Nl1
                        else:
                            ans += theta[idx] * Nl1
                    return ans - (var - one_exposure)

                sptb = fsolve(CBV_CGF_fir_changing2, est_spt)
                lambda3b = CBV_CGF_thi_contri(sptb, one_cbf) / (CBV_CGF_sed_contri(sptb, one_cbf) ** 1.5)
                lambda4b = CBV_CGF_for_contri(sptb, one_cbf) / (CBV_CGF_sed_contri(sptb, one_cbf) ** 2)
                uhatb = sptb * np.sqrt(CBV_CGF_sed_contri(sptb, one_cbf))
                whatb = np.sign(sptb) * np.sqrt(2 * (sptb * (var - one_exposure) - CBV_CGF_contri(sptb, one_cbf)))
                tempb = (1 / (whatb ** 3)) - (1 / (uhatb ** 3)) - (lambda3b / (2 * (uhatb ** 2))) + (1 / uhatb) * (
                        (lambda4b / 8) - (5 * (lambda3b ** 2) / 24))
                tpb = 1 - (norm.cdf(whatb) + norm.pdf(whatb) * (1 / whatb - 1 / uhatb - tempb))
                num_tp += gamma[idx_cbf][idx_sector] * tpb

            one_contri = num_tp / tp * one_exposure * one_pd
            set_contribution.append((one_contri * one_obligor)[0])
    return set_contribution
def CBV_VARC_MARTIN(spt):
    set_contribution = []

    for one_sector in np.arange(1, num_sector + 1):
        idx_sector = one_sector - 1
        exposure_ = exposure[idx_sector]
        pd_ = pd[idx_sector]
        obligor_ = obligor[idx_sector]

        num_contri = delta[idx_sector] * theta[idx_sector] / (1 - delta[idx_sector] * Pk(spt, one_sector))

        for one_cbf in np.arange(3, len(theta) +1):
            num_contri += theta[one_cbf-1] * gamma[one_cbf - 3][one_sector - 1] / (1 - Pl(spt, one_cbf - 2))

        one_contri = obligor_ * exposure_ * pd_ * np.exp(spt * exposure_) * num_contri
        set_contribution.append(one_contri)
    return set_contribution[0].tolist() + set_contribution[1].tolist()

fir_set = []
t_set = np.arange(0, 1, 0.01)
for t in t_set:
    fir_set.append(CBV_CGF_fir(t))
check_set = pan.DataFrame({"t": t_set, "fir":fir_set})


start_time = time.time()
var = fsolve(CBV_tailprob_to_vaR_first_order, est_var)
print("--- %s seconds in MCIS---" % (time.time() - start_time))
start_time = time.time()
var2 = fsolve(CBV_tailprob_to_VaR_second_order, est_var)
print("--- %s seconds in MCIS---" % (time.time() - start_time))
spt = fsolve(CBV_CGF_fir_changing, est_spt)

start_time = time.time()
es1a = CBV_ES1_first_order(var)
print("--- %s seconds in MCIS---" % (time.time() - start_time))
es1b = CBV_ES1_second_order(var)
start_time = time.time()
es2 = CBV_ES2(var, spt)
print("--- %s seconds in MCIS---" % (time.time() - start_time))
start_time = time.time()
es3 = CBV_ES3(var, spt)
print("--- %s seconds in MCIS---" % (time.time() - start_time))

start_time = time.time()
init = np.array(5, dtype="float")
es_check = minimize(CBV_check_function, init, method='Nelder-Mead')
print("--- %s seconds in MCIS---" % (time.time() - start_time))


varc1 =CBV_VaRC_Tasche_first_order(var, est_spt)
varc2 =CBV_VaRC_Tasche_second_order(var, est_spt)
martin_varc = CBV_VARC_MARTIN(spt)

esc1 = CBV_ESC_Tasche_first_order(var, est_spt)
esc2 = CBV_ESC_Tasche_second_order(var, est_spt)

print([round(i,4) for i in varc1])
print([round(i,4) for i in varc2])
print([round(i,4) for i in martin_varc])
print([round(i,4) for i in esc1])
print([round(i,4) for i in esc2])
print(sum(varc1))
print(sum(varc2))
print(sum(martin_varc))
print(sum(esc1))
print(sum(esc2))

resultc = pan.DataFrame()
resultc["varc1"] = varc1
resultc["varc2"] = varc2
resultc["martin_varc"] = martin_varc
resultc["esc1"] = esc1
resultc["esc2"] = esc2

print("mean:", CBV_CGF_fir(0))
print("variance:", CBV_CGF_sed(0))
print("skewness:", skewness())
print("kurtosis:", kurtosis())


all_exposure = (exposure1* obligor1).tolist() + (exposure2* obligor2).tolist()
font2 = {'weight' : 'normal', 'size'   : 20,  }
esc24 = [round(i,4) for i in varc2]

def autolabel( rects, y):  # 在柱状图上面添加 数值
    i = 0
    for rect in rects:
        # 读出列表存储的value值
        value = y[i]
        x_1 = rect.get_x() + rect.get_width() / 2
        y_1 = rect.get_height()
        # x_1，y_1对应柱形的横、纵坐标
        i += 1
        plt.text(x_1, y_1, value, ha='center', va='bottom', fontdict={'size': 8})  # 在fontdict中设置字体大小
        rect.set_edgecolor('white')

plt.figure(figsize=(16,12))
N = len(varc1)
x = np.arange(1, N+1, 1)
y_offset = np.zeros(N)
index = np.arange(N) + 0.3
bar_width = 0.3
cmap = plt.get_cmap("tab20c")
colors = cmap(np.array([1, 3, 6]))
plt.ylim(0.0, 110)
a = plt.bar(x, all_exposure, bar_width, color=colors[2], bottom=y_offset,  label='exposure')
autolabel(a,all_exposure)
plt.ylabel('exposure', font2)
plt.xlabel("obligor", font2)

axes2 = plt.twinx()
b = axes2.bar(x+0.3, esc2, bar_width , color=colors[0], bottom=y_offset, label='esc')
axes2.bar(x+0.3, varc2,bar_width, color=colors[1], bottom=y_offset, label='varc')
axes2.set_ylim(0, 2.5)
autolabel(b,esc24)
axes2.set_ylabel('risk contribution', font2)
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize='xx-large')
plt.savefig("rc_fig.png")
plt.show()



# read current path
fileDir = os.path.join(os.getcwd())
sys.path.append(fileDir)

nsenario = 10
vden = 0
vnen = [0]* (len(all_exposure[0]) + len(all_exposure[1]))
eden = 0
enen = [0]* (len(all_exposure[0]) + len(all_exposure[1]))
contri_dict = {k:{'var_den': vden, 'var_nen': vnen, 'es_den': eden, 'es_nen':enen} for k in '123456789'}
contri_dict["10"] = {'var_den': vden, 'var_nen': vnen, 'es_den': eden,'es_nen':enen}
pathname = fileDir + "/Tasche/standard_crmodel_MC_result"
with open(pathname + "/ismc_data.csv", 'w') as csvfile:
    writer1 = csv.writer(csvfile)
    a = list(map(str, np.arange(1, nsenario + 1)))
    column_name=list(map(lambda x: "loss" + x, a)) + list(map(lambda x: "likehood" + x, a))
    writer1.writerow(column_name)
with open(pathname + "/pmc_loss.csv", 'w') as csvfile:
    writer2 = csv.writer(csvfile)
    senarios = np.arange(1, nsenario + 1)
    writer2.writerow(list(map(str, senarios)))
with open(pathname + '/ismc_contri.json', 'w') as json_file:
    json.dump(contri_dict, json_file)
with open(pathname + '/pmc_contri.json', 'w') as json_file:
    json.dump(contri_dict, json_file)


var_level = var
def MC_twist_para( t0):
    """
    this function is used to define the twist parameters in important sampling
    """
    twist = []  # the twisting terms
    for k in np.arange(0, num_sector):
        twist.append( Pk(t0, k + 1) )
    for l in np.arange(0, num_cbf):
        twist.append( Pl(t0, l + 1) )
    # changing the scale value
    scale = 1 / (1 - np.array(twist))
    return scale
def MCP(nsenario, nSim, PATH_MODEL, CONTRI = False):
    """
    :param nsenario: the number of senario
    :param nSim: the number of simulation in each senario
    :return: the set of loss
    """

    scale = np.ones(len(theta))
    # for all senario data
    sena = np.arange(1, nsenario+1)
    loss_senario_df = pan.DataFrame(columns=list(map(str, sena)))
    senario = 0

    if CONTRI:
        with open(PATH_MODEL + "/pmc_contri.json", 'r') as load_f:
            load_dict = json.load(load_f)

    while senario < nsenario:
        senario += 1
        if senario % 2 == 0:
            print("Congs! ----------The senario is: ", senario)
        # random gamma variable
        ssim = np.random.gamma(np.tile(theta, (nSim, 1)), np.tile(scale, (nSim, 1)))
        # weight
        weig = ssim[:, 0: num_sector] * delta
        for l in np.arange(0, num_cbf):
            weig += np.tile(gamma[l], (nSim, 1)) * ssim[:, len(delta) + l: len(delta) + l + 1]

        plambda = [ (np.transpose(np.tile(x, (nSim, 1))) * weig[:, w - 1]).tolist()
                    for w in np.arange(1, len(delta) + 1) for x in all_pd[w-1]]
        plambda = np.array(plambda)[:, 0, :]
        possion_lambda = np.transpose(plambda)
        pl = np.array([x for w in np.arange(1, len(delta) + 1) for x in all_exposure[w-1] ])
        rv = np.random.poisson(lam=possion_lambda)
        contri_loss = pl * rv
        loss_sim = np.sum(contri_loss, axis=1)
        loss_senario_df[str(senario)] = loss_sim

        if CONTRI:
            # VaR contribution
            index_up = np.where(var_level - loss_sim <= 0.01 * var_level)[0]
            index_down = np.where(var_level - loss_sim >= - 0.01 * var_level)[0]
            index_varc = list(set(index_up) & set(index_down))

            load_dict[str(senario)]["var_den"] += len(index_varc)
            var_nen_ = [contri_loss[ind, :] for ind in index_varc]
            var_nen = np.sum(var_nen_, axis=0)
            load_dict[str(senario)]["var_nen"] = (load_dict[str(senario)]["var_nen"] + var_nen).tolist()
            print("the efficient number of VARC:", len(index_varc))

            # ES contribution
            index_esc = np.where(loss_sim >= var_level)[0]
            load_dict[str(senario)]["es_den"] += len(index_esc)
            es_nen_ = [contri_loss[ind, :] for ind in index_esc]
            es_nen = np.sum(es_nen_, axis=0)
            load_dict[str(senario)]["es_nen"] = (load_dict[str(senario)]["es_nen"] + es_nen).tolist()
            print("the efficient number of ESC:", len(index_esc))

    loss_senario_df.to_csv( PATH_MODEL + "/pmc_loss.csv", mode='a+', header=False, index=False, sep=',')
    print("success write the portfolio loss data")
    if CONTRI:
        with open(PATH_MODEL + '/pmc_contri.json', 'w') as json_file:
            json.dump(load_dict, json_file)
            print("success save the contribution data")
    return loss_senario_df
def MCIS(nsenario, nSim, PATH_MODEL, CONTRI = False):

    #t0 = fsolve(CBV_CGF_fir_changing, est_spt)
    t0 = 0.2
    scale = MC_twist_para(t0) # the twisting scale term for gamma distribution
    if any(scale)<0:
        print("negative value of scale")

    # for all senario data
    a = list(map(str, np.arange(1, nsenario + 1)))
    loss_senario_df = pan.DataFrame(columns=list(map(lambda x:"loss"+x, a))+list(map(lambda x:"likehood"+x, a)))
    if CONTRI:
        with open(PATH_MODEL+"/ismc_contri.json", 'r') as load_f:
            load_dict = json.load(load_f)
    senario = 0

    while senario < nsenario:
        senario += 1
        if senario % 2 == 0:
            print("Congs! ----------The senario is: ", senario)

        # random gamma variable
        ssim = np.random.gamma(np.tile(theta,(nSim,1)), np.tile(scale,(nSim,1)))
        weig = ssim[:, 0: num_sector] * delta
        for l in np.arange(0, num_cbf):
            weig += np.tile(gamma[l], (nSim, 1)) * ssim[:, len(delta) + l: len(delta) + l + 1]
        tlambda = [(np.transpose(np.tile( x*np.exp(y*t0), (nSim,1)))  * weig[:,w-1]).tolist()
                   for w in np.arange(1, num_sector +1) for x,y in zip(all_pd[w-1], all_exposure[w-1])]
        # shape: Num_obligor:1:nSim
        tlambda = np.array(tlambda)[:, 0, :]
        # shape: Num_obligor * nSim
        tilting_lambda = np.transpose(tlambda)
        pl = np.array( [x for w in np.arange(1, num_sector +1) for x in all_exposure[w-1]] )
        rv = np.random.poisson(lam = tilting_lambda)
        contri_loss = pl * rv
        loss_before = np.sum(contri_loss, axis=1)
        tilt_para = np.exp(-t0 * loss_before + CBV_CGF(t0))
        loss_senario_df["loss" + str(senario)] = loss_before
        loss_senario_df["likehood" + str(senario)] = tilt_para

        if CONTRI:
            # VaR contribution
            index_up = np.where(var_level - loss_before <=  0.01 * var_level)[0]
            index_down = np.where(var_level - loss_before >= - 0.01 * var_level)[0]
            index_varc = list(set(index_up) & set(index_down))
          #  index_varc = np.where(loss_before == self.x0)[0].tolist()
            load_dict[str(senario)]["var_den"] += sum(tilt_para[index_varc])
            vnen_ = [contri_loss[ind,:] * tilt_para[ind] for ind in index_varc]
            vnen_ = np.sum(vnen_, axis=0)
            load_dict[str(senario)]["var_nen"] = (load_dict[str(senario)]["var_nen"] + vnen_).tolist()
            print("the efficient number of VARC:", len(index_varc))

            # ES contribution
            index_esc = np.where(loss_before >= var_level)[0]
            load_dict[str(senario)]["es_den"] += sum(tilt_para[index_esc])
            enen_ = [ contri_loss[ind, :] * tilt_para[ind] for ind in index_esc]
            enen_ = np.sum(enen_, axis=0)
            load_dict[str(senario)]["es_nen"] = (load_dict[str(senario)]["es_nen"] + enen_).tolist()
            print("the efficient number of ESC:", len(index_esc))

    loss_senario_df.to_csv( PATH_MODEL + "/ismc_data.csv", mode='a+',header=False, index=False, sep=',')
    print("success write the portfolio loss data")
    if CONTRI:
        with open( PATH_MODEL + '/ismc_contri.json', 'w') as json_file:
            json.dump(load_dict, json_file)
            print("success save the contribution data")
    return loss_senario_df
def pmc_result(datap, tp):
    """
    this function is to generate the var and es at the tail probability level
    :param datap:
    :param tp: the tail probability
    """
    nsenario = 10
    sena = np.arange(1, nsenario + 1)

    pmc_dataframe = pan.DataFrame()
    pmc_dataframe["tp"] = tp
    for ss in sena:
        mcp = datap.iloc[:, ss - 1]
        ttvar = np.percentile(mcp, list(map(lambda x: (1 - x) * 100, tp)))
        tmask = list(map(lambda x: mcp >= x, ttvar))
        ttes = list(map(lambda x: sum(np.array(mcp)[x]) / len(np.array(mcp)[x]), tmask))

        pmc_dataframe["var" + str(ss)] = ttvar
        pmc_dataframe["es" + str(ss)] = ttes

    pmc_dataframe["var_mean"] = pmc_dataframe.loc[:, list(map(lambda x: "var" + str(x), sena))].mean(axis=1)
    pmc_dataframe["var_std"] = pmc_dataframe.loc[:, list(map(lambda x: "var" + str(x), sena))].std(axis=1)

    pmc_dataframe["es_mean"] = pmc_dataframe.loc[:, list(map(lambda x: "es" + str(x), sena))].mean(axis=1)
    pmc_dataframe["es_std"] = pmc_dataframe.loc[:, list(map(lambda x: "es" + str(x), sena))].std(axis=1)
    return pmc_dataframe.loc[:,["tp", "var_mean", "var_std", "es_mean", "es_std"]]
def mc_contri(load_dict_final):

    nsenario = 10
    sena = np.arange(1, nsenario + 1)
    a = list(map(str, sena))
    contr_df = pan.DataFrame(columns=list(map(lambda x: "varc" + x, a)) + list(map(lambda x: "esc" + x, a)))
    final_contri_df = pan.DataFrame(columns=list(map(lambda x: "varc" + x, a)) + list(map(lambda x: "esc" + x, a)))
    for ss in sena:
        sena_dict = load_dict_final[str(ss)]
        varc = [x / sena_dict["var_den"] for x in sena_dict["var_nen"]]
        esc = [x / sena_dict["es_den"] for x in sena_dict["es_nen"]]
        temp_varc =  list(map(lambda x: x, varc))
        temp_esc = list(map(lambda x: x, esc))

        # for obligors
        list1 = [sum(temp_varc[:10000]), sum(temp_varc[10000:20000]), sum(temp_varc[20000:30000]),
                 sum(temp_varc[30000:31000]), sum(temp_varc[31000:31500]), sum(temp_varc[31500:31600]),
                 sum(temp_varc[31600:31610]), sum(temp_varc[31610:31612]), sum(temp_varc[31612:31614]),
                 sum(temp_varc[31614:31615])]
        list2 = [sum(temp_esc[:10000]), sum(temp_esc[10000:20000]), sum(temp_esc[20000:30000]),
                 sum(temp_esc[30000:31000]), sum(temp_esc[31000:31500]), sum(temp_esc[31500:31600]),
                 sum(temp_esc[31600:31610]), sum(temp_esc[31610:31612]), sum(temp_esc[31612:31614]),
                 sum(temp_esc[31614:31615])]
        final_contri_df["varc" + str(ss)] = list1
        final_contri_df["esc" + str(ss)] = list2

    final_contri_df["varc_mean"] = final_contri_df.loc[:, list(map(lambda x: "varc" + str(x), sena))].mean(axis=1)
    final_contri_df["varc_std"] = final_contri_df.loc[:, list(map(lambda x: "varc" + str(x), sena))].std(axis=1) \
                           / final_contri_df.varc_mean.values
    final_contri_df["esc_mean"] = final_contri_df.loc[:, list(map(lambda x: "esc" + str(x), sena))].mean(axis=1)
    final_contri_df["esc_std"] = final_contri_df.loc[:, list(map(lambda x: "esc" + str(x), sena))].std(axis=1) \
                          / final_contri_df.esc_mean.values
    return final_contri_df.loc[:, ["varc_mean", "varc_std", "esc_mean", "esc_std"]]
def ismc_result(datais, losshood):
    """
    this function has two kinds of output:
    when SIGNLE = TRUE, the output is the var and es at a certain loss level;
    when SIGNLE = FALSE, the output is the mean and st of tp and es at a range of var
    :param datais:
    :param losshood:
    :param SINGLE: choose to output a single value of a range of value
    :param level: the losshood when SINGLE is TRUE
    """
    nsenario = 10
    sena = np.arange(1, nsenario + 1)

    ismc_dataframe = pan.DataFrame()
    ismc_dataframe["var"] = losshood

    for ss in sena:
        lossis = datais.iloc[:, ss - 1]
        lhis = datais.iloc[:, ss + 9]
        mask = list(map(lambda x: lossis >= x, losshood))
        tpis = list(map(lambda x: (sum(np.array(lhis)[x])) / len(lossis), mask))
        esis = list(map(lambda x: np.mean(np.array(lossis)[x]), mask))
        ismc_dataframe["tp" + str(ss)] = tpis
        ismc_dataframe["es" + str(ss)] = esis
    ismc_dataframe["tp_mean"] = ismc_dataframe.loc[:, list(map(lambda x: "tp"+str(x), sena))].mean(axis = 1)
    ismc_dataframe["tp_std"] = ismc_dataframe.loc[:, list(map(lambda x: "tp"+str(x), sena))].std(axis = 1) \
                            /ismc_dataframe.tp_mean.values
    ismc_dataframe["es_mean"] = ismc_dataframe.loc[:, list(map(lambda x: "es" + str(x), sena))].mean(axis=1)
    ismc_dataframe["es_std"] = ismc_dataframe.loc[:, list(map(lambda x: "es" + str(x), sena))].std(axis=1) \
                            / ismc_dataframe.es_mean.values
    ans = ismc_dataframe.loc[:,["var", "tp_mean", "tp_std", "es_mean", "es_std"]]

    return ans

start_time = time.time()
pmc_res = MCP(10, 10000, pathname, CONTRI = True)
print("--- %s seconds in MCIS---" % (time.time() - start_time))
start_time = time.time()
ismc_res = MCIS(10, 5000, pathname, CONTRI = True)
print("--- %s seconds in MCIS---" % (time.time() - start_time))

tp = np.arange(0.001, 0.11, 0.01)
datap = pan.read_csv(os.path.join(pathname, 'pmc_loss.csv'))
pmc_ans = pmc_result(datap, tp)

losshood = np.arange(4.3, 6.5, 0.01)
datais = pan.read_csv(os.path.join(pathname, 'ismc_data.csv'))
ismc_ans = ismc_result(datais, losshood)

with open(os.path.join(pathname, 'pmc_contri.json'), 'r') as load_f:
    load_dict_p = json.load(load_f)
pmc_contri = mc_contri(load_dict_p)
with open(os.path.join(pathname, 'ismc_contri.json'), 'r') as load_f:
    load_dict_is = json.load(load_f)
ismc_contri = mc_contri(load_dict_is)

plt.figure(1, figsize=(12, 8))
#   plt.ylim(2.8,3.25)
#plt.xlim(0.0001, 0.01)
#  plt.plot( pmc_ans.var_mean,pmc_ans.tp, "y", label="MC(P)")
plt.plot(np.arange(1, len(varc1)+1), pmc_contri.varc_mean - 1.95 * (pmc_contri.varc_std / np.sqrt(10)), "y--", label="MC(P)-95%CI")
plt.plot(np.arange(1, len(varc1)+1), pmc_contri.varc_mean + 1.95 * (pmc_contri.varc_std / np.sqrt(10)), "y--", label="MC(P)-95%CI")
plt.plot(np.arange(1, len(varc1)+1), varc1, label="SPA-first order")
plt.plot(np.arange(1, len(varc1)+1), varc2, label="SPA-second order")
plt.title("VaR contribution at 0.99", fontsize=20)
plt.xlabel('obligor', fontsize=15)
plt.ylabel('VaR risk contribution', fontsize=15)
plt.legend(fontsize=15)
plt.grid(linestyle='-.')
plt.savefig("varc_new.png")
plt.show()


plt.figure(1, figsize=(12, 8))
#   plt.ylim(2.8,3.25)
#plt.xlim(0.0001, 0.01)
#  plt.plot( pmc_ans.var_mean,pmc_ans.tp, "y", label="MC(P)")
plt.plot(np.arange(1, len(varc1)+1), pmc_contri.esc_mean - 1.95 * (pmc_contri.esc_std / np.sqrt(10)), "y--", label="MC(P)-95%CI")
plt.plot(np.arange(1, len(varc1)+1), pmc_contri.esc_mean + 1.95 * (pmc_contri.esc_std / np.sqrt(10)), "y--", label="MC(P)-95%CI")
plt.plot(np.arange(1, len(esc1)+1), esc1, label="SPA-first order")
plt.plot(np.arange(1, len(esc2)+1), esc2, label="SPA-second order")
plt.title("ESC contribution at 0.99", fontsize=20)
plt.xlabel('obligor', fontsize=15)
plt.ylabel('ESC risk contribution', fontsize=15)
plt.legend(fontsize=15)
plt.grid(linestyle='-.')
plt.savefig("varc_new1.png")
plt.show()


