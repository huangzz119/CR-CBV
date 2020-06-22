import pandas as pan
import numpy as np
import math
from scipy.optimize import fsolve,minimize
from scipy.stats import norm

num_sector = 2
num_cbf = 1

exposure1 = np.array([1, 1, 0.5])*0.01
exposure2 = np.array([5, 8, 10, 15, 20, 30])*0.01
obligor1 = np.array([10000, 10000, 10000])
obligor2 = np.array([1000, 500, 100, 10, 2, 1])
pd1 = np.array([0.005, 0.01, 0.01])
pd2 = np.array([0.01, 0.0175,0.0175,0.0125, 0.007, 0.003])

exposure = np.array([exposure1, exposure2])
obligor = np.array([obligor1, obligor2])
pd = np.array([pd1, pd2])

EL = sum( exposure1*obligor1*pd1) + sum(exposure2*obligor2*pd2)
TE = sum(exposure1 * obligor1) + sum(exposure2 * obligor2)
mu1 = sum(obligor1*pd1)
mu2 = sum(obligor2*pd2)

sigma1hat = 0.16
sigma2hat = 0.56
sigma1 = 0.26
sigma2 = 0.42

alpha1 = (1/sigma1hat)**2 * 0.5 # thetak
alpha2 = (1/sigma2hat)**2
beta1 = sigma1hat ** 2           # delta
beta2 = sigma2hat ** 2

theta1 = 1/(sigma1**2) * 0.5
gamma11 = sigma1**2
gamma12 = sigma1**2

theta2 = 1/(sigma2**2) * 0.25
gamma21 = sigma2**2
gamma22 = sigma2**2

theta = np.array([alpha1, alpha2, theta1])
delta = np.array([beta1, beta2])
gamma = np.array( [[gamma11, gamma12],[gamma21, gamma22]] )

est_spt = 1
est_var = 5
tailprob = 0.01


def P1(t):
    ans = sum((np.exp(t*exposure1) -1) * obligor1 * pd1)
    return ans
def P2(t):
    ans = sum((np.exp(t*exposure2) -1) * obligor2 * pd2)
    return ans
def P1_div(t,n):
    """
    :param n: the order of derivative
    """
    ans = sum((exposure1**n) * np.exp(t*exposure1) * obligor1 * pd1 )
    return ans
def P2_div(t,n):
    """
    :param n: the order of derivative
    """
    ans = sum((exposure2**n) * np.exp(t*exposure2) * obligor2 * pd2)
    return ans
def CGF(t):
    ans = - alpha1 * math.log(1-beta1*P1(t)) - alpha2 * math.log(1-beta2*P2(t))
    return ans
def CGF1(t):
    ans = - (alpha1+1) * math.log(1-beta1*P1(t)) - alpha2 * math.log(1-beta2*P2(t))
    return ans
def CGF2(t):
    ans = - alpha1 * math.log(1-beta1*P1(t)) - (alpha2+1) * math.log(1-beta2*P2(t))
    return ans
def CGF_fir(t):
    ans = alpha1*beta1*P1_div(t, 1)/(1-beta1*P1(t)) + alpha2*beta2*P2_div(t, 1)/(1-beta2*P2(t))
    return ans
def CGF_sed(t):
    ans = alpha1*beta1*( P1_div(t,2)/(1-beta1 * P1(t)) + beta1 * (P1_div(t, 1)/(1-beta1*P1(t)))**2) + \
          alpha2*beta2*( P2_div(t,2)/(1-beta2 * P2(t)) + beta2 * (P2_div(t, 1)/(1-beta2*P2(t)))**2)
    return ans
def CGF_sed1(t):
    ans = (alpha1+1)*beta1*( P1_div(t,2)/(1-beta1 * P1(t)) + beta1 * (P1_div(t, 1)/(1-beta1*P1(t)))**2) + \
          alpha2*beta2*( P2_div(t,2)/(1-beta2 * P2(t)) + beta2 * (P2_div(t, 1)/(1-beta2*P2(t)))**2)
    return ans
def CGF_sed2(t):
    ans = alpha1*beta1*( P1_div(t,2)/(1-beta1 * P1(t)) + beta1 * (P1_div(t, 1)/(1-beta1*P1(t)))**2) + \
          (alpha2+1)*beta2*( P2_div(t,2)/(1-beta2 * P2(t)) + beta2 * (P2_div(t, 1)/(1-beta2*P2(t)))**2)
    return ans
def CGF_thi(t):
    Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
    Nk12 = P1_div(t, 2) / (1 - beta1 * P1(t))
    Nk13 = P1_div(t, 3) / (1 - beta1 * P1(t))
    Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
    Nk22 = P2_div(t, 2) / (1 - beta2 * P2(t))
    Nk23 = P2_div(t, 3) / (1 - beta2 * P2(t))

    temp1 = beta1 * alpha1 * (
            Nk13 + 3 * beta1 * Nk11 * Nk12 + 2 * (beta1 ** 2) * (Nk11 ** 3))

    temp2 = beta2 * alpha2 * (
            Nk23 + 3 * beta2 * Nk21 * Nk22 + 2 * (beta2 ** 2) * (Nk21 ** 3))
    return temp1 + temp2
def CGF_thi1(t):
    Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
    Nk12 = P1_div(t, 2) / (1 - beta1 * P1(t))
    Nk13 = P1_div(t, 3) / (1 - beta1 * P1(t))
    Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
    Nk22 = P2_div(t, 2) / (1 - beta2 * P2(t))
    Nk23 = P2_div(t, 3) / (1 - beta2 * P2(t))

    temp1 = beta1 * (alpha1+1) * (
            Nk13 + 3 * beta1 * Nk11 * Nk12 + 2 * (beta1 ** 2) * (Nk11 ** 3))

    temp2 = beta2 * alpha2 * (
            Nk23 + 3 * beta2 * Nk21 * Nk22 + 2 * (beta2 ** 2) * (Nk21 ** 3))
    return temp1 + temp2
def CGF_thi2(t):
    Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
    Nk12 = P1_div(t, 2) / (1 - beta1 * P1(t))
    Nk13 = P1_div(t, 3) / (1 - beta1 * P1(t))
    Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
    Nk22 = P2_div(t, 2) / (1 - beta2 * P2(t))
    Nk23 = P2_div(t, 3) / (1 - beta2 * P2(t))

    temp1 = beta1 * alpha1 * (
            Nk13 + 3 * beta1 * Nk11 * Nk12 + 2 * (beta1 ** 2) * (Nk11 ** 3))

    temp2 = beta2 * (alpha2+1) * (
            Nk23 + 3 * beta2 * Nk21 * Nk22 + 2 * (beta2 ** 2) * (Nk21 ** 3))
    return temp1 + temp2
def CGF_for(t):
    Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
    Nk12 = P1_div(t, 2) / (1 - beta1 * P1(t))
    Nk13 = P1_div(t, 3) / (1 - beta1 * P1(t))
    Nk14 = P1_div(t, 4) / (1 - beta1 * P1(t))
    Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
    Nk22 = P2_div(t, 2) / (1 - beta2 * P2(t))
    Nk23 = P2_div(t, 3) / (1 - beta2 * P2(t))
    Nk24 = P2_div(t, 4) / (1 - beta2 * P2(t))

    temp1 = beta1 * alpha1 * (Nk14 + 3 * beta1 * Nk12 ** 2 + 4 * beta1 * Nk13 * Nk11 +
                12 * (beta1 ** 2) * (Nk11 ** 2) * Nk12 + 6 * ( beta1 ** 3) * ( Nk11 ** 4))

    temp2 = beta2 * alpha2 * (Nk24 + 3 * beta2 * Nk22 ** 2 + 4 * beta2 * Nk23 * Nk21 +
                              12 * (beta2 ** 2) * (Nk21 ** 2) * Nk22 + 6 * (beta2 ** 3) * (Nk21 ** 4))

    return temp1 + temp2
def CGF_for1(t):
    Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
    Nk12 = P1_div(t, 2) / (1 - beta1 * P1(t))
    Nk13 = P1_div(t, 3) / (1 - beta1 * P1(t))
    Nk14 = P1_div(t, 4) / (1 - beta1 * P1(t))
    Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
    Nk22 = P2_div(t, 2) / (1 - beta2 * P2(t))
    Nk23 = P2_div(t, 3) / (1 - beta2 * P2(t))
    Nk24 = P2_div(t, 4) / (1 - beta2 * P2(t))

    temp1 = beta1 * (alpha1+1) * (Nk14 + 3 * beta1 * Nk12 ** 2 + 4 * beta1 * Nk13 * Nk11 +
                12 * (beta1 ** 2) * (Nk11 ** 2) * Nk12 + 6 * ( beta1 ** 3) * ( Nk11 ** 4))

    temp2 = beta2 * alpha2 * (Nk24 + 3 * beta2 * Nk22 ** 2 + 4 * beta2 * Nk23 * Nk21 +
                              12 * (beta2 ** 2) * (Nk21 ** 2) * Nk22 + 6 * (beta2 ** 3) * (Nk21 ** 4))

    return temp1 + temp2
def CGF_for2(t):
    Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
    Nk12 = P1_div(t, 2) / (1 - beta1 * P1(t))
    Nk13 = P1_div(t, 3) / (1 - beta1 * P1(t))
    Nk14 = P1_div(t, 4) / (1 - beta1 * P1(t))
    Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
    Nk22 = P2_div(t, 2) / (1 - beta2 * P2(t))
    Nk23 = P2_div(t, 3) / (1 - beta2 * P2(t))
    Nk24 = P2_div(t, 4) / (1 - beta2 * P2(t))

    temp1 = beta1 * alpha1 * (Nk14 + 3 * beta1 * Nk12 ** 2 + 4 * beta1 * Nk13 * Nk11 +
                12 * (beta1 ** 2) * (Nk11 ** 2) * Nk12 + 6 * ( beta1 ** 3) * ( Nk11 ** 4))

    temp2 = beta2 * (alpha2+1) * (Nk24 + 3 * beta2 * Nk22 ** 2 + 4 * beta2 * Nk23 * Nk21 +
                              12 * (beta2 ** 2) * (Nk21 ** 2) * Nk22 + 6 * (beta2 ** 3) * (Nk21 ** 4))

    return temp1 + temp2
def tailprob_to_VaR_first_order(var):
    """
    :return: given the var, return the corresponding spt, then return the CDF
    """
    def CGF_fir_changing(t):
        """
        this function is for given an x0, solve out t.
        """
        ans = alpha1*beta1*P1_div(t, 1) / (1-beta1*P1(t)) + alpha2*beta2*P2_div(t, 1) / (1-beta2*P2(t))
        return ans - var

    spt = fsolve(CGF_fir_changing, est_spt)
    uhat = spt * np.sqrt(CGF_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CGF(spt)))
    ans = norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat)
    return (1-ans) - tailprob
def tailprob_to_VaR_second_order(var):
    """
    :return: given the var, return the corresponding spt, then return the CDF
    """
    def CGF_fir_changing(t):
        """
        this function is for given an x0, solve out t.
        """
        ans = alpha1*beta1*P1_div(t, 1) / (1-beta1*P1(t)) + alpha2*beta2*P2_div(t, 1) / (1-beta2*P2(t))
        return ans - var
    spt = fsolve(CGF_fir_changing, est_spt)
    uhat = spt * np.sqrt(CGF_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CGF(spt)))

    lambda3 = CGF_thi(spt) / (CGF_sed(spt) ** 1.5)
    lambda4 = CGF_for(spt) / (CGF_sed(spt) ** 2)
    temp = (1 / (what ** 3)) - (1 / (uhat ** 3)) - (lambda3 / (2 * (uhat ** 2))) + (1 / uhat) * (
            (lambda4 / 8) - (5 * (lambda3 ** 2) / 24))
    tp = 1 - (norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat - temp))
    return tp - tailprob
def CGF_fir_changing(t):
    """
    this function is for given an x0, solve out t.
    """
    ans = alpha1 * beta1 * P1_div(t, 1) / (1 - beta1 * P1(t)) + alpha2 * beta2 * P2_div(t, 1) / (1 - beta2 * P2(t))
    return ans - var
def ES2(var, spt):
    uhat = spt * np.sqrt(CGF_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CGF(spt)))
    ans = EL * (1 - norm.cdf(what)) + norm.pdf(what) * ((var / uhat) - (EL / what))
    ans = ans / tailprob
    return ans
def ES3(var, spt):
    uhat = spt * np.sqrt(CGF_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CGF(spt)))
    ans = EL * (1 - norm.cdf(what)) + norm.pdf(what) * (
            (var / uhat) - (EL / what) + (EL - var) / what ** 3 + 1 / (spt * uhat))
    ans = ans / tailprob
    return ans
def check_function(var):
    """
    using the optimilization to find the minimum as the ES, and the root as VaR
    :param var:
    :return:
    """
    def CGF_fir_changing(t):
        """
        this function is for given an x0, solve out t.
        """
        ans = alpha1*beta1*P1_div(t, 1) / (1-beta1*P1(t)) + alpha2*beta2*P2_div(t, 1) / (1-beta2*P2(t))
        return ans - var

    spt = fsolve(CGF_fir_changing, est_spt)
    DEL = EL - var
    uhat = spt * np.sqrt(CGF_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CGF(spt)))
    ans = DEL * (1 - norm.cdf(what)) - norm.pdf(what) * ((DEL / what) - (DEL / (what ** 3)) - (1 / (spt * uhat)))
    ans = var + 1 / tailprob * ans
    return ans
def KRC( ik, ipd, ipl, t):
    """
    this function is to generating the multivarite CGF of portfolio L and obligor i
    :param ik:  the sector where obligor i belongs to, there are two sectors
    :param ipd:  the probability of default of obligor i
    :param ipl: the potential loss of obligor i
    """
    if ik == 1:
        ans = ipd * np.exp(t * ipl) * (alpha1 * beta1 / (1-beta1*P1(t)))
    if ik == 2:
        ans = ipd * np.exp(t * ipl) * (alpha2 * beta2 / (1 - beta2 * P2(t)))
    return ans
def KRC_fir( ik, ipd, ipl, t):
    if ik ==1 :
        term1a = alpha1 * beta1 * ipl / (1- beta1*P1(t))
        term1b = alpha1 * (beta1**2) * P1_div(t, 1) / ((1- beta1*P1(t))**2)
        ans = ipd * np.exp(t * ipl) * (term1a - term1b)
    if ik ==2 :
        term2a = alpha2 * beta2 * ipl / (1 - beta2 * P2(t))
        term2b = alpha2 * (beta2 ** 2) * P2_div(t, 1) / ((1 - beta2 * P2(t)) ** 2)
        ans = ipd * np.exp(t * ipl) * (term2a - term2b)
    return ans
def KRC_sed(ik, ipd, ipl, t):
    if ik == 1:
        term1a = (ipl**2) / (1- beta1*P1(t))
        term1b = beta1 * (2 * ipl * P1_div(t, 1) + P1_div(t, 2)) / ((1- beta1*P1(t))**2)
        term1c = 2 * (beta1**2) * (P1_div(t, 1)**2) / ((1- beta1*P1(t))**3)
        ans = ipd * np.exp(t * ipl) * alpha1 * beta1 * (term1a - term1b - term1c)
    if ik == 2:
        term2a = (ipl**2) / (1- beta2*P2(t))
        term2b = beta2 * (2 * ipl * P2_div(t, 1) + P2_div(t, 2)) / ((1- beta2*P2(t))**2)
        term2c = 2 * (beta2**2) * (P2_div(t, 1)**2) / ((1- beta2*P2(t))**3)
        ans = ipd * np.exp(t * ipl) * alpha2 * beta2 * (term2a - term2b - term2c)
    return ans
def RHO(r, t):
    """
    :param r: the order of derivative
    :return: standardized cumulant of order r
    """
    if r == 3:
        ans = CGF_thi(t) / CGF_sed(t) ** (3 / 2)
    if r == 4:
        ans = CGF_for(t) / CGF_sed(t) ** 2
    return ans
def VARC_KIM(spt):

    ans_set = []

    for i in np.arange(0, len(exposure1)):
        sing_exposure1 = exposure1[i]
        sing_pd1 = pd1[i]
        sing_obligor1 = obligor1[i]

        krc_fir_value = KRC_fir( 1, sing_pd1, sing_exposure1, spt)
        krc_sed_value = KRC_sed(1, sing_pd1, sing_exposure1, spt)
        krc_value = KRC(1, sing_pd1, sing_exposure1, spt)

        num = RHO(3, spt) * krc_fir_value / (2 * np.sqrt(CGF_sed(spt))) - krc_sed_value / (2*CGF_sed(spt))
        den = 1 + (RHO(4, spt)/8 - 5 * RHO(3, spt)/24)

        ans = krc_value + num/den
        ans_set.append((ans*sing_exposure1*sing_obligor1)[0])

    for i in np.arange(0, len(exposure2)):
        sing_exposure2 = exposure2[i]
        sing_pd2 = pd2[i]
        sing_obligor2 = obligor2[i]

        krc_fir_value = KRC_fir(2, sing_pd2, sing_exposure2, spt)
        krc_sed_value = KRC_sed(2, sing_pd2, sing_exposure2, spt)
        krc_value = KRC(2, sing_pd2, sing_exposure2, spt)

        num = RHO(3, spt) * krc_fir_value / (2 * np.sqrt(CGF_sed(spt))) - krc_sed_value / (2*CGF_sed(spt))
        den = 1 + (RHO(4, spt)/8 - 5 * RHO(3, spt)/24)
        ans = krc_value + num/den

        ans_set.append((ans*sing_exposure2*sing_obligor2)[0])

    return ans_set
def ESC_KIM(spt, var):
    """
    kim's formula to calculate ESC
    :return:
    """
    uhat = spt * np.sqrt(CGF_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CGF(spt)))
    lambda3 = CGF_thi(spt) / (CGF_sed(spt) ** 1.5)
    lambda4 = CGF_for(spt) / (CGF_sed(spt) ** 2)
    temp = (1 / (what ** 3)) - (1 / (uhat ** 3)) - (lambda3 / (2 * (uhat ** 2))) + (1 / uhat) * (
            (lambda4 / 8) - (5 * (lambda3 ** 2) / 24))
    tail_prob_value = 1 - (norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat - temp))
    term1 = 1 / (tail_prob_value * np.sqrt(2 * np.pi)) * np.exp(CGF(spt) - var * spt)

    ans_set = []

    for i in np.arange(0, len(exposure1)):
        sing_exposure1 = exposure1[i]
        sing_pd1 = pd1[i]
        sing_obligor1 = obligor1[i]

        krc_fir_value = KRC_fir(1, sing_pd1, sing_exposure1, spt)
        krc_sed_value = KRC_sed(1, sing_pd1, sing_exposure1, spt)
        krc_value = KRC(1, sing_pd1, sing_exposure1, spt)
        krc_value0 = KRC(1, sing_pd1, sing_exposure1, 0)

        dmean = sing_pd1 * alpha1 * beta1
        a = (krc_value - krc_value0) / uhat
        b = a * (RHO(4,spt)/8) - (5*(RHO(3,spt)**2)/24) - (RHO(3,spt)/(2*uhat)) - (1/(uhat**2))
        c = (RHO(3,spt)/2 + 1/uhat) * krc_fir_value / (spt * CGF_sed(spt))
        d = krc_sed_value / (2 * uhat * CGF_sed(spt))

        ans = dmean + term1 * (a + b + c - d)
        ans_set.append((ans*sing_exposure1*sing_obligor1)[0])

    for i in np.arange(0, len(exposure2)):
        sing_exposure2 = exposure2[i]
        sing_pd2 = pd2[i]
        sing_obligor2 = obligor2[i]

        krc_fir_value = KRC_fir(2, sing_pd2, sing_exposure2, spt)
        krc_sed_value = KRC_sed(2, sing_pd2, sing_exposure2, spt)
        krc_value = KRC(2, sing_pd2, sing_exposure2, spt)
        krc_value0 = KRC(2, sing_pd2, sing_exposure2, 0)

        dmean = sing_pd2 * alpha2 * beta2
        a = (krc_value - krc_value0) / uhat
        b = a * (RHO(4,spt)/8) - (5*(RHO(3,spt)**2)/24) - (RHO(3,spt)/(2*uhat)) - (1/(uhat**2))
        c = (RHO(3,spt)/2 + 1/uhat) * krc_fir_value / (spt * CGF_sed(spt))
        d = krc_sed_value / (2 * uhat * CGF_sed(spt))

        ans = dmean + term1 * (a + b + c - d)
        ans_set.append((ans*sing_exposure2*sing_obligor2)[0])

    return ans_set
def ESC_diff(spt, var):
    ans_set = []

    for i in np.arange(0, len(exposure1)):
        sing_exposure1 = exposure1[i]
        sing_pd1 = pd1[i]
        sing_obligor1 = obligor1[i]

        cgf_drv = spt * alpha1 * beta1 * sing_pd1 * np.exp(spt * sing_exposure1) / ( 1- beta1* P1(t))
        term1 = (spt**3) * alpha1 * beta1 * sing_pd1 * np.exp(spt * sing_exposure1)
        term2 = ( 1/(1- beta1* P1(t)) - 3 * beta1 * sing_pd1 * np.exp(spt * sing_exposure1) / (1- beta1* P1(t))**2
                        -  2*(beta1**2)*(sing_pd1**2)*np.exp(2*spt*sing_exposure1)/(1- beta1* P1(t))**3)
        cgf_drv_sed = term1 * term2

        uhat = spt * np.sqrt(CGF(spt))
        what = np.sign(spt) * np.sqrt(2 * (spt * var - CGF(spt)))

        ans = - norm.pdf(what) * (cgf_drv * (- EL / what ** 3 + var / uhat)
                              - var / 2 * (cgf_drv_sed / (uhat * CGF_sed(spt))))

        ans_set.append((ans / tailprob * sing_exposure1 * sing_obligor1)[0])

    for i in np.arange(0, len(exposure2)):
        sing_exposure2 = exposure2[i]
        sing_pd2 = pd2[i]
        sing_obligor2 = obligor2[i]

        cgf_drv = spt * alpha2 * beta2 * sing_pd2 * np.exp(spt * sing_exposure2) / (1 - beta2 * P2(t))
        term1 = (spt ** 3) * alpha2 * beta2 * sing_pd2 * np.exp(spt * sing_exposure2)
        term2 = (1 / (1 - beta2 * P2(t)) - 3 * beta2 * sing_pd2 * np.exp(spt * sing_exposure2) / (
                    1 - beta2 * P2(t)) ** 2
                 - 2 * (beta2 ** 2) * (sing_pd2 ** 2) * np.exp(2 * spt * sing_exposure2) / (1 - beta2 * P2(t)) ** 3)
        cgf_drv_sed = term1 * term2

        uhat = spt * np.sqrt(CGF(spt))
        what = np.sign(spt) * np.sqrt(2 * (spt * var - CGF(spt)))

        ans = - norm.pdf(what) * (cgf_drv * (- EL / what ** 3 + var / uhat)
                                  - var / 2 * (cgf_drv_sed / (uhat * CGF_sed(spt))))

        ans_set.append((ans / tailprob * sing_exposure2 * sing_obligor2)[0])

    return ans_set
def VaRC_Tasche_first_order(var, est_spt):
    def CGF_fir_changing(t):
        """
        this function is for given an x0, solve out t.
        """
        ans = alpha1 * beta1 * P1_div(t, 1) / (1 - beta1 * P1(t)) + alpha2 * beta2 * P2_div(t, 1) / (1 - beta2 * P2(t))
        return ans - var

    spt = fsolve(CGF_fir_changing, est_spt)
    pdf = np.exp(CGF(spt) - spt * var) / np.sqrt(2 * math.pi * CGF_sed(spt))

    set_contribution = []
    # sector 1
    for i in np.arange(0, len(exposure1)):
        sing_exposure1 = exposure1[i]
        sing_pd1 = pd1[i]
        sing_obligor1 = obligor1[i]

        def CGF_fir_changing1(t):
            """
            this function is for given an x0, solve out t.
            """
            ans = (alpha1+1) * beta1 * P1_div(t, 1) / (1 - beta1 * P1(t)) + alpha2 * beta2 * P2_div(t, 1) / (
                        1 - beta2 * P2(t))
            return ans - (var - sing_exposure1)

        spt1 = fsolve(CGF_fir_changing1, est_spt)
        pdf1 = np.exp(CGF1(spt1) - spt1 * (var-sing_exposure1)) / np.sqrt(2 * math.pi * CGF_sed1(spt1))
        sing_contribution1 = pdf1/pdf * sing_exposure1 * sing_pd1
        set_contribution.append((sing_contribution1 * sing_obligor1)[0])

    for j in np.arange(0, len(exposure2)):
        sing_exposure2 = exposure2[j]
        sing_pd2 = pd2[j]
        sing_obligor2 = obligor2[j]

        def CGF_fir_changing2(t):
            """
            this function is for given an x0, solve out t.
            """
            ans = alpha1 * beta1 * P1_div(t, 1) / (1 - beta1 * P1(t)) + (alpha2+1) * beta2 * P2_div(t, 1) / (
                        1 - beta2 * P2(t))
            return ans - (var - sing_exposure2)
        spt2 = fsolve(CGF_fir_changing2, est_spt)
        pdf2 = np.exp(CGF2(spt2) - spt2 * (var - sing_exposure2)) / np.sqrt(2 * math.pi * CGF_sed2(spt2))
        sing_contribution2 = pdf2 / pdf * sing_exposure2 * sing_pd2
        set_contribution.append((sing_contribution2 * sing_obligor2)[0])

    return set_contribution
def VaRC_Tasche_second_order(var, est_spt):
    def CGF_fir_changing(t):
        """
        this function is for given an x0, solve out t.
        """
        ans = alpha1 * beta1 * P1_div(t, 1) / (1 - beta1 * P1(t)) + alpha2 * beta2 * P2_div(t, 1) / (1 - beta2 * P2(t))
        return ans - var

    spt = fsolve(CGF_fir_changing, est_spt)
    rho3 = CGF_thi(spt) / (CGF_sed(spt) ** 1.5)
    rho4 = CGF_for(spt) / (CGF_sed(spt) ** 2)
    pdf = np.exp(CGF(spt) - spt * var) / np.sqrt(2 * math.pi * CGF_sed(spt)) * (
                1 + 1 / 8 * (rho4 - ( 5 / 3 ) * (rho3 ** 2)))

    set_contribution = []
    # sector 1
    for i in np.arange(0, len(exposure1)):
        sing_exposure1 = exposure1[i]
        sing_pd1 = pd1[i]
        sing_obligor1 = obligor1[i]

        def CGF_fir_changing1(t):
            """
            this function is for given an x0, solve out t.
            """
            ans = (alpha1+1) * beta1 * P1_div(t, 1) / (1 - beta1 * P1(t)) + alpha2 * beta2 * P2_div(t, 1) / (
                        1 - beta2 * P2(t))
            return ans - (var - sing_exposure1)

        spt1 = fsolve(CGF_fir_changing1, est_spt)
        rho13 = CGF_thi1(spt1) / (CGF_sed1(spt1) ** 1.5)
        rho14 = CGF_for1(spt1) / (CGF_sed1(spt1) ** 2)
        pdf1 = np.exp(CGF1(spt1) - spt1 * (var-sing_exposure1)) / np.sqrt(2 * math.pi * CGF_sed1(spt1)) * (
                1 + 1 / 8 * (rho14 - (5 / 3) * (rho13 ** 2)))
        sing_contribution1 = pdf1/pdf * sing_exposure1 * sing_pd1
        set_contribution.append((sing_contribution1 * sing_obligor1)[0])

    for j in np.arange(0, len(exposure2)):
        sing_exposure2 = exposure2[j]
        sing_pd2 = pd2[j]
        sing_obligor2 = obligor2[j]

        def CGF_fir_changing2(t):
            """
            this function is for given an x0, solve out t.
            """
            ans = alpha1 * beta1 * P1_div(t, 1) / (1 - beta1 * P1(t)) + (alpha2+1) * beta2 * P2_div(t, 1) / (
                        1 - beta2 * P2(t))
            return ans - (var - sing_exposure2)

        spt2 = fsolve(CGF_fir_changing2, est_spt)
        rho23 = CGF_thi2(spt2) / (CGF_sed2(spt2) ** 1.5)
        rho24 = CGF_for2(spt2) / (CGF_sed2(spt2) ** 2)
        pdf2 = np.exp(CGF2(spt2) - spt2 * (var - sing_exposure2)) / np.sqrt(2 * math.pi * CGF_sed2(spt2)) * (
                1 + 1 / 8 * (rho24 - (5 / 3) * (rho23 ** 2)))
        sing_contribution2 = pdf2 / pdf * sing_exposure2 * sing_pd2
        set_contribution.append((sing_contribution2 * sing_obligor2)[0])

    return set_contribution
def ESC_Tasche_second_order(var):

    def CGF_fir_changing(t):
        """
        this function is for given an x0, solve out t.
        """
        ans = alpha1 * beta1 * P1_div(t, 1) / (1 - beta1 * P1(t)) + alpha2 * beta2 * P2_div(t, 1) / (1 - beta2 * P2(t))
        return ans - var

    spt = fsolve(CGF_fir_changing, est_spt)
    uhat = spt * np.sqrt(CGF_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CGF(spt)))
    lambda3 = CGF_thi(spt) / (CGF_sed(spt) ** 1.5)
    lambda4 = CGF_for(spt) / (CGF_sed(spt) ** 2)
    temp = (1 / (what ** 3)) - (1 / (uhat ** 3)) - (lambda3 / (2 * (uhat ** 2))) + (1 / uhat) * (
            (lambda4 / 8) - (5 * (lambda3 ** 2) / 24))
    tp = 1 - (norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat - temp))

    set_contribution = []
    for i in np.arange(0, len(exposure1)):
        sing_exposure1 = exposure1[i]
        sing_pd1 = pd1[i]
        sing_obligor1 = obligor1[i]

        def CGF_fir_changing1(t):
            """
            this function is for given an x0, solve out t.
            """
            ans = (alpha1+1) * beta1 * P1_div(t, 1) / (1 - beta1 * P1(t)) + alpha2 * beta2 * P2_div(t, 1) / (
                        1 - beta2 * P2(t))
            return ans - (var - sing_exposure1)

        spt1 = fsolve(CGF_fir_changing1, est_spt)
        uhat1 = spt1 * np.sqrt(CGF_sed1(spt1))
        what1 = np.sign(spt1) * np.sqrt(2 * (spt1 * (var - sing_exposure1) - CGF1(spt1)))
        lambda13 = CGF_thi1(spt1) / (CGF_sed1(spt1) ** 1.5)
        lambda14 = CGF_for1(spt1) / (CGF_sed1(spt1) ** 2)
        temp1 = (1 / (what1 ** 3)) - (1 / (uhat1 ** 3)) - (lambda13 / (2 * (uhat1 ** 2))) + (1 / uhat1) * (
                (lambda14 / 8) - (5 * (lambda13 ** 2) / 24))
        tp1 = 1 - ( norm.cdf(what1) + norm.pdf(what1) * (1 / what1 - 1 / uhat1 - temp1) )
        sing_contribution1 = tp1/tp * sing_exposure1 * sing_pd1
        set_contribution.append((sing_contribution1 * sing_obligor1)[0])


    for j in np.arange(0, len(exposure2)):
        sing_exposure2 = exposure2[j]
        sing_pd2 = pd2[j]
        sing_obligor2 = obligor2[j]

        def CGF_fir_changing2(t):
            """
            this function is for given an x0, solve out t.
            """
            ans = alpha1 * beta1 * P1_div(t, 1) / (1 - beta1 * P1(t)) + (alpha2+1) * beta2 * P2_div(t, 1) / (
                        1 - beta2 * P2(t))
            return ans - (var - sing_exposure2)

        spt2 = fsolve(CGF_fir_changing2, est_spt)
        uhat2 = spt2 * np.sqrt(CGF_sed2(spt2))
        what2 = np.sign(spt2) * np.sqrt(2 * (spt2 * (var - sing_exposure2) - CGF2(spt2)))
        lambda23 = CGF_thi2(spt2) / (CGF_sed2(spt2) ** 1.5)
        lambda24 = CGF_for2(spt2) / (CGF_sed2(spt2) ** 2)
        temp2 = (1 / (what2 ** 3)) - (1 / (uhat2 ** 3)) - (lambda23 / (2 * (uhat2 ** 2))) + (1 / uhat2) * (
                (lambda24 / 8) - (5 * (lambda23 ** 2) / 24))
        tp2 = 1 - (norm.cdf(what2) + norm.pdf(what2) * (1 / what2 - 1 / uhat2 - temp2))

        sing_contribution2 = tp2/tp * sing_exposure2 * sing_pd2
        set_contribution.append((sing_contribution2 * sing_obligor2)[0])
    return set_contribution
def ESC_Tasche_first_order(var):

    def CGF_fir_changing(t):
        """
        this function is for given an x0, solve out t.
        """
        ans = alpha1 * beta1 * P1_div(t, 1) / (1 - beta1 * P1(t)) + alpha2 * beta2 * P2_div(t, 1) / (1 - beta2 * P2(t))
        return ans - var

    spt = fsolve(CGF_fir_changing, est_spt)
    uhat = spt * np.sqrt(CGF_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CGF(spt)))
    tp = 1 - (norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat))

    set_contribution = []
    for i in np.arange(0, len(exposure1)):
        sing_exposure1 = exposure1[i]
        sing_pd1 = pd1[i]
        sing_obligor1 = obligor1[i]

        def CGF_fir_changing1(t):
            """
            this function is for given an x0, solve out t.
            """
            ans = (alpha1+1) * beta1 * P1_div(t, 1) / (1 - beta1 * P1(t)) + alpha2 * beta2 * P2_div(t, 1) / (
                        1 - beta2 * P2(t))
            return ans - (var - sing_exposure1)

        spt1 = fsolve(CGF_fir_changing1, est_spt)
        uhat1 = spt1 * np.sqrt(CGF_sed1(spt1))
        what1 = np.sign(spt1) * np.sqrt(2 * (spt1 * (var - sing_exposure1) - CGF1(spt1)))
        tp1 = 1 - (norm.cdf(what1) + norm.pdf(what1) * (1 / what1 - 1 / uhat1))
        sing_contribution1 = tp1/tp * sing_exposure1 * sing_pd1
        set_contribution.append((sing_contribution1 * sing_obligor1)[0])


    for j in np.arange(0, len(exposure2)):
        sing_exposure2 = exposure2[j]
        sing_pd2 = pd2[j]
        sing_obligor2 = obligor2[j]

        def CGF_fir_changing2(t):
            """
            this function is for given an x0, solve out t.
            """
            ans = alpha1 * beta1 * P1_div(t, 1) / (1 - beta1 * P1(t)) + (alpha2+1) * beta2 * P2_div(t, 1) / (
                        1 - beta2 * P2(t))
            return ans - (var - sing_exposure2)

        spt2 = fsolve(CGF_fir_changing2, est_spt)
        uhat2 = spt2 * np.sqrt(CGF_sed2(spt2))
        what2 = np.sign(spt2) * np.sqrt(2 * (spt2 * (var - sing_exposure2) - CGF2(spt2)))
        tp2 = 1 - (norm.cdf(what2) + norm.pdf(what2) * (1 / what2 - 1 / uhat2))
        sing_contribution2 = tp2/tp * sing_exposure2 * sing_pd2
        set_contribution.append((sing_contribution2 * sing_obligor2)[0])
    return set_contribution
def VARC_MARTIN(spt):
    ans_set = []

    for i in np.arange(0, len(exposure1)):
        sing_exposure1 = exposure1[i]
        sing_pd1 = pd1[i]
        sing_obligor1 = obligor1[i]
        ans = sing_exposure1 * sing_pd1 * np.exp(spt*sing_exposure1)  * alpha1 * beta1 / (1 - beta1*P1(spt))
        ans_set.append((ans * sing_obligor1)[0])

    for i in np.arange(0, len(exposure2)):
        sing_exposure2 = exposure2[i]
        sing_pd2 = pd2[i]
        sing_obligor2 = obligor2[i]
        ans = sing_exposure2 * sing_pd2 * np.exp(spt * sing_exposure2) * alpha2 * beta2 / (1 - beta2 * P2(spt))
        ans_set.append((ans * sing_obligor2)[0])

    return ans_set


def PC1(t):
    ans = gamma11 * P1(t) + gamma12 * P2(t)
    return ans
def PC1_div(t, n):
    ans = gamma11 * P1_div(t, n) + gamma12 * P2_div(t, n)
    return ans
def CGFC1(t):
    ans = - alpha1 * math.log(1-beta1*P1(t)) - alpha2 * math.log(1-beta2*P2(t)) - theta1 * math.log(1 - PC1(t))
    return ans
def CGFC1_1(t):
    ans = - (alpha1+1) * math.log(1-beta1*P1(t)) - alpha2 * math.log(1-beta2*P2(t)) - theta1 * math.log(1 - PC1(t))
    return ans
def CGFC1_2(t):
    ans = - alpha1 * math.log(1-beta1*P1(t)) - (alpha2+1) * math.log(1-beta2*P2(t)) - theta1 * math.log(1 - PC1(t))
    return ans
def CGFC1_3(t):
    ans = - alpha1 * math.log(1-beta1*P1(t)) - alpha2 * math.log(1-beta2*P2(t)) - (theta1+1) * math.log(1 - PC1(t))
    return ans
def CGFC1_fir(t):
    Nk11 = P1_div(t, 1) / (1-beta1*P1(t))
    Nk21 = P2_div(t, 1) / (1-beta2*P2(t))
    Nl11 = PC1_div(t, 1) / (1 - PC1(t))
    ans = alpha1*beta1* Nk11 + alpha2*beta2* Nk21 + theta1 * Nl11
    return ans
def CGFC1_sed(t):
    Nk11 = P1_div(t, 1) / (1-beta1*P1(t))
    Nk12 = P1_div(t, 2) / (1-beta1*P1(t))

    Nk21 = P2_div(t, 1) / (1-beta2*P2(t))
    Nk22 = P2_div(t, 2) / (1-beta2*P2(t))

    Nl11 = PC1_div(t, 1)/(1 - PC1(t))
    Nl12 = PC1_div(t, 2)/(1 - PC1(t))

    ans = alpha1*beta1*( Nk12 + beta1*(Nk11**2) ) + \
          alpha2*beta2*( Nk22 + beta2 *(Nk21**2) ) + \
          theta1 * (Nl12 + (Nl11)**2)
    return ans
def CGFC1_sed_1(t):
    Nk11 = P1_div(t, 1) / (1-beta1*P1(t))
    Nk12 = P1_div(t, 2) / (1-beta1*P1(t))

    Nk21 = P2_div(t, 1) / (1-beta2*P2(t))
    Nk22 = P2_div(t, 2) / (1-beta2*P2(t))

    Nl11 = PC1_div(t, 1)/(1 - PC1(t))
    Nl12 = PC1_div(t, 2)/(1 - PC1(t))

    ans = (alpha1 +1) *beta1*( Nk12 + beta1*(Nk11**2) ) + \
          alpha2*beta2*( Nk22 + beta2 *(Nk21**2) ) + \
          theta1 * (Nl12 + (Nl11)**2)
    return ans
def CGFC1_sed_2(t):
    Nk11 = P1_div(t, 1) / (1-beta1*P1(t))
    Nk12 = P1_div(t, 2) / (1-beta1*P1(t))

    Nk21 = P2_div(t, 1) / (1-beta2*P2(t))
    Nk22 = P2_div(t, 2) / (1-beta2*P2(t))

    Nl11 = PC1_div(t, 1)/(1 - PC1(t))
    Nl12 = PC1_div(t, 2)/(1 - PC1(t))

    ans = alpha1 *beta1*( Nk12 + beta1*(Nk11**2) ) + \
          (alpha2 + 1 )*beta2*( Nk22 + beta2 *(Nk21**2) ) + \
          theta1 * (Nl12 + (Nl11)**2)
    return ans
def CGFC1_sed_3(t):
    Nk11 = P1_div(t, 1) / (1-beta1*P1(t))
    Nk12 = P1_div(t, 2) / (1-beta1*P1(t))

    Nk21 = P2_div(t, 1) / (1-beta2*P2(t))
    Nk22 = P2_div(t, 2) / (1-beta2*P2(t))

    Nl11 = PC1_div(t, 1)/(1 - PC1(t))
    Nl12 = PC1_div(t, 2)/(1 - PC1(t))

    ans = alpha1 *beta1*( Nk12 + beta1*(Nk11**2) ) + \
          alpha2 *beta2*( Nk22 + beta2 *(Nk21**2) ) + \
          (theta1+1) * (Nl12 + (Nl11)**2)
    return ans
def CGFC1_thi(t):
    Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
    Nk12 = P1_div(t, 2) / (1 - beta1 * P1(t))
    Nk13 = P1_div(t, 3) / (1 - beta1 * P1(t))

    Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
    Nk22 = P2_div(t, 2) / (1 - beta2 * P2(t))
    Nk23 = P2_div(t, 3) / (1 - beta2 * P2(t))

    Nl11 = PC1_div(t, 1) / (1 - PC1(t))
    Nl12 = PC1_div(t, 2) / (1 - PC1(t))
    Nl13 = PC1_div(t, 3) / (1 - PC1(t))


    temp1 = beta1 * alpha1 * (
            Nk13 + 3 * beta1 * Nk11 * Nk12 + 2 * (beta1 ** 2) * (Nk11 ** 3))

    temp2 = beta2 * alpha2 * (
            Nk23 + 3 * beta2 * Nk21 * Nk22 + 2 * (beta2 ** 2) * (Nk21 ** 3))

    temp3 = theta1 * ( Nl13 + 3 * Nl11 * Nl12 + 2 * (Nl11 ** 3))
    return temp1 + temp2 + temp3
def CGFC1_for(t):
    Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
    Nk12 = P1_div(t, 2) / (1 - beta1 * P1(t))
    Nk13 = P1_div(t, 3) / (1 - beta1 * P1(t))
    Nk14 = P1_div(t, 4) / (1 - beta1 * P1(t))

    Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
    Nk22 = P2_div(t, 2) / (1 - beta2 * P2(t))
    Nk23 = P2_div(t, 3) / (1 - beta2 * P2(t))
    Nk24 = P2_div(t, 4) / (1 - beta2 * P2(t))

    Nl11 = PC1_div(t, 1) / (1 - PC1(t))
    Nl12 = PC1_div(t, 2) / (1 - PC1(t))
    Nl13 = PC1_div(t, 3) / (1 - PC1(t))
    Nl14 = PC1_div(t, 4) / (1 - PC1(t))

    temp1 = beta1 * alpha1 * (Nk14 + 3 * beta1 * Nk12 ** 2 + 4 * beta1 * Nk13 * Nk11 +
                12 * (beta1 ** 2) * (Nk11 ** 2) * Nk12 + 6 * ( beta1 ** 3) * ( Nk11 ** 4))

    temp2 = beta2 * alpha2 * (Nk24 + 3 * beta2 * Nk22 ** 2 + 4 * beta2 * Nk23 * Nk21 +
                              12 * (beta2 ** 2) * (Nk21 ** 2) * Nk22 + 6 * (beta2 ** 3) * (Nk21 ** 4))

    temp3 = theta1 * (Nl14 + 3*(Nl12**2) + 4*Nl13*Nl11 + 12*(Nl11**2)*Nl12 + 6*(Nl11**4))

    return temp1 + temp2 + temp3
def C1_tailprob_to_VaR_first_order(var):
    """
    :return: given the var, return the corresponding spt, then return the CDF
    """
    def CGFC1_fir_changing(t):
        """
        this function is for given an x0, solve out t.
        """
        Nk11 = P1_div(t, 1) / (1-beta1*P1(t))
        Nk21 = P2_div(t, 1) / (1-beta2*P2(t))
        Nl11 = PC1_div(t, 1) / (1 - PC1(t))
        ans = alpha1*beta1* Nk11 + alpha2*beta2* Nk21 + theta1 * Nl11
        return ans - var

    spt = fsolve(CGFC1_fir_changing, est_spt)
    uhat = spt * np.sqrt(CGFC1_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CGFC1(spt)))
    ans = norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat)
    return (1-ans) - tailprob
def C1_tailprob_to_VaR_second_order(var):
    """
    :return: given the var, return the corresponding spt, then return the CDF
    """

    def CGFC1_fir_changing(t):
        """
        this function is for given an x0, solve out t.
        """
        Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
        Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
        Nl11 = PC1_div(t, 1) / (1 - PC1(t))
        ans = alpha1 * beta1 * Nk11 + alpha2 * beta2 * Nk21 + theta1 * Nl11
        return ans - var

    spt = fsolve(CGFC1_fir_changing, est_spt)
    uhat = spt * np.sqrt(CGFC1_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CGFC1(spt)))

    lambda3 = CGFC1_thi(spt) / (CGFC1_sed(spt) ** 1.5)
    lambda4 = CGFC1_for(spt) / (CGFC1_sed(spt) ** 2)
    temp = (1 / (what ** 3)) - (1 / (uhat ** 3)) - (lambda3 / (2 * (uhat ** 2))) + (1 / uhat) * (
            (lambda4 / 8) - (5 * (lambda3 ** 2) / 24))
    tp = 1 - (norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat - temp))
    return tp - tailprob
def CGFC1_fir_changing(t):
    """
    this function is for given an x0, solve out t.
    """
    Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
    Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
    Nl11 = PC1_div(t, 1) / (1 - PC1(t))
    ans = alpha1 * beta1 * Nk11 + alpha2 * beta2 * Nk21 + theta1 * Nl11
    return ans - var
def C1_ES2(var, spt):
    uhat = spt * np.sqrt(CGFC1_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CGFC1(spt)))
    ans = EL * (1 - norm.cdf(what)) + norm.pdf(what) * ((var / uhat) - (EL / what))
    ans = ans / tailprob
    return ans
def C1_ES3(var, spt):
    uhat = spt * np.sqrt(CGFC1_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CGFC1(spt)))
    ans = EL * (1 - norm.cdf(what)) + norm.pdf(what) * (
            (var / uhat) - (EL / what) + (EL - var) / what ** 3 + 1 / (spt * uhat))
    ans = ans / tailprob
    return ans
def C1_check_function(var):
    """
    using the optimilization to find the minimum as the ES, and the root as VaR
    :param var:
    :return:
    """

    def CGFC1_fir_changing(t):
        """
        this function is for given an x0, solve out t.
        """
        Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
        Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
        Nl11 = PC1_div(t, 1) / (1 - PC1(t))
        ans = alpha1 * beta1 * Nk11 + alpha2 * beta2 * Nk21 + theta1 * Nl11
        return ans - var

    spt = fsolve(CGFC1_fir_changing, est_spt)
    DEL = EL - var
    uhat = spt * np.sqrt(CGFC1_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CGFC1(spt)))
    ans = DEL * (1 - norm.cdf(what)) - norm.pdf(what) * ((DEL / what) - (DEL / (what ** 3)) - (1 / (spt * uhat)))
    ans = var + 1 / tailprob * ans
    return ans
def C1_VaRC_Tasche_first_order(var, est_spt):

    def CGFC1_fir_changing(t):
        """
        this function is for given an x0, solve out t.
        """
        Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
        Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
        Nl11 = PC1_div(t, 1) / (1 - PC1(t))
        ans = alpha1 * beta1 * Nk11 + alpha2 * beta2 * Nk21 + theta1 * Nl11
        return ans - var
    spt = fsolve(CGFC1_fir_changing, est_spt)
    pdf = np.exp(CGFC1(spt) - spt * var) / np.sqrt(2 * math.pi * CGFC1_sed(spt))

    set_contribution = []
    # sector 1
    for i in np.arange(0, len(exposure1)):
        sing_exposure1 = exposure1[i]
        sing_pd1 = pd1[i]
        sing_obligor1 = obligor1[i]

        def CGFC1_fir_changing1(t):
            """
            this function is for given an x0, solve out t.
            """
            Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
            Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
            Nl11 = PC1_div(t, 1) / (1 - PC1(t))
            ans = (alpha1 +1)* beta1 * Nk11 + alpha2 * beta2 * Nk21 + theta1 * Nl11
            return ans - (var - sing_exposure1)

        spt1a = fsolve(CGFC1_fir_changing1, est_spt)
        pdf1a = np.exp(CGFC1_1(spt1a) - spt1a * (var-sing_exposure1)) / np.sqrt(2 * math.pi * CGFC1_sed_1(spt1a))

        def CGFC1_fir_changing2(t):
            """
            this function is for given an x0, solve out t.
            """
            Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
            Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
            Nl11 = PC1_div(t, 1) / (1 - PC1(t))
            ans = alpha1 * beta1 * Nk11 + alpha2 * beta2 * Nk21 + (theta1+1) * Nl11
            return ans - (var - sing_exposure1)

        spt1b = fsolve(CGFC1_fir_changing2, est_spt)
        pdf1b = np.exp(CGFC1_3(spt1b) - spt1b * (var - sing_exposure1)) / np.sqrt(2 * math.pi * CGFC1_sed_3(spt1b))

        sing_contribution1 = (pdf1a + gamma11 * pdf1b)/pdf * sing_exposure1 * sing_pd1
        set_contribution.append((sing_contribution1 * sing_obligor1)[0])

    for j in np.arange(0, len(exposure2)):
        sing_exposure2 = exposure2[j]
        sing_pd2 = pd2[j]
        sing_obligor2 = obligor2[j]

        def CGFC1_fir_changing3(t):
            """
            this function is for given an x0, solve out t.
            """
            Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
            Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
            Nl11 = PC1_div(t, 1) / (1 - PC1(t))
            ans = alpha1 * beta1 * Nk11 + (alpha2+1) * beta2 * Nk21 + theta1 * Nl11
            return ans - (var - sing_exposure2)

        spt2a = fsolve(CGFC1_fir_changing3, est_spt)
        pdf2a = np.exp(CGFC1_2(spt2a) - spt2a * (var - sing_exposure2)) / np.sqrt(2 * math.pi * CGFC1_sed_2(spt2a))

        def CGFC1_fir_changing4(t):
            """
            this function is for given an x0, solve out t.
            """
            Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
            Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
            Nl11 = PC1_div(t, 1) / (1 - PC1(t))
            ans = alpha1 * beta1 * Nk11 + alpha2 * beta2 * Nk21 + (theta1 + 1) * Nl11
            return ans - (var - sing_exposure2)

        spt2b = fsolve(CGFC1_fir_changing4, est_spt)
        pdf2b = np.exp(CGFC1_3(spt2b) - spt2b * (var - sing_exposure2)) / np.sqrt(2 * math.pi * CGFC1_sed_3(spt2b))

        sing_contribution2 = (pdf2a + gamma12 * pdf2b) / pdf * sing_exposure2 * sing_pd2
        set_contribution.append((sing_contribution2 * sing_obligor2)[0])

    return set_contribution

def VARC_MARTIN(spt):
    ans_set = []

    for i in np.arange(0, len(exposure1)):
        sing_exposure1 = exposure1[i]
        sing_pd1 = pd1[i]
        sing_obligor1 = obligor1[i]
        ans = sing_exposure1 * sing_pd1 * np.exp(spt*sing_exposure1)  * alpha1 * beta1 / (1 - beta1*P1(spt))
        ans_set.append((ans * sing_obligor1)[0])

    for i in np.arange(0, len(exposure2)):
        sing_exposure2 = exposure2[i]
        sing_pd2 = pd2[i]
        sing_obligor2 = obligor2[i]
        ans = sing_exposure2 * sing_pd2 * np.exp(spt * sing_exposure2) * alpha2 * beta2 / (1 - beta2 * P2(spt))
        ans_set.append((ans * sing_obligor2)[0])

    return ans_set




def P1(t):
    ans = sum((np.exp(t*exposure1) -1) * obligor1 * pd1)
    return ans
def P2(t):
    ans = sum((np.exp(t*exposure2) -1) * obligor2 * pd2)
    return ans
def P1_div(t,n):
    """
    :param n: the order of derivative
    """
    ans = sum((exposure1**n) * np.exp(t*exposure1) * obligor1 * pd1 )
    return ans
def P2_div(t,n):
    """
    :param n: the order of derivative
    """
    ans = sum((exposure2**n) * np.exp(t*exposure2) * obligor2 * pd2)
    return ans
def PC1(t):
    ans = gamma11 * P1(t) + gamma12 * P2(t)
    return ans

def PC1_div(t, n):
    ans = gamma11 * P1_div(t, n) + gamma12 * P2_div(t, n)
    return ans
def CGFC1(t):
    ans = - alpha1 * math.log(1-beta1*P1(t)) - alpha2 * math.log(1-beta2*P2(t)) - theta1 * math.log(1 - PC1(t))
    return ans
def CGFC1_1(t):
    ans = - (alpha1+1) * math.log(1-beta1*P1(t)) - alpha2 * math.log(1-beta2*P2(t)) - theta1 * math.log(1 - PC1(t))
    return ans
def CGFC1_2(t):
    ans = - alpha1 * math.log(1-beta1*P1(t)) - (alpha2+1) * math.log(1-beta2*P2(t)) - theta1 * math.log(1 - PC1(t))
    return ans

def CGFC1_new1_3(t):
    ans = - alpha1 * math.log(1-beta1*P1(t)) - alpha2 * math.log(1-beta2*P2(t))
    ans += - theta1 * math.log(1 - gamma12 * P2(t))
    ans += - (theta1 + 1) * math.log( 1 - gamma11 * P1(t))
    return ans
def CGFC1_new2_3(t):
    ans = - alpha1 * math.log(1-beta1*P1(t)) - alpha2 * math.log(1-beta2*P2(t))
    ans += - (theta1 + 1)  * math.log(1 - gamma12 * P2(t))
    ans += - theta1 * math.log( 1 - gamma11 * P1(t))
    return ans

def CGFC1_fir(t):
    Nk11 = P1_div(t, 1) / (1-beta1*P1(t))
    Nk21 = P2_div(t, 1) / (1-beta2*P2(t))
    Nl11 = PC1_div(t, 1) / (1 - PC1(t))
    ans = alpha1*beta1* Nk11 + alpha2*beta2* Nk21 + theta1 * Nl11
    return ans
def CGFC1_sed(t):
    Nk11 = P1_div(t, 1) / (1-beta1*P1(t))
    Nk12 = P1_div(t, 2) / (1-beta1*P1(t))

    Nk21 = P2_div(t, 1) / (1-beta2*P2(t))
    Nk22 = P2_div(t, 2) / (1-beta2*P2(t))

    Nl11 = PC1_div(t, 1)/(1 - PC1(t))
    Nl12 = PC1_div(t, 2)/(1 - PC1(t))

    ans = alpha1*beta1*( Nk12 + beta1*(Nk11**2) ) + \
          alpha2*beta2*( Nk22 + beta2 *(Nk21**2) ) + \
          theta1 * (Nl12 + (Nl11)**2)
    return ans
def CGFC1_sed_1(t):
    Nk11 = P1_div(t, 1) / (1-beta1*P1(t))
    Nk12 = P1_div(t, 2) / (1-beta1*P1(t))

    Nk21 = P2_div(t, 1) / (1-beta2*P2(t))
    Nk22 = P2_div(t, 2) / (1-beta2*P2(t))

    Nl11 = PC1_div(t, 1)/(1 - PC1(t))
    Nl12 = PC1_div(t, 2)/(1 - PC1(t))

    ans = (alpha1 +1) *beta1*( Nk12 + beta1*(Nk11**2) ) + \
          alpha2*beta2*( Nk22 + beta2 *(Nk21**2) ) + \
          theta1 * (Nl12 + (Nl11)**2)
    return ans
def CGFC1_sed_2(t):
    Nk11 = P1_div(t, 1) / (1-beta1*P1(t))
    Nk12 = P1_div(t, 2) / (1-beta1*P1(t))

    Nk21 = P2_div(t, 1) / (1-beta2*P2(t))
    Nk22 = P2_div(t, 2) / (1-beta2*P2(t))

    Nl11 = PC1_div(t, 1)/(1 - PC1(t))
    Nl12 = PC1_div(t, 2)/(1 - PC1(t))

    ans = alpha1 *beta1*( Nk12 + beta1*(Nk11**2) ) + \
          (alpha2 + 1 )*beta2*( Nk22 + beta2 *(Nk21**2) ) + \
          theta1 * (Nl12 + (Nl11)**2)
    return ans
def CGFC1_sed_new1_3(t):
    Nk11 = P1_div(t, 1) / (1-beta1*P1(t))
    Nk12 = P1_div(t, 2) / (1-beta1*P1(t))

    Nk21 = P2_div(t, 1) / (1-beta2*P2(t))
    Nk22 = P2_div(t, 2) / (1-beta2*P2(t))

    Nl_ex1 = gamma12 * P2_div(t, 1) / (1 - gamma12 * P2(t))
    Nl_ex2 = gamma12 * P2_div(t, 2) / (1 - gamma12 * P2(t))

    Nl_one1 = gamma11 * P1_div(t, 1) / (1 - gamma11 * P1(t))
    Nl_one2 = gamma11 * P1_div(t, 2) / (1 - gamma11 * P1(t))

    ans = alpha1 *beta1*( Nk12 + beta1*(Nk11**2) )+alpha2 *beta2*( Nk22 + beta2 *(Nk21**2))
    ans += theta1 * ( Nl_ex2 + (Nl_ex1)**2 )
    ans += (theta1+1) * ( Nl_one2 + (Nl_one1)**2 )
    return ans
def CGFC1_sed_new2_3(t):
    Nk11 = P1_div(t, 1) / (1-beta1*P1(t))
    Nk12 = P1_div(t, 2) / (1-beta1*P1(t))

    Nk21 = P2_div(t, 1) / (1-beta2*P2(t))
    Nk22 = P2_div(t, 2) / (1-beta2*P2(t))

    Nl_ex1 = gamma12 * P2_div(t, 1) / (1 - gamma12 * P2(t))
    Nl_ex2 = gamma12 * P2_div(t, 2) / (1 - gamma12 * P2(t))

    Nl_one1 = gamma11 * P1_div(t, 1) / (1 - gamma11 * P1(t))
    Nl_one2 = gamma11 * P1_div(t, 2) / (1 - gamma11 * P1(t))

    ans = alpha1 *beta1*( Nk12 + beta1*(Nk11**2) )+alpha2 *beta2*( Nk22 + beta2 *(Nk21**2))
    ans += (theta1 + 1) * ( Nl_ex2 + (Nl_ex1)**2 )
    ans += theta1 * ( Nl_one2 + (Nl_one1)**2 )
    return ans

def CGFC1_thi(t):
    Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
    Nk12 = P1_div(t, 2) / (1 - beta1 * P1(t))
    Nk13 = P1_div(t, 3) / (1 - beta1 * P1(t))

    Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
    Nk22 = P2_div(t, 2) / (1 - beta2 * P2(t))
    Nk23 = P2_div(t, 3) / (1 - beta2 * P2(t))

    Nl11 = PC1_div(t, 1) / (1 - PC1(t))
    Nl12 = PC1_div(t, 2) / (1 - PC1(t))
    Nl13 = PC1_div(t, 3) / (1 - PC1(t))


    temp1 = beta1 * alpha1 * (
            Nk13 + 3 * beta1 * Nk11 * Nk12 + 2 * (beta1 ** 2) * (Nk11 ** 3))

    temp2 = beta2 * alpha2 * (
            Nk23 + 3 * beta2 * Nk21 * Nk22 + 2 * (beta2 ** 2) * (Nk21 ** 3))

    temp3 = theta1 * ( Nl13 + 3 * Nl11 * Nl12 + 2 * (Nl11 ** 3))
    return temp1 + temp2 + temp3
def CGFC1_for(t):
    Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
    Nk12 = P1_div(t, 2) / (1 - beta1 * P1(t))
    Nk13 = P1_div(t, 3) / (1 - beta1 * P1(t))
    Nk14 = P1_div(t, 4) / (1 - beta1 * P1(t))

    Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
    Nk22 = P2_div(t, 2) / (1 - beta2 * P2(t))
    Nk23 = P2_div(t, 3) / (1 - beta2 * P2(t))
    Nk24 = P2_div(t, 4) / (1 - beta2 * P2(t))

    Nl11 = PC1_div(t, 1) / (1 - PC1(t))
    Nl12 = PC1_div(t, 2) / (1 - PC1(t))
    Nl13 = PC1_div(t, 3) / (1 - PC1(t))
    Nl14 = PC1_div(t, 4) / (1 - PC1(t))

    temp1 = beta1 * alpha1 * (Nk14 + 3 * beta1 * Nk12 ** 2 + 4 * beta1 * Nk13 * Nk11 +
                12 * (beta1 ** 2) * (Nk11 ** 2) * Nk12 + 6 * ( beta1 ** 3) * ( Nk11 ** 4))

    temp2 = beta2 * alpha2 * (Nk24 + 3 * beta2 * Nk22 ** 2 + 4 * beta2 * Nk23 * Nk21 +
                              12 * (beta2 ** 2) * (Nk21 ** 2) * Nk22 + 6 * (beta2 ** 3) * (Nk21 ** 4))

    temp3 = theta1 * (Nl14 + 3*(Nl12**2) + 4*Nl13*Nl11 + 12*(Nl11**2)*Nl12 + 6*(Nl11**4))

    return temp1 + temp2 + temp3
def C1_tailprob_to_VaR_first_order(var):
    """
    :return: given the var, return the corresponding spt, then return the CDF
    """
    def CGFC1_fir_changing(t):
        """
        this function is for given an x0, solve out t.
        """
        Nk11 = P1_div(t, 1) / (1-beta1*P1(t))
        Nk21 = P2_div(t, 1) / (1-beta2*P2(t))
        Nl11 = PC1_div(t, 1) / (1 - PC1(t))
        ans = alpha1*beta1* Nk11 + alpha2*beta2* Nk21 + theta1 * Nl11
        return ans - var

    spt = fsolve(CGFC1_fir_changing, est_spt)
    uhat = spt * np.sqrt(CGFC1_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CGFC1(spt)))
    ans = norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat)
    return (1-ans) - tailprob
def C1_tailprob_to_VaR_second_order(var):
    """
    :return: given the var, return the corresponding spt, then return the CDF
    """

    def CGFC1_fir_changing(t):
        """
        this function is for given an x0, solve out t.
        """
        Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
        Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
        Nl11 = PC1_div(t, 1) / (1 - PC1(t))
        ans = alpha1 * beta1 * Nk11 + alpha2 * beta2 * Nk21 + theta1 * Nl11
        return ans - var

    spt = fsolve(CGFC1_fir_changing, est_spt)
    uhat = spt * np.sqrt(CGFC1_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CGFC1(spt)))

    lambda3 = CGFC1_thi(spt) / (CGFC1_sed(spt) ** 1.5)
    lambda4 = CGFC1_for(spt) / (CGFC1_sed(spt) ** 2)
    temp = (1 / (what ** 3)) - (1 / (uhat ** 3)) - (lambda3 / (2 * (uhat ** 2))) + (1 / uhat) * (
            (lambda4 / 8) - (5 * (lambda3 ** 2) / 24))
    tp = 1 - (norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat - temp))
    return tp - tailprob
def CGFC1_fir_changing(t):
    """
    this function is for given an x0, solve out t.
    """
    Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
    Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
    Nl11 = PC1_div(t, 1) / (1 - PC1(t))
    ans = alpha1 * beta1 * Nk11 + alpha2 * beta2 * Nk21 + theta1 * Nl11
    return ans - var
def C1_ES2(var, spt):
    uhat = spt * np.sqrt(CGFC1_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CGFC1(spt)))
    ans = EL * (1 - norm.cdf(what)) + norm.pdf(what) * ((var / uhat) - (EL / what))
    ans = ans / tailprob
    return ans
def C1_ES3(var, spt):
    uhat = spt * np.sqrt(CGFC1_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CGFC1(spt)))
    ans = EL * (1 - norm.cdf(what)) + norm.pdf(what) * (
            (var / uhat) - (EL / what) + (EL - var) / what ** 3 + 1 / (spt * uhat))
    ans = ans / tailprob
    return ans
def C1_check_function(var):
    """
    using the optimilization to find the minimum as the ES, and the root as VaR
    :param var:
    :return:
    """

    def CGFC1_fir_changing(t):
        """
        this function is for given an x0, solve out t.
        """
        Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
        Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
        Nl11 = PC1_div(t, 1) / (1 - PC1(t))
        ans = alpha1 * beta1 * Nk11 + alpha2 * beta2 * Nk21 + theta1 * Nl11
        return ans - var

    spt = fsolve(CGFC1_fir_changing, est_spt)
    DEL = EL - var
    uhat = spt * np.sqrt(CGFC1_sed(spt))
    what = np.sign(spt) * np.sqrt(2 * (spt * var - CGFC1(spt)))
    ans = DEL * (1 - norm.cdf(what)) - norm.pdf(what) * ((DEL / what) - (DEL / (what ** 3)) - (1 / (spt * uhat)))
    ans = var + 1 / tailprob * ans
    return ans
def C1_VaRC_Tasche_first_order(var, est_spt):

    def CGFC1_fir_changing(t):
        """
        this function is for given an x0, solve out t.
        """
        Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
        Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
        Nl11 = PC1_div(t, 1) / (1 - PC1(t))
        ans = alpha1 * beta1 * Nk11 + alpha2 * beta2 * Nk21 + theta1 * Nl11
        return ans - var
    spt = fsolve(CGFC1_fir_changing, est_spt)
    pdf = np.exp(CGFC1(spt) - spt * var) / np.sqrt(2 * math.pi * CGFC1_sed(spt))

    set_contribution = []
    # sector 1
    for i in np.arange(0, len(exposure1)):
        sing_exposure1 = exposure1[i]
        sing_pd1 = pd1[i]
        sing_obligor1 = obligor1[i]

        def CGFC1_fir_changing1(t):
            """
            this function is for given an x0, solve out t.
            """
            Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
            Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
            Nl11 = PC1_div(t, 1) / (1 - PC1(t))
            ans = (alpha1 +1)* beta1 * Nk11 + alpha2 * beta2 * Nk21 + theta1 * Nl11
            return ans - (var - sing_exposure1)

        spt1a = fsolve(CGFC1_fir_changing1, est_spt)
        pdf1a = np.exp(CGFC1_1(spt1a) - spt1a * (var-sing_exposure1)) / np.sqrt(2 * math.pi * CGFC1_sed_1(spt1a))

        def CGFC1_fir_changing2(t):
            """
            this function is for given an x0, solve out t.
            """
            Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
            Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
            ans = alpha1 * beta1 * Nk11 + alpha2 * beta2 * Nk21

            Nl_ex1 = gamma12 * P2_div(t, 1) / ( 1 - gamma12 * P2(t))
            Nl_one1 = gamma11 * P1_div(t, 1) / ( 1 - gamma11 * P1(t))

            ans += theta1 * Nl_ex1 + (theta1 + 1) * Nl_one1
            return ans - (var - sing_exposure1)

        spt1b = fsolve(CGFC1_fir_changing2, est_spt)
        pdf1b = np.exp(CGFC1_new1_3(spt1b) - spt1b * (var - sing_exposure1)) / np.sqrt(2 * math.pi * CGFC1_sed_new1_3(spt1b))

        sing_contribution1 = ( beta1 * pdf1a + gamma11 * pdf1b)/pdf * sing_exposure1 * sing_pd1
        set_contribution.append((sing_contribution1 * sing_obligor1)[0])

    for j in np.arange(0, len(exposure2)):
        sing_exposure2 = exposure2[j]
        sing_pd2 = pd2[j]
        sing_obligor2 = obligor2[j]

        def CGFC1_fir_changing3(t):
            """
            this function is for given an x0, solve out t.
            """
            Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
            Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
            Nl11 = PC1_div(t, 1) / (1 - PC1(t))
            ans = alpha1 * beta1 * Nk11 + (alpha2+1) * beta2 * Nk21 + theta1 * Nl11
            return ans - (var - sing_exposure2)

        spt2a = fsolve(CGFC1_fir_changing3, est_spt)
        pdf2a = np.exp(CGFC1_2(spt2a) - spt2a * (var - sing_exposure2)) / np.sqrt(2 * math.pi * CGFC1_sed_2(spt2a))

        def CGFC1_fir_changing4(t):
            """
            this function is for given an x0, solve out t.
            """
            Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
            Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))

            Nl_ex1 = gamma11 * P1_div(t, 1) / (1 - gamma11 * P1(t))

            Nl_one1 = gamma12 * P2_div(t, 1) / (1 - gamma12 * P2(t))

            ans = alpha1 * beta1 * Nk11 + alpha2 * beta2 * Nk21 + theta1 * Nl_ex1 + (theta1 +1 ) * Nl_one1
            return ans - (var - sing_exposure2)

        spt2b = fsolve(CGFC1_fir_changing4, est_spt)
        pdf2b = np.exp(CGFC1_new2_3(spt2b) - spt2b * (var - sing_exposure2)) / np.sqrt(2 * math.pi * CGFC1_sed_new2_3(spt2b))

        sing_contribution2 = ( beta2 * pdf2a + gamma12 * pdf2b) / pdf * sing_exposure2 * sing_pd2
        set_contribution.append((sing_contribution2 * sing_obligor2)[0])

    return set_contribution
def C1_VARC_MARTIN(spt):
    ans_set = []

    for i in np.arange(0, len(exposure1)):
        sing_exposure1 = exposure1[i]
        sing_pd1 = pd1[i]
        sing_obligor1 = obligor1[i]
        ans = sing_exposure1 * sing_pd1 * np.exp(spt*sing_exposure1)  * (alpha1 * beta1 / (1 - beta1*P1(spt)) +  theta1 /( 1- PC1(spt)))
        ans_set.append((ans * sing_obligor1)[0])

    for i in np.arange(0, len(exposure2)):
        sing_exposure2 = exposure2[i]
        sing_pd2 = pd2[i]
        sing_obligor2 = obligor2[i]
        ans = sing_exposure2 * sing_pd2 * np.exp(spt * sing_exposure2) * (alpha2 * beta2 / (1 - beta2 * P2(spt)) + theta1/(1 - PC1(spt)))
        ans_set.append((ans * sing_obligor2)[0])

    return ans_set
def CGFC1_3(t):
    ans = - alpha1 * math.log(1-beta1*P1(t)) - alpha2 * math.log(1-beta2*P2(t)) - (theta1+1) * math.log(1 - PC1(t))
    return ans
def CGFC1_sed_3(t):
    Nk11 = P1_div(t, 1) / (1-beta1*P1(t))
    Nk12 = P1_div(t, 2) / (1-beta1*P1(t))

    Nk21 = P2_div(t, 1) / (1-beta2*P2(t))
    Nk22 = P2_div(t, 2) / (1-beta2*P2(t))

    Nl11 = PC1_div(t, 1)/(1 - PC1(t))
    Nl12 = PC1_div(t, 2)/(1 - PC1(t))

    ans = alpha1 *beta1*( Nk12 + beta1*(Nk11**2) ) + \
          alpha2 *beta2*( Nk22 + beta2 *(Nk21**2) ) + \
          (theta1+1) * (Nl12 + (Nl11)**2)
    return ans
def C1_VaRC_Tasche_first_order(var, est_spt):

    def CGFC1_fir_changing(t):
        """
        this function is for given an x0, solve out t.
        """
        Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
        Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
        Nl11 = PC1_div(t, 1) / (1 - PC1(t))
        ans = alpha1 * beta1 * Nk11 + alpha2 * beta2 * Nk21 + theta1 * Nl11
        return ans - var
    spt = fsolve(CGFC1_fir_changing, est_spt)
    pdf = np.exp(CGFC1(spt) - spt * var) / np.sqrt(2 * math.pi * CGFC1_sed(spt))

    set_contribution = []
    # sector 1
    for i in np.arange(0, len(exposure1)):
        sing_exposure1 = exposure1[i]
        sing_pd1 = pd1[i]
        sing_obligor1 = obligor1[i]

        def CGFC1_fir_changing1(t):
            """
            this function is for given an x0, solve out t.
            """
            Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
            Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
            Nl11 = PC1_div(t, 1) / (1 - PC1(t))
            ans = (alpha1 +1)* beta1 * Nk11 + alpha2 * beta2 * Nk21 + theta1 * Nl11
            return ans - (var - sing_exposure1)

        spt1a = fsolve(CGFC1_fir_changing1, est_spt)
        pdf1a = np.exp(CGFC1_1(spt1a) - spt1a * (var-sing_exposure1)) / np.sqrt(2 * math.pi * CGFC1_sed_1(spt1a))

        def CGFC1_fir_changing2(t):
            """
            this function is for given an x0, solve out t.
            """
            Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
            Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
            Nl11 = PC1_div(t, 1) / (1 - PC1(t))
            ans = alpha1 * beta1 * Nk11 + alpha2 * beta2 * Nk21 + (theta1+1) * Nl11
            return ans - (var - sing_exposure1)

        spt1b = fsolve(CGFC1_fir_changing2, est_spt)
        pdf1b = np.exp(CGFC1_3(spt1b) - spt1b * (var - sing_exposure1)) / np.sqrt(2 * math.pi * CGFC1_sed_3(spt1b))

        sing_contribution1 = ( pdf1a + gamma11 * pdf1b)/pdf * sing_exposure1 * sing_pd1
        set_contribution.append((sing_contribution1 * sing_obligor1)[0])

    for j in np.arange(0, len(exposure2)):
        sing_exposure2 = exposure2[j]
        sing_pd2 = pd2[j]
        sing_obligor2 = obligor2[j]

        def CGFC1_fir_changing3(t):
            """
            this function is for given an x0, solve out t.
            """
            Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
            Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
            Nl11 = PC1_div(t, 1) / (1 - PC1(t))
            ans = alpha1 * beta1 * Nk11 + (alpha2+1) * beta2 * Nk21 + theta1 * Nl11
            return ans - (var - sing_exposure2)

        spt2a = fsolve(CGFC1_fir_changing3, est_spt)
        pdf2a = np.exp(CGFC1_2(spt2a) - spt2a * (var - sing_exposure2)) / np.sqrt(2 * math.pi * CGFC1_sed_2(spt2a))

        def CGFC1_fir_changing4(t):
            """
            this function is for given an x0, solve out t.
            """
            Nk11 = P1_div(t, 1) / (1 - beta1 * P1(t))
            Nk21 = P2_div(t, 1) / (1 - beta2 * P2(t))
            Nl11 = PC1_div(t, 1) / (1 - PC1(t))
            ans = alpha1 * beta1 * Nk11 + alpha2 * beta2 * Nk21 + (theta1 + 1) * Nl11
            return ans - (var - sing_exposure2)

        spt2b = fsolve(CGFC1_fir_changing4, est_spt)
        pdf2b = np.exp(CGFC1_3(spt2b) - spt2b * (var - sing_exposure2)) / np.sqrt(2 * math.pi * CGFC1_sed_3(spt2b))

        sing_contribution2 = ( pdf2a + gamma12 * pdf2b) / pdf * sing_exposure2 * sing_pd2
        set_contribution.append((sing_contribution2 * sing_obligor2)[0])

    return set_contribution


