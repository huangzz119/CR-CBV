import numpy as np
import math

from DataPre.info_portfolio_CBV import portfolio_info,CBVmodel

import os
import sys
filename = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(filename)

from scipy.stats import norm
from scipy.optimize import fsolve,minimize
from cgf_functions import cgf_calculation
import pandas as pan
import time


class SPAcalculation():

    def __init__(self, coca, est_spt = 0.4, est_var = 3, tailprob= 0.05):

        self.coca = coca

        self.df = coca.df
        self.K = coca.K
        self.Lmean = coca.Lmean

        self.L = coca.L
        self.delta = coca.delta
        self.thetak = coca.thetak
        self.thetal = coca.thetal
        self.gamma = coca.gamma


        # the confidence level
        self.tailprob = tailprob

        # estimator the value
        self.est_spt = est_spt
        self.est_var = est_var

    def function_tailprob_changeto_VaR(self, var):
        """
        :return: given the var, return the corresponding spt, then return the CDF
        """

        # this function is to change x to t
        def KL_fir_changing(t):
            """
            this function is for given an x0, solve out t.
            :param t:
            :return:
            """
            ans = 0
            # sum of all sectors
            for i in np.arange(0, self.K):
                pk = self.coca.PK(i + 1, t)
                pk1 = self.coca.PK_drv(i + 1, t, 1)
                temp = self.delta[i] * self.thetak[i] * pk1 / (1 - self.delta[i] * pk)
                ans += temp

            for l in np.arange(0, self.L):
                pl = 0
                pl1 = 0
                for i in np.arange(0, self.K):
                    pl += self.coca.PK(i + 1, t) * self.gamma[l][i]
                    pl1 += self.coca.PK_drv(i + 1, t, 1) * self.gamma[l][i]
                temp_ = self.thetal[l] * pl1 / (1 - pl)
                ans += temp_
            return ans - var

        troot = fsolve(KL_fir_changing, self.est_spt)

        uhat = troot * np.sqrt(self.coca.KL_sec(troot))
        try:
            what = np.sign(troot) * np.sqrt(2 * (troot * var - self.coca.KL(troot)))
        except:
            print("the sqrt term in what is invalid")
            ans = np.nan
        else:
            ans = norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat)
        return (1-ans) - self.tailprob
    def function_tailprob_changeto_VaR_2nd(self, var):
        """
        :return: given the var, return the corresponding spt, then return the CDF
        """

        # this function is to change x to t
        def KL_fir_changing(t):
            """
            this function is for given an x0, solve out t.
            :param t:
            :return:
            """
            ans = 0
            # sum of all sectors
            for i in np.arange(0, self.K):
                pk = self.coca.PK(i + 1, t)
                pk1 = self.coca.PK_drv(i + 1, t, 1)
                temp = self.delta[i] * self.thetak[i] * pk1 / (1 - self.delta[i] * pk)
                ans += temp

            for l in np.arange(0, self.L):
                pl = 0
                pl1 = 0
                for i in np.arange(0, self.K):
                    pl += self.coca.PK(i + 1, t) * self.gamma[l][i]
                    pl1 += self.coca.PK_drv(i + 1, t, 1) * self.gamma[l][i]
                temp_ = self.thetal[l] * pl1 / (1 - pl)
                ans += temp_
            return ans - var

        troot = fsolve(KL_fir_changing, self.est_spt)

        uhat = troot * np.sqrt(self.coca.KL_sec(troot))

        KL2 = self.coca.KL_sec(troot)
        lambda3 = self.coca.KL_thi(troot) / (KL2 ** 1.5)
        lambda4 = self.coca.KL_for(troot) / (KL2 ** 2)
        try:
            what = np.sign(troot) * np.sqrt(2 * (troot * var - self.coca.KL(troot)))
        except:
            print("the sqrt term in what is invalid")
            ans = np.nan
        else:
           # temp = 0
            temp = (1 / (what ** 3)) - (1 / (uhat ** 3)) - (lambda3 / (2 * (uhat ** 2))) + (1 / uhat) * (
                        lambda4 / 8 - 5 * (lambda3 ** 2) / 24)
            ans = norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat -temp)
        return (1-ans) - self.tailprob

    def solver_tailprob_changeto_VaR_spt(self):
        """
        :return: return the corresponding VaR and spt given the tail probability
        """
        var = fsolve(self.function_tailprob_changeto_VaR, self.est_var)

        def KL_fir_changing(t):
            """
            this function is for given an x0, solve out t.
            :param t:
            :return:
            """
            ans = 0
            # sum of all sectors
            for i in np.arange(0, self.K):
                pk = self.coca.PK(i + 1, t)
                pk1 = self.coca.PK_drv(i + 1, t, 1)
                temp = self.delta[i] * self.thetak[i] * pk1 / (1 - self.delta[i] * pk)
                ans += temp

            for l in np.arange(0, self.L):
                pl = 0
                pl1 = 0
                for i in np.arange(0, self.K):
                    pl += self.coca.PK(i + 1, t) * self.gamma[l][i]
                    pl1 += self.coca.PK_drv(i + 1, t, 1) * self.gamma[l][i]
                temp_ = self.thetal[l] * pl1 / (1 - pl)
                ans += temp_
            return ans - var

        spt = fsolve(KL_fir_changing, self.est_spt)

        return var, spt
    def solver_tailprob_changeto_VaR_spt_2nd(self):

        """
        :return: return the corresponding VaR and spt given the tail probability
        """
        var = fsolve( self.function_tailprob_changeto_VaR_2nd, self.est_var)

        def KL_fir_changing(t):
            """
            this function is for given an x0, solve out t.
            :param t:
            :return:
            """
            ans = 0
            # sum of all sectors
            for i in np.arange(0, self.K):
                pk = self.coca.PK(i + 1, t)
                pk1 = self.coca.PK_drv(i + 1, t, 1)
                temp = self.delta[i] * self.thetak[i] * pk1 / (1 - self.delta[i] * pk)
                ans += temp

            for l in np.arange(0, self.L):
                pl = 0
                pl1 = 0
                for i in np.arange(0, self.K):
                    pl += self.coca.PK(i + 1, t) * self.gamma[l][i]
                    pl1 += self.coca.PK_drv(i + 1, t, 1) * self.gamma[l][i]
                temp_ = self.thetal[l] * pl1 / (1 - pl)
                ans += temp_
            return ans - var

        spt = fsolve(KL_fir_changing, self.est_spt)

        return var,spt


    # two calls of SPA
    def ES1(self):

        var = fsolve( self.function_tailprob_changeto_VaR, self.est_var)

        def KL_fir_change(t):

            ans = self.coca.KL_fir(t) + self.coca.KL_sec(t) / self.coca.KL_fir(t)
            return ans - var

        troot = fsolve(KL_fir_change, np.array([0.1]))

        KL_ = self.coca.KL(troot) + np.log(self.coca.KL_fir(troot)) - np.log(self.coca.KL_fir(0))
        try:
            KLsec_ = self.coca.KL_sec(troot) + (self.coca.KL_thi(troot) * self.coca.KL_fir(troot) - self.coca.KL_sec(troot) ** 2) / (self.coca.KL_fir(troot) ** 2)
        except:
            print("the denominate term is invalid")
            ans = np.nan

        else:
            uhat = troot * np.sqrt(KLsec_)
            try:
                what = np.sign(troot) * np.sqrt(2 * (troot * var - KL_))
            except:
                print("the sqrt term in w_hat is invalid")
                ans = np.nan
            else:
                cdf_ = norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat)
                ans = self.Lmean / (self.tailprob) * (1 - cdf_)
        return ans

    # ES Approximation formula
    def ES2(self):

        var, spt = self.solver_tailprob_changeto_VaR_spt()

        uhat = spt * np.sqrt(self.coca.KL_sec(spt))
        try:
            what = np.sign(spt) * np.sqrt(2 * (spt * var - self.coca.KL(spt)))
        except:
            print("the sqrt term in w_hat is invalid")
            ans = None
        else:
            ans = self.Lmean * (1 - norm.cdf(what)) + norm.pdf(what) * ((var / uhat) - (self.Lmean / what))
            ans = ans / self.tailprob
        return ans

    def ES3(self):

        var, spt = self.solver_tailprob_changeto_VaR_spt()

        uhat = spt * np.sqrt(self.coca.KL_sec(spt))
        try:
            what = np.sign(spt) * np.sqrt(2 * (spt * var - self.coca.KL(spt)))
        except:
            print("the sqrt term in w_hat is invalid")
            ans = None
        else:
            ans = self.Lmean * (1 - norm.cdf(what)) + norm.pdf(what) * (
                    (var / uhat) - (self.Lmean / what) + (self.Lmean - var) / what ** 3 + 1 / (spt * uhat))
            ans = ans / self.tailprob
        return ans

    def check_function(self, var):
        """
        using the optimilization to find the minimum as the ES, and the root as VaR
        :param var:
        :return:
        """

        def KL_fir_changing(t):
            """
            this function is for given an x0, solve out t.
            :param t:
            :return:
            """
            ans = 0
            # sum of all sectors
            for i in np.arange(0, self.K):
                pk = self.coca.PK(i + 1, t)
                pk1 = self.coca.PK_drv(i + 1, t, 1)
                temp = self.delta[i] * self.thetak[i] * pk1 / (1 - self.delta[i] * pk)
                ans += temp

            for l in np.arange(0, self.L):
                pl = 0
                pl1 = 0
                for i in np.arange(0, self.K):
                    pl += self.coca.PK(i + 1, t) * self.gamma[l][i]
                    pl1 += self.coca.PK_drv(i + 1, t, 1) * self.gamma[l][i]
                temp_ = self.thetal[l] * pl1 / (1 - pl)
                ans += temp_
            return ans - var

        spt = fsolve(KL_fir_changing, self.est_spt)

        DEL = self.Lmean - var

        uhat = spt * np.sqrt(self.coca.KL_sec(spt))
        try:
            what = np.sign(spt) * np.sqrt(2 * (spt * var - self.coca.KL(spt)))
        except:
            print("the sqrt term in w_hat is invalid")
            ans = None
        else:
            ans = DEL * (1 - norm.cdf(what)) - norm.pdf(what) * ( (DEL / what) - (DEL / (what**3)) - (1 / (spt*uhat)))
            ans = var + 1/self.tailprob * ans
        return ans



if __name__ == '__main__':


    pf = portfolio_info()
    pf.init_rcobligor()

    cbvpara = CBVmodel()
    cbvpara.CBV2()

    coca = cgf_calculation(pf, cbvpara)
    root = coca.QL_root()

    print("the mean of the portfolio loss:", coca.Lmean)
    print("the first order derivative of cgf at t=0:", coca.KL_fir(0))
    print("minimum upper bound of t inside the root of cgf:", root)

    ans_set = []
    spt_set = np.arange(0.01, root-0.005, 0.05)
    for t in spt_set :
        ans = coca.KL_fir(t)
        ans_set.append(ans)

    df_ans = pan.DataFrame({"spt":spt_set, "KL_fir":ans_set})
    print("the relation between spt and the first order of CGF:", df_ans)

    model = SPAcalculation(coca,  est_spt = 2.8, est_var = 0.45, tailprob= 0.05)

    start_time = time.time()
    var, spt = model.solver_tailprob_changeto_VaR_spt_2nd()
    print("var value", var)
    print("--- %s seconds in var---" % (time.time() - start_time))
    # var = 0.82766, spt = 2.5232
    # var = 0.445, spt = 2.789

    start_time = time.time()
    es1 = model.ES1()
    print("--- %s seconds in ES1---" % (time.time() - start_time))
    # 0.7134

    start_time = time.time()
    es2 = model.ES2()
    print("--- %s seconds in ES2---" % (time.time() - start_time))
    # 0.800

    start_time = time.time()
    es3 = model.ES3()
    print("--- %s seconds in ES3---" % (time.time() - start_time))
    # 0.6548

    start_time = time.time()
    init = np.array(0.445, dtype="float")
    es4 = minimize(model.check_function, init, method='Nelder-Mead')
    print("--- %s seconds in ES4---" % (time.time() - start_time))
    # 0.6546







