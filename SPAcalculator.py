import pandas as pan
import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import fsolve
import json
import csv

class SPA:
    # portfolio information
    """
    1. all tenors are set to one year
    2. obligors are given by peers(counterparty)
    3. each obligor was given a hundred percent weight to the sector which it belongs;
    so the correlation of sectors are connected by common background factors
    4. neglect the assumption of normalized sector

    counterparty number: 3000 obligors 2 term loans each
    sector number: 10
    rating number: 7

    PD: probability of default
    SD(PD): the standard derivation of probability of default
    EAD: exposure at default
    LGD: loss given default
    PL: potential loss, PL = EAD * LGD
    EL: expected loss, EL = PL * PD
    """

    # 类变量
    df_origin = pan.read_excel("Testing.xls", header=1)
    df_ = df_origin.iloc[0:df_origin.shape[0], [2, 7,  19, 15, 6]]
    df_.columns = ["obligor", "PD",    "PL", "sector", "rating"]
    df_["EL"] = df_.apply(lambda x: x["PD"] * x["PL"], axis=1)

    # Sort Descending
    df_ = df_.sort_values(by='PL', ascending=False)
    df_new = df_.iloc[range(0, len(df_), 2)]
    df_new.loc[:,"PL"] = list(map(lambda x: x * 2 / 1000000000, df_new["PL"]))
    df_new.loc[:,"EL"] = list(map(lambda x: x * 2 / 1000000000, df_new["EL"]))

    df = df_new


    K=10  # sector number
    Lmean = sum(df.EL)  # the mean of expected loss

    def __init__(self, L, delta, thetak, gamma, thetal, i0 = 1, x0 = 2.5, alpha0 = 0.95):
        self.L = L
        self.delta = delta
        self.thetak = thetak
        self.gamma = gamma
        self.thetal = thetal

        # these parameters are set for contribution
        self.ik = SPA.df.sector.values[i0-1]  #ik represents sectors
        self.ipd = SPA.df.PD.values[i0-1]
        self.ipl = SPA.df.PL.values[i0-1]

        # this parameters are set for important sampling
        self.x0 = x0

        # this parameters are set for given a particular alpha
        self.alpha0 = alpha0


    def up_spt(self):
        """
        :return: the minimum upper boundary of saddlepoint
        """
        up_bound = []
        for k in np.arange(0, SPA.K):
            st = SPA.df.loc[ SPA.df["sector"] == k + 1]
            up = math.log(1 / self.delta[k] + sum(st.PD)) / sum(st.PD * st.PL)
            up_bound.append(up)

        for l in np.arange(0, self.L):
            num = 0
            den = 0
            for i in np.arange(0, SPA.K):
                st = SPA.df.loc[ SPA.df["sector"] == i + 1]
                temp_num = sum(st.PD) * self.gamma[l][i]
                temp_den = sum(st.PD * st.PL) * self.gamma[l][i]
                num = num + temp_num
                den = den + temp_den
            up = math.log(1 + num) / den
            up_bound.append(up)

        min_upbound = min(up_bound)
        return min_upbound

    def QL_root(self):
        """
        :return: the root inside log as the minimum upper boundary
        """
        root_set = []
        gamma = self.gamma
        for l in np.arange(0, self.L):
            def eff(t):
                Ql = 0
                for i in np.arange(0, SPA.K):
                    st = SPA.df.loc[SPA.df["sector"] == i + 1]
                    tempQl = sum(st.PD * (np.exp(t * st.PL) - 1)) * gamma[l][i]
                    Ql = Ql + tempQl
                ans = 1 - Ql
                return ans
            root_ = fsolve(eff, 1.5)
            root_set.append(root_)

        min_root = min(root_set)
        return min_root

    def __PK(self, k, t):
        """
        :param k: k-th sector
        :param t: saddlepoint
        :return: Qk value
        """
        st = SPA.df.loc[SPA.df["sector"] == k]
        pk = sum(st.PD * (np.exp(t * st.PL) - 1))
        return pk

    def __PK_drv(self, k, t, n):
        """
        :param n: the order of derivative
        :return:
        """
        st = SPA.df.loc[SPA.df["sector"] == k]
        pk_drv = sum((st.PL ** n) * st.PD * (np.exp(t * st.PL)))
        return pk_drv



    def KL(self, t):
        ans = 0
        # sum of all sectors
        for i in np.arange(0, SPA.K):
            pk = self.__PK(i + 1, t)
            temp = - self.thetak[i] * math.log(1 - self.delta[i] * pk)
            ans = ans + temp

        for l in np.arange(0, self.L):
            pl = 0
            for i in np.arange(0, SPA.K):
                pl += self.__PK(i + 1, t) * self.gamma[l][i]
            try:
                temp_ = - self.thetal[l] * math.log(1 - pl)
            except:
                print("the sqrt term in w_hat is invalid")
                ans = np.nan
            else:
                ans = ans + temp_
        return ans

    def KL_fir(self, t):
        ans = 0

        # sum of all sectors
        for i in np.arange(0, SPA.K):
            pk = self.__PK(i + 1, t)
            pk1 = self.__PK_drv(i + 1, t, 1)
            temp = self.delta[i] * self.thetak[i] * pk1 / (1 - self.delta[i] * pk)
            ans += temp

        for l in np.arange(0, self.L):
            pl = 0
            pl1 = 0
            for i in np.arange(0, SPA.K):
                pl += self.__PK(i + 1, t) * self.gamma[l][i]
                pl1 += self.__PK_drv(i + 1, t, 1) * self.gamma[l][i]
            temp_ = self.thetal[l] * pl1 / (1 - pl)
            ans += temp_

        return ans

    def KL_sec(self, t):
        ans = 0

        # sum of all sectors
        for i in np.arange(0, SPA.K):
            pk = self.__PK(i + 1, t)
            pk1 = self.__PK_drv(i + 1, t, 1)
            pk2 = self.__PK_drv(i + 1, t, 2)
            Nk1 = pk1 / (1 - self.delta[i] * pk)
            Nk2 = pk2 / (1 - self.delta[i] * pk)
            temp = self.delta[i] * self.thetak[i] * (Nk2 + self.delta[i] * (Nk1 ** 2))
            ans += temp

        for l in np.arange(0, self.L):
            pl = 0
            pl1 = 0
            pl2 = 0
            for i in np.arange(0, SPA.K):
                pl += self.__PK(i + 1, t) * self.gamma[l][i]
                pl1 += self.__PK_drv(i + 1, t, 1) * self.gamma[l][i]
                pl2 += self.__PK_drv(i + 1, t, 2) * self.gamma[l][i]
            Nl1 = pl1 / (1 - pl)
            Nl2 = pl2 / (1 - pl)
            temp_ = self.thetal[l] * (Nl2 + (Nl1 ** 2))
            ans += temp_

        return ans

    def KL_thi(self, t):
        ans = 0

        # sum of all sectors
        for i in np.arange(0, SPA.K):
            pk = self.__PK(i + 1, t)
            pk1 = self.__PK_drv(i + 1, t, 1)
            pk2 = self.__PK_drv(i + 1, t, 2)
            pk3 = self.__PK_drv(i + 1, t, 3)
            Nk1 = pk1 / (1 - self.delta[i] * pk)
            Nk2 = pk2 / (1 - self.delta[i] * pk)
            Nk3 = pk3 / (1 - self.delta[i] * pk)
            temp = self.delta[i] * self.thetak[i] * (
                        Nk3 + 3 * self.delta[i] * Nk1 * Nk2 + 2 * (self.delta[i] ** 2) * (Nk1 ** 3))
            ans = ans + temp

        for l in np.arange(0, self.L):
            pl = 0
            pl1 = 0
            pl2 = 0
            pl3 = 0
            for i in np.arange(0, SPA.K):
                pl += self.__PK(i + 1, t) * self.gamma[l][i]
                pl1 += self.__PK_drv(i + 1, t, 1) * self.gamma[l][i]
                pl2 += self.__PK_drv(i + 1, t, 2) * self.gamma[l][i]
                pl3 += self.__PK_drv(i + 1, t, 3) * self.gamma[l][i]
            Nl1 = pl1 / (1 - pl)
            Nl2 = pl2 / (1 - pl)
            Nl3 = pl3 / (1 - pl)
            temp_ = self.thetal[l] * (Nl3 + 3 * Nl1 * Nl2 + 2 * Nl1 ** 3)
            ans = ans + temp_
        return ans

    def KL_for(self, t):
        ans = 0

        # sum of all sectors
        for i in np.arange(0, SPA.K):
            pk = self.__PK(i + 1, t)
            pk1 = self.__PK_drv(i + 1, t, 1)
            pk2 = self.__PK_drv(i + 1, t, 2)
            pk3 = self.__PK_drv(i + 1, t, 3)
            pk4 = self.__PK_drv(i + 1, t, 4)
            Nk1 = pk1 / (1 - self.delta[i] * pk)
            Nk2 = pk2 / (1 - self.delta[i] * pk)
            Nk3 = pk3 / (1 - self.delta[i] * pk)
            Nk4 = pk4 / (1 - self.delta[i] * pk)
            temp = self.delta[i] * self.thetak[i] * (Nk4 + 3 * self.delta[i] * Nk2 ** 2 + 4 * self.delta[i] * Nk3 * Nk1
                                           + 12 * (self.delta[i] ** 2) * (Nk1 ** 2) * Nk2 + 6 * (self.delta[i] ** 3) * (
                                                       Nk1 ** 4))
            ans = ans + temp

        for l in np.arange(0, self.L):
            pl = 0
            pl1 = 0
            pl2 = 0
            pl3 = 0
            pl4 = 0
            for i in np.arange(0, SPA.K):
                pl += self.__PK(i + 1, t) * self.gamma[l][i]
                pl1 += self.__PK_drv(i + 1, t, 1) * self.gamma[l][i]
                pl2 += self.__PK_drv(i + 1, t, 2) * self.gamma[l][i]
                pl3 += self.__PK_drv(i + 1, t, 3) * self.gamma[l][i]
                pl4 += self.__PK_drv(i + 1, t, 4) * self.gamma[l][i]
            Nl1 = pl1 / (1 - pl)
            Nl2 = pl2 / (1 - pl)
            Nl3 = pl3 / (1 - pl)
            Nl4 = pl3 / (1 - pl)
            temp_ = self.thetal[l] * (Nl4 + 3 * Nl2 ** 2 + 4 * Nl3 * Nl1 + 12 * (Nl1 ** 2) * Nl2 + 6 * (Nl1 ** 4))
            ans = ans + temp_

        return ans

    def skewness(self):
        ans = self.KL_thi(0) / (self.KL_sec(0))**1.5
        return ans

    def kurtosis(self):
        ans = self.KL_for(0) / (self.KL_sec(0))**2 + 3
        return ans

    def CDF_changing(self, x):
        """
        :return: given the x, return the corresponding spt, then return the CDF
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
            for i in np.arange(0, SPA.K):
                pk = self.__PK(i + 1, t)
                pk1 = self.__PK_drv(i + 1, t, 1)
                temp = self.delta[i] * self.thetak[i] * pk1 / (1 - self.delta[i] * pk)
                ans += temp

            for l in np.arange(0, self.L):
                pl = 0
                pl1 = 0
                for i in np.arange(0, SPA.K):
                    pl += self.__PK(i + 1, t) * self.gamma[l][i]
                    pl1 += self.__PK_drv(i + 1, t, 1) * self.gamma[l][i]
                temp_ = self.thetal[l] * pl1 / (1 - pl)
                ans += temp_
            return ans - x

        troot = fsolve(KL_fir_changing, 1)

        uhat = troot * np.sqrt(self.KL_sec(troot))
        try:
            what = np.sign(troot) * np.sqrt(2 * (troot * x - self.KL(troot)))
        except:
            print("the sqrt term in what is invalid")
            ans = np.nan
        else:
            ans = norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat)
        return ans - self.alpha0


    def VaR_given_alpha(self):
        ans = fsolve( self.CDF_changing, 1)
        return ans

    def spt_given_alpha(self, var):

        def KL_fir_changing(t):
            """
            this function is for given an x0, solve out t.
            :param t:
            :return:
            """
            ans = 0
            # sum of all sectors
            for i in np.arange(0, SPA.K):
                pk = self.__PK(i + 1, t)
                pk1 = self.__PK_drv(i + 1, t, 1)
                temp = self.delta[i] * self.thetak[i] * pk1 / (1 - self.delta[i] * pk)
                ans += temp

            for l in np.arange(0, self.L):
                pl = 0
                pl1 = 0
                for i in np.arange(0, SPA.K):
                    pl += self.__PK(i + 1, t) * self.gamma[l][i]
                    pl1 += self.__PK_drv(i + 1, t, 1) * self.gamma[l][i]
                temp_ = self.thetal[l] * pl1 / (1 - pl)
                ans += temp_
            return ans - var

        ans = fsolve(KL_fir_changing, 1)
        return ans


    def CDF(self, x, t):
        """
        :return: the cumulative probability
        """
        uhat = t * np.sqrt(self.KL_sec(t))
        try:
            what = np.sign(t) * np.sqrt(2 * (t * x - self.KL(t)))
        except:
            print("the sqrt term in what is invalid")
            ans = np.nan
        else:
            ans = norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat)

        return ans



    # two calls of SPA
    def ES1(self, x, alpha):

        def KL_fir_(t):

            ans = self.KL_fir(t) + self.KL_sec(t) / self.KL_fir(t)
            return ans - x

        t_ = fsolve(KL_fir_, np.array([0.1]))

        KL_ = self.KL(t_) + np.log(self.KL_fir(t_)) - np.log(self.KL_fir(0))
        try:
            KLsec_ = self.KL_sec(t_) + (self.KL_thi(t_) * self.KL_fir(t_) - self.KL_sec(t_) ** 2) / (self.KL_fir(t_) ** 2)
        except:
            print("the denominate term is invalid")
            ans = np.nan

        else:
            uhat = t_ * np.sqrt(KLsec_)
            try:
                what = np.sign(t_) * np.sqrt(2 * (t_ * x - KL_))
            except:
                print("the sqrt term in w_hat is invalid")
                ans = np.nan
            else:
                cdf_ = norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat)
                ans = SPA.Lmean / (1 - alpha) * (1 - cdf_)
        return ans

    # ES Approximation formula
    def ES2(self, x, t, alpha):
        uhat = t * np.sqrt(self.KL_sec(t))
        try:
            what = np.sign(t) * np.sqrt(2 * (t * x - self.KL(t)))
        except:
            print("the sqrt term in w_hat is invalid")
            ans = None
        else:
            ans = SPA.Lmean * (1 - norm.cdf(what)) + norm.pdf(what) * ((x / uhat) - (SPA.Lmean / what))
            ans = ans / (1 - alpha)
        return ans

    def ES3(self, x, t, alpha):
        uhat = t * np.sqrt(self.KL_sec(t))
        try:
            what = np.sign(t) * np.sqrt(2 * (t * x - self.KL(t)))
        except:
            print("the sqrt term in w_hat is invalid")
            ans = None
        else:
            ans = SPA.Lmean * (1 - norm.cdf(what)) + norm.pdf(what) * (
                    (x / uhat) - (SPA.Lmean / what) + (SPA.Lmean - x) / what ** 3 + 1 / (t * uhat))
            ans = ans / (1 - alpha)
        return ans

    def __KRC(self, t):
        """
        this function is to generating the multivarite CGF of portfolio L and obligor i
        :param ik:  the sector where obligor i belongs to
        :param ipd:  the probability of default of obligor i
        :param ipl: the potential loss of obligor i
        :return: the answer of KRC
        """
        idx = self.ik - 1

        den1 = 1 - self.delta[idx] * self.__PK(self.ik, t)
        num1 = self.delta[idx] * self.thetak[idx]
        ins = num1 / den1

        for l in np.arange(0, self.L):
            pl = 0
            for i in np.arange(0, SPA.K):
                pl += self.__PK(i + 1, t) * self.gamma[l][i]
            ins += (self.thetal[l] * self.gamma[l][idx]) / (1 - pl)

        ans = self.ipd * np.exp(t * self.ipl) * ins

        return ans

    def __KRC_fir(self, t):

        idx = self.ik - 1


        a = self.ipl / (1 - self.delta[idx] * self.__PK(self.ik, t))
        b = self.delta[idx] * self.__PK_drv(self.ik, t, 1) / (1 - self.delta[idx] * self.__PK(self.ik, t)) ** 2
        ins = self.thetak[idx] * self.delta[idx] * (a + b)

        for l in np.arange(0,self.L):
            pl = 0
            pl1 = 0
            for i in np.arange(0, SPA.K):
                pl += self.__PK(i + 1, t) * self.gamma[l][i]
                pl1 += self.__PK_drv(i + 1, t, 1) * self.gamma[l][i]
            ins += self.thetal[l] * self.gamma[l][idx] * (self.ipl / (1 - pl) + pl1 / (1 - pl) ** 2)

        ans = self.ipd * np.exp(t * self.ipl) * ins
        return ans

    def __KRC_sed(self, t):
        pk = self.__PK(self.ik, t)
        pk1 = self.__PK_drv(self.ik, t, 1)
        pk2 = self.__PK_drv(self.ik, t, 2)

        idx = self.ik -1

        a1 = (self.ipl ** 2 + self.ipl * self.delta[idx] * pk1) / (1 - self.delta[idx] * pk)
        a2 = (self.ipl * self.delta[idx] * pk1 + self.delta[idx] * pk2) / (1 - self.delta[idx] * pk) ** 2
        a3 = (2 * (self.delta[idx] ** 2) * (pk1 ** 2)) / (1 - self.delta[idx] * pk) ** 3
        ins = self.thetak[idx] * self.delta[idx] * (a1 + a2 + a3)

        for l in np.arange(0, self.L):
            pl = 0
            pl1 = 0
            pl2 = 0
            for i in np.arange(0, SPA.K):
                pl += self.__PK(i + 1, t) * self.gamma[l][i]
                pl1 += self.__PK_drv(i + 1, t, 1) * self.gamma[l][i]
                pl1 += self.__PK_drv(i + 1, t, 2) * self.gamma[l][i]
            b1 = (self.ipl ** 2 + self.ipl * pl1) / (1 - pl)
            b2 = (self.ipl * pl1 + pl2) / (1 - pl) ** 2
            b3 = (2 * pl1 ** 2) / (1 - pl) ** 3
            ins += self.thetal[l] * self.gamma[l][idx] * (b1 + b2 + b3)

        ans = self.ipd * np.exp(t * self.ipl) * ins
        return ans

    def __RHO(self, r, t):
        """
        :param r: the order of derivative
        :return: standardized cumulant of order r
        """
        if r == 3:
            ans = self.KL_thi(t) / self.KL_sec(t) ** (3 / 2)
        if r == 4:
            ans = self.KL_for(t) / self.KL_sec(t) ** 2
        return ans

    def KL_fir_x(self, t):
        """
        this function is for given an x0, solve out t.
        :param t:
        :return:
        """
        ans = 0
        # sum of all sectors
        for i in np.arange(0, SPA.K):
            pk = self.__PK(i + 1, t)
            pk1 = self.__PK_drv(i + 1, t, 1)
            temp = self.delta[i] * self.thetak[i] * pk1 / (1 - self.delta[i] * pk)
            ans += temp

        for l in np.arange(0, self.L):
            pl = 0
            pl1 = 0
            for i in np.arange(0, SPA.K):
                pl += self.__PK(i + 1, t) * self.gamma[l][i]
                pl1 += self.__PK_drv(i + 1, t, 1) * self.gamma[l][i]
            temp_ = self.thetal[l] * pl1 / (1 - pl)
            ans += temp_
        return ans - self.x0

    def VARC_MARTIN(self):
        """
        to calculate the VARC, the necessary input is i0 and x0
        :return:
        """
        # from the x0 to get t0
        t0 = fsolve(self.KL_fir_x, 1)

        # the obligor is in sector ik, but the index for para is idx
        idx = self.ik - 1
        temp1 = self.ipl * self.ipd * np.exp(t0 * self.ipl)

        # the second term:
        den1 = 1 - self.delta[idx] * self.__PK(self.ik, t0)
        num1 = self.delta[idx] * self.thetak[idx]
        temp2 = num1 / den1

        # the third term:

        for l in np.arange(0, self.L):
            pl = 0
            for i in np.arange(0, SPA.K):
                pl += self.__PK(i + 1, t0) * self.gamma[l][i]
            temp2 += self.thetal[l] / (1 - pl)

        ans = temp1 * temp2

        return ans

    def VARC_KIM(self):
        """
        :param t: the saddlepoint corresponding to loss x0
        :return: the VaRC of obligor i
        """
        t = fsolve(self.KL_fir_x, 1)

        num = self.__RHO(3, t) * self.__KRC_fir(t) / (2 * np.sqrt(self.KL_sec(t))) - self.__KRC_sed(t) / (2 * self.KL_sec(t))
        den = 1 + ((self.__RHO(4, t) / 8 ) - ( 5 * self.__RHO(3, t) ** 2 / 24))

        ans = self.__KRC(t) + num / den

        return self.ipl * ans

    def __DMEAN(self):
        idx = self.ik -1
        ans = self.ipd * self.delta[idx] * self.thetak[idx]
        for l in np.arange(0, self.L):
            ans += self.ipd * self.gamma[l][idx] * self.thetal[l]
        return ans

    def ESC_KIM(self):
        """
        kim's formula to calculate ESC
        :return:
        """
        dmean = self.__DMEAN()
        t = fsolve(self.KL_fir_x, 1)
        zhat = t * np.sqrt(self.KL_sec(t))

        tp = 1 / ((1 - self.CDF(self.x0, t)) * np.sqrt(2 * np.pi)) * np.exp(self.KL(t) - self.x0 * t)

        a = (self.__KRC(t) - self.__KRC(0)) / zhat * (1 - (self.__RHO(4, t) / 8 - (5 * self.__RHO(3, t) ** 2 / 24)
                                             - (self.__RHO(3, t) / (2 * zhat)) - (1 / (zhat ** 2))))
        b = (self.__RHO(3, t) / 2 + 1 / zhat) * self.__KRC_fir(t) / (zhat * self.KL_sec(t))
        c = self.__KRC_sed(t) / (2 * zhat * self.KL_sec(t))

        ans = dmean + tp * (a + b - c)

        return self.ipl * ans




    def __MC_twist_para(self, t0):
        """
        this function is used to define the twist parameters
        :param t0:
        :return:
        """

        twist = []  # the twisting terms
        for k in np.arange(0, SPA.K):
            twist.append(self.__PK(k + 1, t0))

        for l in np.arange(0, self.L):
            pl = 0
            for i in np.arange(0, SPA.K):
                pl += self.__PK(i + 1, t0) * self.gamma[l][i]
            twist.append(pl)

        # combine the theta,
        theta = self.thetak + self.thetal

        # changing the scale value
        scale = 1 / (1 -  np.array(twist))
        return theta, scale

    def MCIS(self, nsenario, nsimulate, dict_cont):
        """
        this function is to generation the set of portfolio loss for every simulation
        :param nsenario: the number of senario
        :param nsimulate: the number of simulation in each senario
        :return: the set of loss
        """

        # the twisting parameter
        t0 = fsolve(self.KL_fir_x, 1)
        # the twisting term for gamma distribution
        theta, scale = self.__MC_twist_para(t0)

        # for all senario data
        a = list(map(str, np.arange(1, nsenario + 1)))
        loss_senario_df = pan.DataFrame(columns=list(map(lambda x:"loss"+x, a))+list(map(lambda x:"lh"+x, a)))

        senario = 0

        while senario < nsenario:
            senario += 1
            if senario % 2 == 0:
                print("Congs! ----------The senario is: ", senario)

            dfc = SPA.df.copy()

            vden = dict_cont[str(senario)]["vden"]
            eden = dict_cont[str(senario)]["eden"]
            dfc["vnen"] = dict_cont[str(senario)]["vnen"]
            dfc["enen"] = dict_cont[str(senario)]["enen"]

            loss_before_set = []
            twist_set = []
            count = 0

            while count < nsimulate:

                count += 1
                if count % 100 == 0:
                    print("---haha------------The simulation is: ", count)

                # random gamma variable
                ssim = np.random.gamma(np.array(theta), np.array(scale))


                # calulate the conditional default probabilities

                for k in np.arange(0, SPA.K):
                    # the random in the lambda of possion distribution
                    weig = self.delta[k] * ssim[k]
                    for l in np.arange(0, self.L):
                        weig += np.array(self.gamma)[l][k] * ssim[SPA.K + l]

                    # conditional probability of default, that is, the lambda
                    dfc.loc[dfc.sector == k+1, 'PDS'] = dfc.PD[dfc.sector == k+1 ] * weig
                    dfc.loc[dfc.sector == k+1, "twistlambda"] = dfc.PDS[dfc.sector == k+1] * np.exp(dfc.PL[dfc.sector == k+1] * t0)

                # poisson random  parameter
                rv = np.random.poisson(lam= dfc.twistlambda)
                dfc["closs"+str(count)] = dfc.PL * rv

                # the portfolio loss in this simulation
                loss_before = sum(dfc["closs"+str(count)])
                eta = np.exp(-t0 * loss_before + self.KL(t0))


                if loss_before >= self.x0 - 0.005 and loss_before <= self.x0 + 0.005:
                    vden += eta
                    dfc["vnen"] += dfc["closs"+str(count)].map(lambda x: x * eta)
                    dfc["vnen"] = list(map(lambda x: x[0], dfc.vnen.values))


                if loss_before >= self.x0:
                    eden += eta
                    dfc["enen"] += dfc["closs" + str(count)].map(lambda x: x * eta)
                    dfc["enen"] = list(map(lambda x: x[0], dfc.enen.values))

                loss_before_set.append(loss_before)
                twist_set.append(eta[0])

            try:
                dict_cont[str(senario)]["vden"] = vden.tolist()
            except:
                dict_cont[str(senario)]["vden"] = vden

            try:
                dict_cont[str(senario)]["eden"] = eden.tolist()
            except:
                dict_cont[str(senario)]["eden"] = eden

            dict_cont[str(senario)]["vnen"] = dfc.vnen.values.tolist()
            dict_cont[str(senario)]["enen"] = dfc.enen.values.tolist()

            loss_senario_df["loss"+str(senario)] = loss_before_set
            loss_senario_df["lh"+str(senario)] = twist_set

        loss_senario_df.to_csv("mcis_loss.csv", mode='a+',header=False, index=False, sep=',')
        print("success write the portfolio loss data")



        with open('contri_data.json', 'w') as json_file:
            json.dump(dict_cont, json_file)
            print("success save the contribution data")

        return loss_senario_df, dict_cont

    def MCP(self, nsenario, nsimulate):
        """
        this function is to generation the set of portfolio loss for every simulation
        pain MC method
        :param nsenario: the number of senario
        :param nsimulate: the number of simulation in each senario
        :return: the set of loss
        """

        theta = self.thetak + self.thetal
        scale = np.ones(len(theta))

        # for all senario data
        sena = np.arange(1, nsenario+1)
        loss_senario_df = pan.DataFrame(columns=list(map(str, sena)))
        senario = 0

        while senario < nsenario:
            senario += 1
            if senario % 20 == 0:
                print("Congs! ----------The senario is: ", senario)
            count = 0

            loss_sim_set = []
            dfc = SPA.df.copy()

            while count < nsimulate:
                count += 1
                if count % 100 == 0:
                    print("---------------The simulation is: ", count)

                # random gamma variable
                ssim = np.random.gamma(np.array(theta), np.array(scale))


                # calulate the conditional default probabilities
                for k in np.arange(0, SPA.K):
                    # the random in the lambda of possion distribution
                    weig = self.delta[k] * ssim[k]
                    for l in np.arange(0, self.L):
                        weig += np.array(self.gamma)[l][k] * ssim[SPA.K + l]
                    # conditional probability of default, that is, the lambda
                    dfc.loc[dfc.sector == k + 1, 'PDS'] = dfc.PD[dfc.sector == k + 1] * weig


                # poisson random  parameter
                rv = np.random.poisson(lam=dfc.PDS)
                dfc["closs" + str(count)] = dfc.PL * rv

                # the portfolio loss in this simulation
                loss_sim = sum(dfc["closs" + str(count)])
                loss_sim_set.append(loss_sim)

            loss_senario_df[str(senario)] = loss_sim_set
        loss_senario_df.to_csv("mcp_loss.csv", mode='a+', header=False, index=False, sep=',')


        return loss_senario_df









"""
                col_name = cont_df.columns.tolist()
                col_name.insert(col_name.index("varc"+str(senario-1)) + 1, "varc"+str(senario))  # 在 B 列前面插入
                col_name.insert(col_name.index("esc" + str(senario - 1)) + 1, "esc" + str(senario ))

                cont_df.reindex(columns=col_name)

                cont_df["varc" + str(senario)] = varc
                cont_df["esc" + str(senario)] = esc
"""







