import pandas as pan
import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import fsolve

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
    df = df_origin.iloc[0:df_origin.shape[0], [2, 7, 8, 18, 22, 19, 15, 6]]
    df.columns = ["obligor", "PD", "SD(PD)", "EAD", "LGD", "PL", "sector", "rating"]
    df["PL"] = df["PL"].map(lambda x: x / 1000000000)
    df["EAD"] = df["EAD"].map(lambda x: x / 1000000)
    df["EL"] = df.apply(lambda x: x["PD"] * x["PL"], axis=1)

    K=10  # sector number
    Lmean = sum(df.EL)  # the mean of expected loss

    def __init__(self, L, delta, thetak, gamma, thetal):
        self.L = L
        self.delta = delta
        self.thetak = thetak
        self.gamma = gamma
        self.thetal = thetal

    #the minimum upper boundary of saddlepoint
    def up_spt(self):
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

    # the root inside log as the minimum upper boundary
    def QL_root(self):
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

    # CGF
    def KL(self, t):
        ans = 0

        # sum of all sectors
        for i in np.arange(0, SPA.K):
            st = SPA.df.loc[ SPA.df["sector"] == i + 1]  # sectors
            Qk = sum(st.PD * (np.exp(t * st.PL) - 1))
            temp = - self.thetak[i] * math.log(1 - self.delta[i] * Qk)
            ans = ans + temp

        for l in np.arange(0, self.L):
            Ql = 0
            for i in np.arange(0, SPA.K):
                st = SPA.df.loc[ SPA.df["sector"] == i + 1]
                tempQl = sum(st.PD * (np.exp(t * st.PL) - 1)) * self.gamma[l][i]
                Ql = Ql + tempQl
            # add the common factor
            temp_ = - self.thetal[l] * math.log(1 - Ql)
            ans = ans + temp_

        return ans

    # CGF first derivative
    def KL_fir(self, t):
        ans = 0

        # sum of all sectors
        for i in np.arange(0, SPA.K):
            st = SPA.df.loc[ SPA.df["sector"] == i + 1]
            Qk = sum(st.PD * (np.exp(t * st.PL) - 1))
            Qk1 = sum(st.PL * st.PD * (np.exp(t * st.PL)))
            temp = self.delta[i] * self.thetak[i] * Qk1 / (1 - self.delta[i] * Qk)
            ans = ans + temp

        for l in np.arange(0, self.L):
            Ql = 0
            Ql1 = 0
            for i in np.arange(0, SPA.K):
                st = SPA.df.loc[ SPA.df["sector"] == i + 1]
                tempQl = sum(st.PD * (np.exp(t * st.PL) - 1)) * self.gamma[l][i]
                tempQl1 = sum(st.PL * st.PD * np.exp(t * st.PL)) * self.gamma[l][i]
                Ql = Ql + tempQl
                Ql1 = Ql1 + tempQl1
            # add the common factor
            temp_ = self.thetal[l] * Ql1 / (1 - Ql)
            ans = ans + temp_

        return ans

    # CGF second derivative
    def KL_sec(self, t):
        ans = 0

        # sum of all sectors
        for i in np.arange(0, SPA.K):
            st = SPA.df.loc[ SPA.df["sector"] == i + 1]
            Qk = sum(st.PD * (np.exp(t * st.PL) - 1))
            Qk1 = sum(st.PL * st.PD * (np.exp(t * st.PL)))
            Qk2 = sum(st.PL ** 2 * st.PD * (np.exp(t * st.PL)))
            Nk1 = Qk1 / (1 - self.delta[i] * Qk)
            Nk2 = Qk2 / (1 - self.delta[i] * Qk)
            temp = self.delta[i] * self.thetak[i] * (Nk2 + self.delta[i] * (Nk1 ** 2))
            ans = ans + temp

        for l in np.arange(0, self.L):
            Ql = 0
            Ql1 = 0
            Ql2 = 0
            for i in np.arange(0, SPA.K):
                st = SPA.df.loc[ SPA.df["sector"] == i + 1]
                tempQl = sum(st.PD * (np.exp(t * st.PL) - 1)) * self.gamma[l][i]
                tempQl1 = sum(st.PL * st.PD * np.exp(t * st.PL)) * self.gamma[l][i]
                tempQl2 = sum((st.PL ** 2) * st.PD * np.exp(t * st.PL)) * self.gamma[l][i]
                Ql = Ql + tempQl
                Ql1 = Ql1 + tempQl1
                Ql2 = Ql2 + tempQl2
            Nl1 = Ql1 / (1 - Ql)
            Nl2 = Ql2 / (1 - Ql)
            temp_ = self.thetal[l] * (Nl2 + (Nl1 ** 2))
            ans = ans + temp_

        return ans

    # the CDF
    def CDF(self, t, loss):
        what = np.sign(t) * np.sqrt(2 * (t * loss - self.KL(t)))
        uhat = t * np.sqrt(self.KL_sec(t))
        ans = norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat)
        return ans

    #




