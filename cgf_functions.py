import numpy as np
import math

from DataPre.info_portfolio_CBV import portfolio_info,CBVmodel

import os
import sys
filename = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(filename)

from scipy.optimize import fsolve

class cgf_calculation():

    def __init__(self,port_info, cbv_para):

        self.df = port_info.df
        self.K = port_info.K
        self.Lmean = port_info.Lmean

        self.L = cbv_para.L
        self.delta = cbv_para.delta
        self.thetak = cbv_para.thetak
        self.thetal = cbv_para.thetal
        self.gamma = cbv_para.gamma

    # 0.5152
    def QL_root(self):
        """
        :return: the root inside log as the minimum upper boundary
        """
        root_set = []
        for l in np.arange(0, self.L):
            def eff(t):
                Ql = 0
                for i in np.arange(0, self.K):
                    Ql += self.PK(i + 1, t) * self.gamma[l][i]
                ans = 1 - Ql
                return ans
            root_ = fsolve(eff, 0.5)
            root_set.append(root_)

        min_root = min(root_set)
        return min_root

    def PK(self, k, t):
        """
        :param k: k-th sector
        :param t: saddlepoint
        :return: Qk value
        """
        st = self.df.loc[self.df["sector"] == k]
        pk = sum(st.PD * (np.exp(t * st.PL) - 1))
        return pk

    def PK_drv(self, k, t, n):
        """
        :param n: the order of derivative
        :return:
        """
        st = self.df.loc[self.df["sector"] == k]
        pk_drv = sum((st.PL ** n) * st.PD * (np.exp(t * st.PL)))
        return pk_drv

    def KL(self, t):
        ans = 0
        # sum of all sectors
        for i in np.arange(0, self.K):
            pk = self.PK(i + 1, t)
            temp = - self.thetak[i] * math.log(1 - self.delta[i] * pk)
            ans = ans + temp

        for l in np.arange(0, self.L):
            pl = 0
            for i in np.arange(0, self.K):
                pl += self.PK(i + 1, t) * self.gamma[l][i]
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
        for i in np.arange(0, self.K):
            pk = self.PK(i + 1, t)
            pk1 = self.PK_drv(i + 1, t, 1)
            temp = self.delta[i] * self.thetak[i] * pk1 / (1 - self.delta[i] * pk)
            ans += temp

        for l in np.arange(0, self.L):
            pl = 0
            pl1 = 0
            for i in np.arange(0, self.K):
                pl += self.PK(i + 1, t) * self.gamma[l][i]
                pl1 += self.PK_drv(i + 1, t, 1) * self.gamma[l][i]
            temp_ = self.thetal[l] * pl1 / (1 - pl)
            ans += temp_

        return ans

    def KL_sec(self, t):
        ans = 0

        # sum of all sectors
        for i in np.arange(0, self.K):
            pk = self.PK(i + 1, t)
            pk1 = self.PK_drv(i + 1, t, 1)
            pk2 = self.PK_drv(i + 1, t, 2)
            Nk1 = pk1 / (1 - self.delta[i] * pk)
            Nk2 = pk2 / (1 - self.delta[i] * pk)
            temp = self.delta[i] * self.thetak[i] * (Nk2 + self.delta[i] * (Nk1 ** 2))
            ans += temp

        for l in np.arange(0, self.L):
            pl = 0
            pl1 = 0
            pl2 = 0
            for i in np.arange(0, self.K):
                pl += self.PK(i + 1, t) * self.gamma[l][i]
                pl1 += self.PK_drv(i + 1, t, 1) * self.gamma[l][i]
                pl2 += self.PK_drv(i + 1, t, 2) * self.gamma[l][i]
            Nl1 = pl1 / (1 - pl)
            Nl2 = pl2 / (1 - pl)
            temp_ = self.thetal[l] * (Nl2 + (Nl1 ** 2))
            ans += temp_

        return ans

    def KL_thi(self, t):
        ans = 0

        # sum of all sectors
        for i in np.arange(0, self.K):
            pk = self.PK(i + 1, t)
            pk1 = self.PK_drv(i + 1, t, 1)
            pk2 = self.PK_drv(i + 1, t, 2)
            pk3 = self.PK_drv(i + 1, t, 3)
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
            for i in np.arange(0, self.K):
                pl += self.PK(i + 1, t) * self.gamma[l][i]
                pl1 += self.PK_drv(i + 1, t, 1) * self.gamma[l][i]
                pl2 += self.PK_drv(i + 1, t, 2) * self.gamma[l][i]
                pl3 += self.PK_drv(i + 1, t, 3) * self.gamma[l][i]
            Nl1 = pl1 / (1 - pl)
            Nl2 = pl2 / (1 - pl)
            Nl3 = pl3 / (1 - pl)
            temp_ = self.thetal[l] * (Nl3 + 3 * Nl1 * Nl2 + 2 * Nl1 ** 3)
            ans = ans + temp_
        return ans

    def KL_for(self, t):
        ans = 0

        # sum of all sectors
        for i in np.arange(0, self.K):
            pk = self.PK(i + 1, t)
            pk1 = self.PK_drv(i + 1, t, 1)
            pk2 = self.PK_drv(i + 1, t, 2)
            pk3 = self.PK_drv(i + 1, t, 3)
            pk4 = self.PK_drv(i + 1, t, 4)
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
            for i in np.arange(0, self.K):
                pl += self.PK(i + 1, t) * self.gamma[l][i]
                pl1 += self.PK_drv(i + 1, t, 1) * self.gamma[l][i]
                pl2 += self.PK_drv(i + 1, t, 2) * self.gamma[l][i]
                pl3 += self.PK_drv(i + 1, t, 3) * self.gamma[l][i]
                pl4 += self.PK_drv(i + 1, t, 4) * self.gamma[l][i]
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


