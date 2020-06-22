import numpy as np
import math

from DataPre.info_portfolio_CBV import portfolio_info,CBVmodel

import os
import sys
filename = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(filename)
import matplotlib.pyplot as plt

import pandas as pan
from scipy.optimize import fsolve

class cgf_calculation():

    def __init__(self,port_info, cbv_para):

        self.port_info = port_info
        self.df = port_info.df
        self.K = port_info.K
        self.Lmean = port_info.Lmean

        self.L = cbv_para.L
        self.delta = cbv_para.delta
        self.thetak = cbv_para.thetak
        self.thetal = cbv_para.thetal
        self.gamma = cbv_para.gamma

    # 0.5152
    def contri_df(self, i):
        """
        :param i: the index of the obligor
        :return:
        """
        self.df = self.port_info.df
        self.df = self.df.drop(self.df.index[i])
        return self.df

    def original_df(self):
        self.df = self.port_info.df
        return self.df

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

    def PL(self, l, t):
        """
        :param l: the l-th parameter
        :param t:
        :return:
        """
        ans = 0
        for i in np.arange(0, self.K):
            ans += self.PK(i + 1, t) * self.gamma[l][i]
        return ans

    def PL_drv(self, l, t, n):
        """
        :param l: the l-th parameter
        :param t: the saddlepoint
        :param n: the derivative order
        :return:
        """
        ans = 0
        for i in np.arange(0, self.K):
            ans += self.PK_drv(i + 1, t, n) * self.gamma[l][i]
        return ans


    def new_KL(self, t, i):
        exposure = self.df.PL.values[i]
        prob_default = self.df.PD.values[i]
        index_sector = self.df.sector.values[i]
        idx = index_sector -1

        pk = self.PK(index_sector, t)
        sum_pk = self.thetak[idx]*self.delta[idx]/(1-self.delta[idx]*pk)
        sum_pl = 0
        for l in np.arange(0, self.L):
            pl = self.PL(l, t)
            sum_pl += (self.thetal[l] * self.gamma[l][idx] / (1 - pl) )

        ans = t * exposure + math.log(prob_default) + math.log( sum_pk + sum_pl )

        ans = ans + self.KL(t)
        return ans

    def new_KL_fir(self, t, i):
        exposure = self.df.PL.values[i]
        index_sector = self.df.sector.values[i]
        idx = index_sector - 1

        # den
        pk = self.PK(index_sector, t)
        den_pk = self.thetak[idx] * self.delta[idx] / (1 - self.delta[idx] * pk)
        den_pl = 0
        for l in np.arange(0, self.L):
            pl = self.PL(l, t)
            den_pl += (self.thetal[l] * self.gamma[l][idx] / (1 - pl))

        # num
        pk1 = self.PK_drv(index_sector, t, 1)
        num_pk = self.thetak[idx] * (self.delta[idx])**2 * pk1 / (1 - self.delta[idx] * pk)**2
        num_pl = 0
        for l in np.arange(0, self.L):
            pl1 = self.PL_drv(l, t, 1)
            num_pl += (self.thetal[l] * self.gamma[l][idx] * pl1 / (1 - pl)**2)

        ans = exposure + ((-num_pk -num_pl)/(den_pk+den_pl)) + self.KL_fir(t)

        return ans

    def solve_new_KL_fir(self, t):
        exposure = self.df.PL.values[1]
        index_sector = self.df.sector.values[1]
        idx = index_sector - 1

        # den
        pk = self.PK(index_sector, t)
        den_pk = self.thetak[idx] * self.delta[idx] / (1 - self.delta[idx] * pk)
        den_pl = 0
        for l in np.arange(0, self.L):
            pl = self.PL(l, t)
            den_pl += (self.thetal[l] * self.gamma[l][idx] / (1 - pl))

        # num
        pk1 = self.PK_drv(index_sector, t, 1)
        num_pk = self.thetak[idx] * (self.delta[idx])**2 * pk1 / (1 - self.delta[idx] * pk)**2
        num_pl = 0
        for l in np.arange(0, self.L):
            pl1 = self.PL_drv(l, t, 1)
            num_pl += (self.thetal[l] * self.gamma[l][idx] * pl1 / (1 - pl)**2)

        ans = exposure - ((num_pk + num_pl)/(den_pk+den_pl)) + self.KL_fir(t)

        return ans-0.4

    def new_KL_sec(self, t, i):
        index_sector = self.df.sector.values[i]
        idx = index_sector - 1

        # den
        pk = self.PK(index_sector, t)
        den_pk = self.thetak[idx] * self.delta[idx] / (1 - self.delta[idx] * pk)
        den_pl = 0
        for l in np.arange(0, self.L):
            pl = self.PL(l, t)
            den_pl += (self.thetal[l] * self.gamma[l][idx] / (1 - pl))
        den_sum = den_pl + den_pk

        # num
        pk1 = self.PK_drv(index_sector, t, 1)
        num1_pk = self.thetak[idx] * (self.delta[idx]) ** 2 * pk1 / ((1 - self.delta[idx] * pk) ** 2)
        num1_pl = 0
        for l in np.arange(0, self.L):
            pl1 = self.PL_drv( l, t, 1)
            num1_pl += (self.thetal[l] * self.gamma[l][idx] * pl1 / (1 - pl) ** 2)
        num_sum = num1_pl + num1_pk

        pk2 = self.PK_drv(index_sector, t, 2)
        num2_pk = self.thetak[idx] * (self.delta[idx]**2) * (pk2 - self.delta[idx]*pk*pk2+2*self.delta[idx]*(pk1**2)) / (1 - self.delta[idx] * pk) ** 3
        num2_pl = 0
        for l in np.arange(0, self.L):
            pl2 = self.PL_drv(l, t, 2)
            num2_pl += ( self.thetal[l] * self.gamma[l][idx] * (pl2 - pl*pl2 + 2*(pl1**2)) / (1 - pl) ** 3 )

        ans = -(num2_pl+num2_pk)/den_sum - (num_sum/den_sum)**2 + self.KL_sec(t)
        return ans


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

    def changeK_KL(self, t, idx_sector):
        """
        this function is to calculate the risk contribution, for sector
        :param t: saddlepoint
        :param sing_sector: the idx of sector for this special obligor
        :return: KL value
        """
        ans = 0
        # sum of all sectors
        for i in np.arange(0, self.K):
            pk = self.PK(i + 1, t)
            # add 1 for this special sector
            if i == idx_sector:
                temp = - ( self.thetak[i] + 1 ) * math.log(1 - self.delta[i] * pk)
            else:
                temp = - self.thetak[i] * math.log(1 - self.delta[i] * pk)
            ans = ans + temp

        for l in np.arange(0, self.L):
            pl = 0
            for i in np.arange(0, self.K):
                pl += self.PK(i + 1, t) * self.gamma[l][i]
            try:
                temp_ = - ( self.thetal[l] ) * math.log(1 - pl)
            except:
                print("the sqrt term in w_hat is invalid")
                ans = np.nan
            else:
                ans = ans + temp_
        return ans

    def changeL_KL(self, t, idx_common):
        """
        this function is to calculate the risk contribution, for common factor
        :param t: saddlepoint
        :param l: the idx of common sector
        :return: KL value
        """
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
                # add 1 for this special common factor
                if l == idx_common:
                    temp_ = - ( self.thetal[l] + 1) * math.log(1 - pl)
                else:
                    temp_ = - self.thetal[l]  * math.log(1 - pl)
            except:
                print("the sqrt term in w_hat is invalid")
                ans = np.nan
            else:
                ans = ans + temp_
        return ans


    def KL_drv(self, t):

        temp1 = t * self.df.PD.values * np.exp(t * self.df.PL.values)
        temp2 = [(self.delta[w - 1] * self.thetak[w - 1]) / (1 - self.delta[w - 1] * self.PK(w, t)) for w in
                 self.df.sector.values]
        temp3 = np.zeros(len(temp1))
        for l in np.arange(0, self.L):
            pl = 0
            for i in np.arange(0, self.K):
                pl += self.PK(i + 1, t) * self.gamma[l][i]
            temp3 += np.array([self.thetal[l] * self.gamma[l][w - 1] / (1 - pl) for w in self.df.sector.values])
        ans = temp1 * (np.array(temp2) + temp3)

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

    def changeK_KL_fir(self, t, idx_sector):
        ans = 0
        # sum of all sectors
        for i in np.arange(0, self.K):
            pk = self.PK(i + 1, t)
            pk1 = self.PK_drv(i + 1, t, 1)
            # add 1 for this special sector
            if i == idx_sector:
                temp = self.delta[i] * (self.thetak[i] +1 ) * pk1 / (1 - self.delta[i] * pk)
            else:
                temp = self.delta[i] * self.thetak[i] * pk1 / (1 - self.delta[i] * pk)
            ans += temp

        for l in np.arange(0, self.L):
            pl = 0
            pl1 = 0
            for i in np.arange(0, self.K):
                pl += self.PK(i + 1, t) * self.gamma[l][i]
                pl1 += self.PK_drv(i + 1, t, 1) * self.gamma[l][i]
            temp_ = self.thetal[l]  * pl1 / (1 - pl)
            ans += temp_
        return ans

    def changeL_KL_fir(self, t, idx_common):
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
            # add 1 for this special common factor
            if l == idx_common:
                temp_ = (self.thetal[l] + 1) * pl1 / (1 - pl)
            else:
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

    def changeK_KL_sec(self, t, idx_sector):
        ans = 0

        # sum of all sectors
        for i in np.arange(0, self.K):
            pk = self.PK(i + 1, t)
            pk1 = self.PK_drv(i + 1, t, 1)
            pk2 = self.PK_drv(i + 1, t, 2)
            Nk1 = pk1 / (1 - self.delta[i] * pk)
            Nk2 = pk2 / (1 - self.delta[i] * pk)
            # add 1 for this special sector
            if i == idx_sector:
                temp = self.delta[i] * (self.thetak[i]+1) * (Nk2 + self.delta[i] * (Nk1 ** 2))
            else:
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
            temp_ = self.thetal[l]  * (Nl2 + (Nl1 ** 2))
            ans += temp_

        return ans

    def changeL_KL_sec(self, t, idx_common):
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
            # add 1 for all L
            if l == idx_common:
                temp_ = (self.thetal[l] +1) * (Nl2 + (Nl1 ** 2))
            else:
                temp_ = self.thetal[l] * (Nl2 + (Nl1 ** 2))
            ans += temp_

        return ans

    def KL_sec_drv(self, t):

        ploss = self.df.PL.values

        temp1 = []
        Num = 0
        for w in self.df.sector.values:
            tempa =  (2*ploss[Num] + t*(ploss[Num]**2))/(1-self.delta[w-1]*self.PK(w, t))
            tempb = self.delta[w-1]* (t*self.PK_drv(w,t,2) + 2*(1+t*ploss[Num])*self.PK_drv(w,t,1))/((1-self.delta[w-1]*self.PK(w, t))**2)
            tempc =  2*t*((self.delta[w-1]*self.PK_drv(w,t,1))**2)/((1-self.delta[w-1]*self.PK(w, t))**3)
            temp = (self.delta[w - 1] * self.thetak[w - 1]) *(tempa + tempb + tempc)
            temp1.append(temp[0])
            Num += 1

        temp2 = np.zeros(len(temp1))
        for l in np.arange(0, self.L):
            tempa = (2 * ploss + t * (ploss ** 2)) / (1 - self.PL(l, t))
            tempb = (t * self.PL_drv(l, t, 2) + 2 * (1 + t * ploss) * self.PL_drv(l, t, 1)) / ((1 - self.PL(l, t)) ** 2)
            tempc = ((2 * t * self.PL_drv(l, t, 1)) ** 2) / ((1 - self.PL(l, t)) ** 3)

            temp = np.array([(self.thetal[l] * self.gamma[l][w - 1]) for w in self.df.sector.values ])
            temp2 += temp * (tempa + tempb + tempc)

        ans = self.df.PD.values * np.exp(t * ploss) * (np.array(temp1) + temp2 )

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

    def changeK_KL_thi(self, t, idx_sector):
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

            # add 1 for this special sector
            if i == idx_sector:
                temp = self.delta[i] * (self.thetak[i] +1 )* (
                        Nk3 + 3 * self.delta[i] * Nk1 * Nk2 + 2 * (self.delta[i] ** 2) * (Nk1 ** 3))
            else:
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

    def changeL_KL_thi(self, t, idx_common):
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
            # add 1 for this special L
            if l == idx_common:
                temp_ = (self.thetal[l]+1) * (Nl3 + 3 * Nl1 * Nl2 + 2 * Nl1 ** 3)
            else:
                temp_ = self.thetal[l]  * (Nl3 + 3 * Nl1 * Nl2 + 2 * Nl1 ** 3)
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

    def changeK_KL_for(self, t, idx_sector):
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

            # add 1 for this special sector
            if i == idx_sector :
                temp = self.delta[i] * (self.thetak[i] +1 ) * (
                        Nk4 + 3 * self.delta[i] * Nk2 ** 2 + 4 * self.delta[i] * Nk3 * Nk1
                        + 12 * (self.delta[i] ** 2) * (Nk1 ** 2) * Nk2 + 6 * (self.delta[i] ** 3) * (
                                    Nk1 ** 4))
            else:
                temp = self.delta[i] * self.thetak[i] * (
                            Nk4 + 3 * self.delta[i] * Nk2 ** 2 + 4 * self.delta[i] * Nk3 * Nk1
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

            temp_ = (self.thetal[l] ) * (Nl4 + 3 * Nl2 ** 2 + 4 * Nl3 * Nl1 + 12 * (Nl1 ** 2) * Nl2 + 6 * (Nl1 ** 4))
            ans = ans + temp_

        return ans

    def changeL_KL_for(self, t, idx_common):
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
            temp = self.delta[i] * self.thetak[i] * (
                            Nk4 + 3 * self.delta[i] * Nk2 ** 2 + 4 * self.delta[i] * Nk3 * Nk1
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
            # add 1 for this special sector
            if l == idx_common:
                temp_ = (self.thetal[l] +1 ) * (Nl4 + 3 * Nl2 ** 2 + 4 * Nl3 * Nl1 + 12 * (Nl1 ** 2) * Nl2 + 6 * (Nl1 ** 4))
            else:
                temp_ = (self.thetal[l] ) * (
                            Nl4 + 3 * Nl2 ** 2 + 4 * Nl3 * Nl1 + 12 * (Nl1 ** 2) * Nl2 + 6 * (Nl1 ** 4))
            ans = ans + temp_

        return ans



    def skewness(self):
        ans = self.KL_thi(0) / (self.KL_sec(0))**1.5
        return ans

    def kurtosis(self):
        ans = self.KL_for(0) / (self.KL_sec(0))**2 + 3
        return ans

if __name__ == '__main__':

    pf = portfolio_info()
    pf.init_obligor()
    df = pf.df

    cbvpara = CBVmodel()
    cbvpara.CBV3()

    coca = cgf_calculation(pf, cbvpara)
    skew_value = coca.skewness()
    kurt_value = coca.kurtosis()
    coca.KL_sec(0)

    root = coca.QL_root()

    ans_set1 = []
    ans_set2 = []
    spt_set = np.arange(3, 5, 0.05)
    i = 92
    for spt in spt_set:
        ans_set1.append(coca.new_KL_fir(spt, i))
        ans_set2.append(coca.new_KL_sec(spt, i))

    result = pan.DataFrame()
    result["spt"] = spt_set
    result["fir"] = ans_set1
    result["sed"] = ans_set2


    plt.figure(2, figsize=(20, 15))
    # plt.yscale("log")
    #plt.plot(spt_set, ans_set1, 'o-', label='first derivative')
    plt.plot(result[result["sed"]<500].spt, result[result["sed"]<500].sed, 'o-', label='second derivative')
    plt.plot(result.spt, result.fir, '--', label='first derivative')
   # plt.plot(spt_set, ans_set2, label="second derivative")

    plt.xlabel('Saddlepoint', fontsize=30)
    plt.ylabel('Value', fontsize=30)
    plt.grid(which='both', linestyle=':', linewidth=1)
    plt.legend(fontsize=30)
    plt.savefig("test.png")
    # plt.title("Example 1",fontsize = 40)
    plt.show()









