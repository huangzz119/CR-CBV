from DataPre.info_portfolio_CBV import portfolio_info,CBVmodel
import os
import sys
filename = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(filename)

from scipy.stats import norm
from scipy.optimize import fsolve,minimize
from scipy import special

import numpy as np


class heavytail_case1():

    def __init__(self,port_info, cbv_para, level ):

        self.df = port_info.df
        self.K = port_info.K
        self.Lmean = port_info.Lmean

        self.L = cbv_para.L
        self.delta = cbv_para.delta
        self.thetak = cbv_para.thetak
        self.thetal = cbv_para.thetal
        self.gamma = cbv_para.gamma

        self.level = level

        # Hk_calculate
        Hkk = []
        for k in np.arange(0, self.K):
            st = self.df.loc[self.df["sector"] == k+1]
            hk = sum(st.PD * st.PL)
            Hkk.append(hk)
        self.Hk = Hkk

        # Hl_calculate
        Hll = []
        for l in np.arange(0, self.L):
            hl = 0
            for k in np.arange(0, self.K):
                hl += self.Hk[k] * self.gamma[l][k]
            Hll.append(hl)
        self.Hl = Hll

        # parameters for the approximation of sum of gamma distribution
        mean = sum(np.array(self.thetak) * np.array(self.delta) * np.array(self.Hk)) + sum(np.array(self.thetal) * np.array(self.Hl))
        variance = sum(np.array(self.thetak) * (np.array(self.delta)**2) * (np.array(self.Hk)**2)) + sum(np.array(self.thetal) * (np.array(self.Hl)**2))

        self.p = mean**2 / (variance)
        self.y = variance / mean



    def kXX(self, t1, t2):
        ans = np.exp(t1) - 1

        for k in np.arange(0, self.K):
            temp = - self.thetak[k] * np.log( 1 - self.delta[k] * self.Hk[k] * t2)
            ans += temp

        for l in np.arange(0, self.L):
            temp = - self.thetal[l] * np.log( 1- self.Hl[l] * t2)
            ans += temp
        return ans



    def kXX_gradient(self, t1, t2):
        kXX1 = np.exp(t1)
        ans = 0
        for k in np.arange(0, self.K):
            temp = ( self.thetak[k] * self.delta[k] * self.Hk[k]) /( 1 - self.delta[k]*self.Hk[k]*t2)
            ans += temp
        for l in np.arange(0, self.L):
            temp = (self.thetal[l]*self.Hl[l])/(1-self.Hl[l]*t2)
            ans += temp
        kXX2 = ans
        return [kXX1,kXX2]


    def kXX_Hessian(self, t1, t2):
        kXX11 = np.exp(t1)
        ans = 0
        for k in np.arange(0, self.K):
            temp = (self.thetak[k] * (self.delta[k]**2) * (self.Hk[k]**2)) / (1 - self.delta[k] * self.Hk[k] * t2)**2
            ans += temp
        for l in np.arange(0, self.L):
            temp = (self.thetal[l] * (self.Hl[l]**2)) / (1-self.Hl[l]*t2)**2
            ans += temp
        kXX22 = ans
        return np.array([[kXX11[0], 0],[0, kXX22[0]]])


    def Jacobian(self, y1, y2):
        return np.array([[1/y2[0], -y1* (y2**(-2))[0]],[0, 1]])


    def ES_approx_heavytail(self, c):

        def kXX_approx(t1, t2):
            return np.exp(t1) - 1 + np.log(special.gamma(self.p + t2)) - np.log(special.gamma(self.p)) + t2 * np.log(
                self.y)

        def kXX_approx_Hessian(t1, t2):
            kXX11 = np.exp(t1)
            kXX22 = special.polygamma(1, self.p + t2)
            return np.array([[kXX11, 0], [0, kXX22]])

        def g_function_solve_y2(y2):
            t2 = c * (y2 ** (-2)) * np.log(c / y2)
            ans = np.log(self.y) + special.digamma(self.p + t2) - y2
            return ans

        y2 = fsolve(g_function_solve_y2, 1.0)
        t1 = np.log(c / y2)
        t2 = c * (y2 ** (-2)) * np.log(c / y2)

        # the Lmean
        alpha0 = np.exp(  np.log(self.y) + special.digamma(self.p) )
        alpha1 = special.polygamma(1, self.p ) * np.exp( np.log(self.y) + special.digamma(self.p) )

        wc = np.sign((c - alpha0)) * np.sqrt(2 * (t1 * (c / y2) + t2 * y2 - kXX_approx(t1, t2)))

        hessian = kXX_approx_Hessian(t1, t2).astype(float)
        invhessian = np.linalg.inv(hessian)
        jacobian = self.Jacobian(c, y2).astype(float)
        gy2 = jacobian[:, 1]
        gy1 = jacobian[:, 0]
        bmt = np.array([t1[0], t2[0]])

        kc = (np.linalg.det(hessian) / (np.linalg.det(jacobian) ** 2)) * (
                np.dot(np.dot(gy2, invhessian), gy2) + np.dot(bmt, gy2 ** 2))

        uc = np.dot(bmt, gy1) * np.sqrt(kc)

        temp1 = (1 - norm.cdf(wc)) * (alpha0+alpha1 - c)

        temp2 = norm.pdf(wc) * ((alpha0+alpha1 - c) / wc - (alpha0 - c) / (wc ** 3)
                                - 1 / (np.dot(bmt, gy1) * uc))

        ans = c + 1 / self.level * (temp1 - temp2)

        return ans


    def ES_heavytail(self, c):

        # solve out the solution
        def g_function_solve_y2(y2):
            t2 = c * (y2 ** (-2)) * np.log(c / y2)

            temp1 = np.array(self.thetak) * np.array(self.delta) * np.array(self.Hk) / (
                    1 - np.array(self.delta) * np.array(self.Hk) * t2)
            temp2 = np.array(self.thetal) * np.array(self.Hl) / (1 - np.array(self.Hl) * t2)

            ans = sum(temp1) + sum(temp2) - y2

            return ans

        y2 = fsolve(g_function_solve_y2, 1.36)
        t1 = np.log(c / y2)
        t2 = c * (y2 ** (-2)) * np.log(c / y2)

        temp1_alpha0 = np.array(self.thetak) * np.array(self.delta) * np.array(self.Hk)
        temp2_alpha0 = np.array(self.thetal) * np.array(self.Hl)

        # the Lmean
        alpha0 = sum(temp1_alpha0) + sum(temp2_alpha0)

        alpha1 = 0

        wc =np.sign((c - alpha0)) * np.sqrt(2*( t1*(c/y2) + t2*y2 - self.kXX(t1,t2)))

        hessian = self.kXX_Hessian(t1, t2).astype(float)
        invhessian = np.linalg.inv(hessian)
        jacobian = self.Jacobian(c , y2).astype(float)
        gy2 = jacobian[:, 1]
        gy1 = jacobian[:, 0]
        bmt = np.array([t1[0], t2[0]])

        # the product of matrix
        kc = ( np.linalg.det(hessian)/ (np.linalg.det(jacobian)**2)) * (
                np.dot(np.dot(gy2, invhessian),gy2) + np.dot(bmt, gy2**2) )

        uc = np.dot(bmt, gy1) * np.sqrt(kc)

        temp1 = (1 - norm.cdf(wc))*( self.Lmean - c)

        temp2 = norm.pdf(wc) * ( ( self.Lmean - c )/wc - (alpha0 - c)/(wc**3)
                                 - 1/( np.dot(bmt,gy1)*uc ))

        ans = c + 1/self.level*(temp1-temp2)

        return ans


if __name__ == '__main__':


    pf = portfolio_info()
    pf.init_obligor()


    cbvpara = CBVmodel()
    cbvpara.CBV3()

    ht = heavytail_case1(pf, cbvpara, 0.05)

   # ans = ht.ES_heavytail(3.27)

    init = np.array(1.6, dtype="float")

    ans = ht.ES_approx_heavytail(2.2)

    minimize(ht.ES_heavytail, init, method='Nelder-Mead')

    minimize(ht.ES_approx_heavytail, init, method='Nelder-Mead')












