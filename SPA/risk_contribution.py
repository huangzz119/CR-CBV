from scipy.optimize import fsolve
import numpy as np
from cgf_functions import cgf_calculation
from DataPre.info_portfolio_CBV import portfolio_info,CBVmodel
from scipy.stats import norm
import matplotlib.pyplot as plt
import math

class Kimfunction():
    def __init__(self, coca):

        self.coca = coca

        self.df = coca.df
        self.K = coca.K
        self.Lmean = coca.Lmean

        self.L = coca.L
        self.delta = coca.delta
        self.thetak = coca.thetak
        self.thetal = coca.thetal
        self.gamma = coca.gamma


    def RHO(self, r, t):
        """
        :param r: the order of derivative
        :return: standardized cumulant of order r
        """
        if r == 3:
            ans = self.coca.KL_thi(t) / self.coca.KL_sec(t) ** (3 / 2)
        if r == 4:
            ans = self.coca.KL_for(t) / self.coca.KL_sec(t) ** 2
        return ans


    def KRC(self, ik, ipd, ipl, t):
        """
        this function is to generating the multivarite CGF of portfolio L and obligor i
        :param ik:  the sector where obligor i belongs to
        :param ipd:  the probability of default of obligor i
        :param ipl: the potential loss of obligor i
        :return: the answer of KRC
        """
        idx = ik - 1

        den1 = 1 - self.delta[idx] * self.coca.PK(ik, t)
        num1 = self.delta[idx] * self.thetak[idx]
        ins = num1 / den1

        for l in np.arange(0, self.L):
            pl = 0
            for i in np.arange(0, self.K):
                pl += self.coca.PK(i + 1, t) * self.gamma[l][i]
            ins += (self.thetal[l] * self.gamma[l][idx]) / (1 - pl)

        ans = ipd * np.exp(t * ipl) * ins

        return ans

    def KRC_fir(self, ik, ipd, ipl, t):

        idx = ik - 1

        a = ipl / (1 - self.delta[idx] * self.coca.PK(ik, t))
        b = self.delta[idx] * self.coca.PK_drv(ik, t, 1) / (1 - self.delta[idx] * self.coca.PK(ik, t)) ** 2
        ins = self.thetak[idx] * self.delta[idx] * (a + b)

        for l in np.arange(0,self.L):
            pl = 0
            pl1 = 0
            for i in np.arange(0, self.K):
                pl += self.coca.PK(i + 1, t) * self.gamma[l][i]
                pl1 += self.coca.PK_drv(i + 1, t, 1) * self.gamma[l][i]
            ins += self.thetal[l] * self.gamma[l][idx] * (ipl / (1 - pl) + pl1 / ((1 - pl) ** 2))

        ans = ipd * np.exp(t * ipl) * ins
        return ans

    def KRC_sed(self, ik, ipl, ipd, t):
        pk = self.coca.PK( ik, t )
        pk1 = self.coca.PK_drv( ik, t, 1 )
        pk2 = self.coca.PK_drv( ik, t, 2 )

        idx = ik -1

        a1 = (ipl ** 2 + ipl * self.delta[idx] * pk1) / (1 - self.delta[idx] * pk)
        a2 = (ipl * self.delta[idx] * pk1 + self.delta[idx] * pk2) / (1 - self.delta[idx] * pk) ** 2
        a3 = (2 * (self.delta[idx] ** 2) * (pk1 ** 2)) / (1 - self.delta[idx] * pk) ** 3
        ins = self.thetak[idx] * self.delta[idx] * (a1 + a2 + a3)

        for l in np.arange(0, self.L):
            pl = 0
            pl1 = 0
            pl2 = 0
            for i in np.arange(0, self.K):
                pl += self.coca.PK(i + 1, t) * self.gamma[l][i]
                pl1 += self.coca.PK_drv(i + 1, t, 1) * self.gamma[l][i]
                pl1 += self.coca.PK_drv(i + 1, t, 2) * self.gamma[l][i]
            b1 = (ipl ** 2 + ipl * pl1) / (1 - pl)
            b2 = (ipl * pl1 + pl2) / (1 - pl) ** 2
            b3 = (2 * pl1 ** 2) / (1 - pl) ** 3
            ins += self.thetal[l] * self.gamma[l][idx] * (b1 + b2 + b3)

        ans = ipd * np.exp(t * ipl) * ins
        return ans


    def DMEAN(self, ik, ipd):
        idx = ik -1
        ans = ipd * self.delta[idx] * self.thetak[idx]
        for l in np.arange(0, self.L):
            ans += ipd * self.gamma[l][idx] * self.thetal[l]
        return ans

    def CDF(self, x, t):
        """
        :return: the cumulative probability
        """
        uhat = t * np.sqrt(self.coca.KL_sec(t))
        try:
            what = np.sign(t) * np.sqrt(2 * (t * x - self.coca.KL(t)))
        except:
            print("the sqrt term in what is invalid")
            ans = np.nan
        else:
            ans = norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat)

        return ans



class RCcalculation():

    def __init__(self, kimca, est_spt= 2.789, x0 = 0.445):

        self.coca = kimca.coca
        self.kimca = kimca

        self.df = kimca.df
        self.K = kimca.K
        self.Lmean = kimca.Lmean

        self.L = kimca.L
        self.delta = kimca.delta
        self.thetak = kimca.thetak
        self.thetal = kimca.thetal
        self.gamma = kimca.gamma

        # estimator the value
        self.est_spt = est_spt
        self.x0 = x0

    def __KL_fir_x(self, t):
        """
        this function is for given an x0, solve out the corresponding saddlepoint t.
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

        return ans - self.x0

    def PDF(self):

        spt = fsolve(self.__KL_fir_x, self.est_spt)
        rho3 = self.coca.KL_thi(spt) / (self.coca.KL_sec(spt)**1.5)
        rho4 = self.coca.KL_for(spt) / (self.coca.KL_sec(spt)**2)
        ans = np.exp(self.coca.KL(spt) - spt*self.x0)/np.sqrt(2*math.pi*self.coca.KL_sec(spt))*(1+1/8*(rho4-5/3*rho3**2))

        return ans



    def VARC_MARTIN(self):

        spt = fsolve(self.__KL_fir_x, self.est_spt)
        KLdrv = self.coca.KL_drv(spt)
        ans = self.df.PL.values / spt * KLdrv
        return ans

    def VARC_diff(self):

        spt = fsolve(self.__KL_fir_x, self.est_spt)
        KLdrv = self.coca.KL_drv(spt)
        KLdrv2 = self.coca.KL_sec_drv(spt)

        uhat = spt * np.sqrt(self.coca.KL_sec(spt))
        what = np.sign(spt) * np.sqrt(2 * (spt * self.x0 - self.coca.KL(spt)))

        cdf_drv = - norm.pdf(what) * ( KLdrv * (1/what**3 - 1/uhat) + 0.5* (KLdrv2 / (uhat*self.coca.KL_sec(spt))) )
        pdf = self.PDF()
        ans = cdf_drv/pdf * self.df.PL.values

        return ans

    def ESC_diff(self):

        spt = fsolve(self.__KL_fir_x, self.est_spt)
        KLdrv = self.coca.KL_drv(spt)
        KLdrv2 = self.coca.KL_sec_drv(spt)

        uhat = spt * np.sqrt(self.coca.KL_sec(spt))
        what = np.sign(spt) * np.sqrt(2 * (spt * self.x0 - self.coca.KL(spt)))

        ans = - norm.pdf(what) * (KLdrv * ( - self.Lmean / what ** 3 + self.x0 / uhat)
                                      - self.x0/2 * (KLdrv2 / (uhat * self.coca.KL_sec(spt))))

        return ans* self.df.PL.values




    def VARC_KIM(self):
        """
        :param t: the saddlepoint corresponding to loss x0
        :return: the VaRC of obligor i
        """
        spt = fsolve(self.__KL_fir_x, self.est_spt)

        ans_set = []

        for i in np.arange(0, len(self.df)):
            ik = self.df.sector[i]
            ipd = self.df.PD[i]
            ipl = self.df.PL[i]

            KRCfir =  self.kimca.KRC_fir( ik, ipd, ipl, spt)
            KRCsed = self.kimca.KRC_sed(ik, ipd, ipl, spt)
            KRC = self.kimca.KRC(ik, ipd, ipl, spt)

            num = self.kimca.RHO(3, spt) * KRCfir / (2 * np.sqrt(self.coca.KL_sec(spt))) - KRCsed / (2 * self.coca.KL_sec(spt))
            den = 1 + ((self.kimca.RHO(4, spt) / 8 ) - ( 5 * self.kimca.RHO(3, spt) ** 2 / 24))

            ans = KRC + num / den
            ans_set.append(ipl * ans)

        return ans_set


    def ESC_KIM(self):
        """
        kim's formula to calculate ESC
        :return:
        """
        spt = fsolve(self.__KL_fir_x, self.est_spt)

        ans_set = []
        for i in np.arange(0, len(self.df)):
            ik = self.df.sector[i]
            ipd = self.df.PD[i]
            ipl = self.df.PL[i]

            KRCfir = self.kimca.KRC_fir(ik, ipd, ipl, spt)
            KRCsed = self.kimca.KRC_sed(ik, ipd, ipl, spt)
            KRC = self.kimca.KRC(ik, ipd, ipl, spt)
            KRC0 = self.kimca.KRC(ik, ipd, ipl, 0)

            dmean = self.kimca.DMEAN(ik, ipd)

            zhat = spt * np.sqrt(self.coca.KL_sec(spt))

            tp = 1 / ((1 - self.kimca.CDF(self.x0, spt)) * np.sqrt(2 * np.pi)) * np.exp(self.coca.KL(spt) - self.x0 * spt)

            a = (KRC - KRC0) / zhat * (1 - (self.kimca.RHO(4, spt) / 8 - (5 * self.kimca.RHO(3, spt) ** 2 / 24)
                                                 - (self.kimca.RHO(3, spt) / (2 * zhat)) - (1 / (zhat ** 2))))
            b = (self.kimca.RHO(3, spt) / 2 + 1 / zhat) * KRCfir / (zhat * self.coca.KL_sec(spt))
            c = KRCsed / (2 * zhat * self.coca.KL_sec(spt))

            ans = (dmean + tp * (a + b - c))* ipl
            ans_set.append(ans)

        return ans_set



if __name__ == '__main__':


    pf = portfolio_info()
    pf.init_rcobligor()

    cbvpara = CBVmodel()
    cbvpara.CBV2()

    coca = cgf_calculation(pf, cbvpara)
    kimca = Kimfunction(coca)

    model = RCcalculation( kimca, est_spt = 2.78, x0 = 0.4419)

   # martin = model.VARC_MARTIN()
    esc_diff = model.ESC_diff()


"""
    df = pf.df
    df["m_varc"] = martin
    df["varc_diff"] = varc_diff

    df = df.sort_values(by='EL', ascending=True)

    plt.figure(0, figsize=(12, 8))
    plt.plot(df.EL, df.m_varc, label="M")
    plt.plot(df.EL, df.varc_diff, label="M2")
    plt.xlabel('EL ', fontsize=15)
    plt.ylabel('es contribution', fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig("haha.png")
    plt.show()
"""