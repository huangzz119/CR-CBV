from scipy.optimize import fsolve
import numpy as np
from cgf_functions import cgf_calculation
from DataPre.info_portfolio_CBV import portfolio_info,CBVmodel
from scipy.stats import norm

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


    def VARC_MARTIN(self):
        """
        to calculate the VARC, the necessary input is i0 and x0
        :return:
        """

        spt = fsolve(self.__KL_fir_x, self.est_spt)

        # the obligor is in sector ik, but the index for para is idx
        # idx = self.ik - 1
        # temp1 = self.ipl * self.ipd * np.exp(t0 * self.ipl)

        temp1 = self.df.PL.values * self.df.PD.values * np.exp(spt * self.df.PL.values)

        # self.coca.PK(i + 1, t)

        temp2 = [(self.delta[ w -1] * self.thetak[ w -1]) / (1 - self.delta[ w -1] * self.coca.PK(w, spt))
                 for w in self.df.sector.values]

        temp3 = 0
        for l in np.arange(0, self.L):
            pl = 0
            for i in np.arange(0, self.K):
                pl += self.coca.PK(i + 1, spt) * self.gamma[l][i]
            temp3 +=  self.thetal[l] /  ( 1 -pl)

        ans = temp1 * (np.array(temp2) + temp3)

        return ans


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

    model = RCcalculation( kimca, est_spt = 2.80, x0 = 0.45)

    martin = model.VARC_MARTIN()
   # kim = model.VARC_KIM()
    es = model.ESC_KIM()
