import numpy as np
from SPA.cgf_functions import cgf_calculation
from DataPre.info_portfolio_CBV import portfolio_info,CBVmodel
import pandas as pan
from scipy.stats import norm
import math
from scipy.optimize import fsolve,minimize
from SPA.SPA_given_tailprob import SPAcalculation

class RCcalculation():

    def __init__(self, coca, est_spt= 0.99, x0 = 1.7):

        self.coca = coca

        self.df = coca.df
        self.K = coca.K
        self.Lmean = coca.Lmean

        self.L = coca.L
        self.delta = coca.delta
        self.thetak = coca.thetak
        self.thetal = coca.thetal
        self.gamma = coca.gamma

        # estimator the value
        self.est_spt = est_spt
        # var level
        self.x0 = x0


    def PDF(self):

        spt = fsolve(self.__KL_fir_x, self.est_spt)
        rho3 = self.coca.KL_thi(spt) / (self.coca.KL_sec(spt)**1.5)
        rho4 = self.coca.KL_for(spt) / (self.coca.KL_sec(spt)**2)
        ans = np.exp(self.coca.KL(spt) - spt*self.x0)/np.sqrt(2*math.pi*self.coca.KL_sec(spt)) * (1+1/8*(rho4-5/3*rho3**2))

        return ans

    def PDF_first_order(self):

        spt = fsolve(self.__KL_fir_x, self.est_spt)
        ans = np.exp(self.coca.KL(spt) - spt * self.x0) / np.sqrt(2 * math.pi * self.coca.KL_sec(spt))

        return ans

    def __KL_fir_x(self, t):
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


    def VARC_HUANG_first(self):

        # den; return to the original df first
        self.df = self.coca.original_df()
        spt = fsolve(self.__KL_fir_x, self.est_spt)
        pdf = np.exp(self.coca.KL(spt) - spt*self.x0)/np.sqrt(2*math.pi*self.coca.KL_sec(spt))

        # num; change to the contribution version first
        set_contribution = []
        for i in np.arange(0, len(self.df)):
            if i % 10 == 0:
                print("the obligor = " + str(i + 1))
            self.df = self.coca.original_df()
            sing_exposure = self.df.PL.values[i]
            sing_pd = self.df.PD.values[i]
            sing_sector = self.df.sector.values[i]
            idx_sector = sing_sector -1

           # self.df = self.coca.contri_df(i)

            # step 1:
            # for the sector gamma distribution
            def __changeK_KL_ex_fir_x(t):
                ans = 0
                # sum of all sectors
                for i in np.arange(0, self.K):
                    pk = self.coca.PK(i + 1, t)
                    pk1 = self.coca.PK_drv(i + 1, t, 1)
                    # add 1 for this special sector
                    if i == idx_sector:
                        temp = self.delta[i] * (self.thetak[i]+1) * pk1 / (1 - self.delta[i] * pk)
                    else:
                        temp = self.delta[i] * self.thetak[i] * pk1 / (1 - self.delta[i] * pk)
                    ans += temp

                for l in np.arange(0, self.L):
                    pl = 0
                    pl1 = 0
                    for i in np.arange(0, self.K):
                        pl += self.coca.PK(i + 1, t) * self.gamma[l][i]
                        pl1 += self.coca.PK_drv(i + 1, t, 1) * self.gamma[l][i]
                    temp_ = (self.thetal[l]) * pl1 / (1 - pl)
                    ans += temp_
                return ans - (self.x0 - sing_exposure)

            spt_hat1 = fsolve(__changeK_KL_ex_fir_x, self.est_spt)
            pdf_hat1 = np.exp(self.coca.changeK_KL(spt_hat1, idx_sector) - spt_hat1 * (self.x0 - sing_exposure))/np.sqrt( 2 * math.pi * self.coca.changeK_KL_sec(spt_hat1, idx_sector))
            sing_contribution =  pdf_hat1 * self.delta[idx_sector]

            # step 2
            # add all the common factor one by one
            for idx_common in np.arange(0, self.L):
                def __changeL_KL_ex_fir_x(t):
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
                        if l == idx_common:
                            temp_ = (self.thetal[l] + 1) * pl1 / (1 - pl)
                        else:
                            temp_ = (self.thetal[l] ) * pl1 / (1 - pl)
                        ans += temp_
                    return ans - (self.x0 - sing_exposure)

                spt_hat2 = fsolve(__changeL_KL_ex_fir_x, self.est_spt)
                pdf_hat2 = np.exp(self.coca.changeL_KL(spt_hat2, idx_common) - spt_hat2 * (self.x0 - sing_exposure)) / np.sqrt(
                    2 * math.pi * self.coca.changeL_KL_sec(spt_hat2, idx_common))
                sing_contribution +=  pdf_hat2 * self.gamma[idx_common][idx_sector]

            # put into set
            set_contribution.append(sing_exposure * sing_pd * sing_contribution / pdf)

        return set_contribution


    def VARC_HUANG_first_modify(self, epsilon = 0.00001):

        self.df = self.coca.original_df()

        def __KL_fir_x1(t):
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

            return ans - (self.x0 + epsilon/2)
        spt1 = fsolve(__KL_fir_x1, self.est_spt)
        uhat1 = spt1 * np.sqrt(self.coca.KL_sec(spt1))
        what1 = np.sign(spt1) * np.sqrt(2 * (spt1 * self.x0 - self.coca.KL(spt1)))
        tp1 = (norm.cdf(what1) + norm.pdf(what1) * (1 / what1 - 1 / uhat1))

        def __KL_fir_x2( t):
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

            return ans - (self.x0 - epsilon / 2)
        spt2 = fsolve(__KL_fir_x2, self.est_spt)
        uhat2 = spt2 * np.sqrt(self.coca.KL_sec(spt2))
        what2 = np.sign(spt2) * np.sqrt(2 * (spt2 * self.x0 - self.coca.KL(spt2)))
        tp2 = (norm.cdf(what2) + norm.pdf(what2) * (1 / what2 - 1 / uhat2))

        # num; change to the contribution version first
        set_contribution = []
        for i in np.arange(0, len(self.df)):
            if i % 10 == 0:
                print("the obligor = " + str(i + 1))
            self.df = self.coca.original_df()
            sing_exposure = self.df.PL.values[i]
            sing_pd = self.df.PD.values[i]
            sing_sector = self.df.sector.values[i]
            idx_sector = sing_sector -1

            # step 1:
            # for the sector gamma distribution
            def __changeK_KL_ex_fir_x1(t):
                ans = 0
                # sum of all sectors
                for i in np.arange(0, self.K):
                    pk = self.coca.PK(i + 1, t)
                    pk1 = self.coca.PK_drv(i + 1, t, 1)
                    # add 1 for this special sector
                    if i == sing_sector - 1:
                        temp = self.delta[i] * (self.thetak[i] + 1) * pk1 / (1 - self.delta[i] * pk)
                    else:
                        temp = self.delta[i] * self.thetak[i] * pk1 / (1 - self.delta[i] * pk)
                    ans += temp

                for l in np.arange(0, self.L):
                    pl = 0
                    pl1 = 0
                    for i in np.arange(0, self.K):
                        pl += self.coca.PK(i + 1, t) * self.gamma[l][i]
                        pl1 += self.coca.PK_drv(i + 1, t, 1) * self.gamma[l][i]
                    temp_ = (self.thetal[l]) * pl1 / (1 - pl)
                    ans += temp_
                return ans - (self.x0 - sing_exposure + epsilon/2)
            spt_hat1a = fsolve(__changeK_KL_ex_fir_x1, self.est_spt)
            uhat1a = spt_hat1a * np.sqrt(self.coca.changeK_KL_sec(spt_hat1a, idx_sector))
            what1a = np.sign(spt_hat1a) * np.sqrt(2 * (spt_hat1a * self.x0 - self.coca.changeK_KL(spt_hat1a, idx_sector)))
            tp_hat1a = norm.cdf(what1a) + norm.pdf(what1a) * (1 / what1a - 1 / uhat1a)

            def __changeK_KL_ex_fir_x2(t):
                ans = 0
                # sum of all sectors
                for i in np.arange(0, self.K):
                    pk = self.coca.PK(i + 1, t)
                    pk1 = self.coca.PK_drv(i + 1, t, 1)
                    # add 1 for this special sector
                    if i == sing_sector - 1:
                        temp = self.delta[i] * (self.thetak[i] + 1) * pk1 / (1 - self.delta[i] * pk)
                    else:
                        temp = self.delta[i] * self.thetak[i] * pk1 / (1 - self.delta[i] * pk)
                    ans += temp

                for l in np.arange(0, self.L):
                    pl = 0
                    pl1 = 0
                    for i in np.arange(0, self.K):
                        pl += self.coca.PK(i + 1, t) * self.gamma[l][i]
                        pl1 += self.coca.PK_drv(i + 1, t, 1) * self.gamma[l][i]
                    temp_ = (self.thetal[l]) * pl1 / (1 - pl)
                    ans += temp_
                return ans - (self.x0 - sing_exposure - epsilon/2)
            spt_hat1b = fsolve(__changeK_KL_ex_fir_x2, self.est_spt)
            uhat1b = spt_hat1b * np.sqrt(self.coca.changeK_KL_sec(spt_hat1b, idx_sector))
            what1b = np.sign(spt_hat1b) * np.sqrt(2 * (spt_hat1b * self.x0 - self.coca.changeK_KL(spt_hat1b, idx_sector)))
            tp_hat1b =  norm.cdf(what1b) + norm.pdf(what1b) * (1 / what1b - 1 / uhat1b)

            sing_contribution = abs(tp_hat1a -tp_hat1b) * self.delta[idx_sector]

            # step 2
            # add all the common factor one by one
            for idx_common in np.arange(0, self.L):

                def __changeL_KL_ex_fir_x1(t):
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
                        if l == idx_common:
                            temp_ = (self.thetal[l] + 1) * pl1 / (1 - pl)
                        else:
                            temp_ = (self.thetal[l]) * pl1 / (1 - pl)
                        ans += temp_
                    return ans - (self.x0 - sing_exposure - epsilon/2)
                spt_hat2a = fsolve(__changeL_KL_ex_fir_x1, self.est_spt)
                uhat2a = spt_hat2a * np.sqrt(self.coca.changeL_KL_sec(spt_hat2a, idx_common))
                what2a = np.sign(spt_hat2a) * np.sqrt(
                    2 * (spt_hat2a * self.x0 - self.coca.changeL_KL(spt_hat2a, idx_common)))
                tp_hat2a = norm.cdf(what2a) + norm.pdf(what2a) * (1 / what2a - 1 / uhat2a)

                def __changeL_KL_ex_fir_x2(t):
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
                        if l == idx_common:
                            temp_ = (self.thetal[l] + 1) * pl1 / (1 - pl)
                        else:
                            temp_ = (self.thetal[l]) * pl1 / (1 - pl)
                        ans += temp_
                    return ans - (self.x0 - sing_exposure + epsilon/2)
                spt_hat2b = fsolve(__changeL_KL_ex_fir_x2, self.est_spt)
                uhat2b = spt_hat2b * np.sqrt(self.coca.changeL_KL_sec(spt_hat2b, idx_common))
                what2b = np.sign(spt_hat2b) * np.sqrt(
                    2 * (spt_hat2b * self.x0 - self.coca.changeL_KL(spt_hat2b, idx_common)))
                tp_hat2b = norm.cdf(what2b) + norm.pdf(what2b) * (1 / what2b - 1 / uhat2b)

                sing_contribution += abs(tp_hat2b - tp_hat2a) * self.gamma[idx_common][idx_sector]

            # put into set
            set_contribution.append(sing_exposure * sing_pd * sing_contribution / abs(tp1-tp2))

        return set_contribution


    def VARC_HUANG_second(self):

        # den; return to the original df first
        self.df = self.coca.original_df()
        spt = fsolve(self.__KL_fir_x, self.est_spt)
        rho3 = self.coca.KL_thi(spt) / (self.coca.KL_sec(spt) ** 1.5)
        rho4 = self.coca.KL_for(spt) / (self.coca.KL_sec(spt) ** 2)
        pdf = np.exp(self.coca.KL(spt) - spt*self.x0)/np.sqrt(2*math.pi*self.coca.KL_sec(spt)) * (1+1/8*(rho4-5/3*rho3**2))

        # num; change to the contribution version first
        set_contribution = []

        for i in np.arange(0, len(self.df)):
            if i % 10 == 0:
                print("the obligor = " + str(i + 1))
            self.df = self.coca.original_df()
            sing_exposure = self.df.PL.values[i]
            sing_pd = self.df.PD.values[i]
            sing_sector = self.df.sector.values[i]
            idx_sector = sing_sector -1

            # step 1:
            # for the sector gamma distribution
            def __changeK_KL_ex_fir_x(t):
                ans = 0
                # sum of all sectors
                for i in np.arange(0, self.K):
                    pk = self.coca.PK(i + 1, t)
                    pk1 = self.coca.PK_drv(i + 1, t, 1)
                    # add 1 for this special sector
                    if i == sing_sector - 1:
                        temp = self.delta[i] * (self.thetak[i]+1) * pk1 / (1 - self.delta[i] * pk)
                    else:
                        temp = self.delta[i] * self.thetak[i] * pk1 / (1 - self.delta[i] * pk)
                    ans += temp

                for l in np.arange(0, self.L):
                    pl = 0
                    pl1 = 0
                    for i in np.arange(0, self.K):
                        pl += self.coca.PK(i + 1, t) * self.gamma[l][i]
                        pl1 += self.coca.PK_drv(i + 1, t, 1) * self.gamma[l][i]
                    temp_ = (self.thetal[l]) * pl1 / (1 - pl)
                    ans += temp_
                return ans - (self.x0 - sing_exposure)

            spt_hat1 = fsolve(__changeK_KL_ex_fir_x, self.est_spt)

            rho3 = self.coca.changeK_KL_thi(spt_hat1, idx_sector) / (
                        self.coca.changeK_KL_sec(spt_hat1, idx_sector) ** 1.5)
            rho4 = self.coca.changeK_KL_for(spt_hat1, idx_sector) / (self.coca.changeK_KL_sec(spt_hat1, idx_sector) ** 2)
            pdf_hat1 = np.exp(self.coca.changeK_KL(spt_hat1, idx_sector) - spt_hat1 * self.x0) / np.sqrt(
                2 * math.pi * self.coca.changeK_KL_sec(spt_hat1, idx_sector)) * (
                              1 + 1 / 8 * (rho4 - 5 / 3 * rho3 ** 2))

            #pdf_hat1 = np.exp(self.coca.changeK_KL(spt_hat1, idx_sector) - spt_hat1 * self.x0)/np.sqrt( 2 * math.pi * self.coca.changeK_KL_sec(spt_hat1, idx_sector))
            sing_contribution = sing_exposure * sing_pd * pdf_hat1

            # step 2
            # add all the common factor one by one
            for idx_common in np.arange(0, self.L):
                def __changeL_KL_ex_fir_x(t):
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
                        if l == idx_common:
                            temp_ = (self.thetal[l] + 1) * pl1 / (1 - pl)
                        else:
                            temp_ = (self.thetal[l] ) * pl1 / (1 - pl)
                        ans += temp_
                    return ans - (self.x0 - sing_exposure)

                spt_hat2 = fsolve(__changeL_KL_ex_fir_x, self.est_spt)

                rho3 = self.coca.changeL_KL_thi(spt_hat2, idx_common) / (
                        self.coca.changeL_KL_sec(spt_hat2, idx_common) ** 1.5)
                rho4 = self.coca.changeL_KL_for(spt_hat2, idx_common) / (
                            self.coca.changeL_KL_sec(spt_hat2, idx_common) ** 2)
                pdf_hat2 = np.exp(self.coca.changeL_KL(spt_hat2, idx_common) - spt_hat2 * self.x0) / np.sqrt(
                    2 * math.pi * self.coca.changeL_KL_sec(spt_hat2, idx_common)) * (
                                   1 + 1 / 8 * (rho4 - 5 / 3 * rho3 ** 2))

                sing_contribution += sing_exposure * sing_pd * pdf_hat2 * self.gamma[idx_common][idx_sector]

            # put into set
            set_contribution.append(sing_contribution / pdf)

        return set_contribution

    def VARC_MARTIN(self):

        spt = fsolve(self.__KL_fir_x, self.est_spt)
        KLdrv = self.coca.KL_drv(spt)
        ans = self.df.PL.values / spt * KLdrv
        return ans

    def ESC_HUANG_first(self):

        # den; return to the original df first
        self.df = self.coca.original_df()
        spt = fsolve(self.__KL_fir_x, self.est_spt)

        uhat = spt * np.sqrt(self.coca.KL_sec(spt))
        what = np.sign(spt) * np.sqrt(2 * (spt * self.x0 - self.coca.KL(spt)))
        tp = 1 - (norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat))

        # num; change to the contribution version first
        set_contribution = []
        for i in np.arange(0, len(self.df)):
            if i % 10 == 0:
                print("the obligor = " + str(i + 1))
            self.df = self.coca.original_df()
            sing_exposure = self.df.PL.values[i]
            sing_pd = self.df.PD.values[i]
            sing_sector = self.df.sector.values[i]
            idx_sector = sing_sector - 1

            # self.df = self.coca.contri_df(i)
            # step 1:
            # for the sector gamma distribution
            def __changeK_KL_ex_fir_x(t):
                ans = 0
                # sum of all sectors
                for i in np.arange(0, self.K):
                    pk = self.coca.PK(i + 1, t)
                    pk1 = self.coca.PK_drv(i + 1, t, 1)
                    # add 1 for this special sector
                    if i == sing_sector - 1:
                        temp = self.delta[i] * (self.thetak[i] + 1) * pk1 / (1 - self.delta[i] * pk)
                    else:
                        temp = self.delta[i] * self.thetak[i] * pk1 / (1 - self.delta[i] * pk)
                    ans += temp

                for l in np.arange(0, self.L):
                    pl = 0
                    pl1 = 0
                    for i in np.arange(0, self.K):
                        pl += self.coca.PK(i + 1, t) * self.gamma[l][i]
                        pl1 += self.coca.PK_drv(i + 1, t, 1) * self.gamma[l][i]
                    temp_ = (self.thetal[l]) * pl1 / (1 - pl)
                    ans += temp_
                return ans - (self.x0 - sing_exposure)

            spt_hat1 = fsolve(__changeK_KL_ex_fir_x, self.est_spt)
            uhat1 = spt_hat1 * np.sqrt(self.coca.changeK_KL_sec(spt_hat1, idx_sector))
            what1 = np.sign(spt_hat1) * np.sqrt(2 * (spt_hat1 * self.x0 - self.coca.changeK_KL(spt_hat1, idx_sector)))
            tp_hat1 = 1 - (norm.cdf(what1) + norm.pdf(what1) * (1 / what1 - 1 / uhat1))

            sing_contribution = tp_hat1

            # step 2
            # add all the common factor one by one
            for idx_common in np.arange(0, self.L):
                def __changeL_KL_ex_fir_x(t):
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
                        if l == idx_common:
                            temp_ = (self.thetal[l] + 1) * pl1 / (1 - pl)
                        else:
                            temp_ = (self.thetal[l]) * pl1 / (1 - pl)
                        ans += temp_
                    return ans - (self.x0 - sing_exposure)

                spt_hat2 = fsolve(__changeL_KL_ex_fir_x, self.est_spt)

                uhat2 = spt_hat2 * np.sqrt(self.coca.changeL_KL_sec(spt_hat2, idx_common))
                what2 = np.sign(spt_hat2) * np.sqrt(2 * (spt_hat2 * self.x0 - self.coca.changeL_KL(spt_hat2, idx_common)))
                tp_hat2 = 1 - (norm.cdf(what2) + norm.pdf(what2) * (1 / what2 - 1 / uhat2))
                sing_contribution += tp_hat2 * self.gamma[idx_common][idx_sector]

            # put into set
            set_contribution.append((sing_exposure * sing_pd * sing_contribution / tp)[0] )

        return set_contribution


    def ESC_HUANG_second(self):

        # den; return to the original df first
        self.df = self.coca.original_df()
        spt = fsolve(self.__KL_fir_x, self.est_spt)

        uhat = spt * np.sqrt(self.coca.KL_sec(spt))
        what = np.sign(spt) * np.sqrt(2 * (spt * self.x0 - self.coca.KL(spt)))
        lambda3 = self.coca.KL_thi(spt) / (self.coca.KL_sec(spt) ** 1.5)
        lambda4 = self.coca.KL_for(spt) / (self.coca.KL_sec(spt) ** 2)

        temp = (1 / (what ** 3)) - (1 / (uhat ** 3)) - (lambda3 / (2 * (uhat ** 2))) + (1 / uhat) * (
            (lambda4 / 8) - (5 * (lambda3 ** 2) / 24))
        tp = 1 - (norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat - temp))

       # tp_alter1 = 1 - norm.cdf( what - (1/what) * np.log(1/uhat) )

       # temp0_alter2 = self.coca.KL(spt) - spt * self.x0 + spt**2 / 2
       # temp1_alter2 = 1 - lambda3*(spt**3)/6 + lambda4*(spt**4)/24 + (lambda3**2)*(spt**6)/72
       # temp2_alter2 = lambda3*(spt**2 -1)/6 - lambda4*(spt**3-spt)/24 - (lambda3**2)*(spt**5 - spt**3 + 3*spt)/72

       # tp_alter2 = np.exp( temp0_alter2 ) * ( (1 - norm.cdf(spt))*temp1_alter2 + norm.pdf(spt)*temp2_alter2)

        # num; change to the contribution version first
        set_contribution = []
        for i in np.arange(0, len(self.df)):
            if i % 10 == 0:
                print("the obligor = " + str(i + 1))
            self.df = self.coca.original_df()
            sing_exposure = self.df.PL.values[i]
            sing_pd = self.df.PD.values[i]
            sing_sector = self.df.sector.values[i]
            idx_sector = sing_sector - 1

            self.df = self.coca.contri_df(i)
            # step 1:
            # for the sector gamma distribution
            def __changeK_KL_ex_fir_x(t):
                ans = 0
                # sum of all sectors
                for i in np.arange(0, self.K):
                    pk = self.coca.PK(i + 1, t)
                    pk1 = self.coca.PK_drv(i + 1, t, 1)
                    # add 1 for this special sector
                    if i == sing_sector - 1:
                        temp = self.delta[i] * (self.thetak[i] + 1) * pk1 / (1 - self.delta[i] * pk)
                    else:
                        temp = self.delta[i] * self.thetak[i] * pk1 / (1 - self.delta[i] * pk)
                    ans += temp

                for l in np.arange(0, self.L):
                    pl = 0
                    pl1 = 0
                    for i in np.arange(0, self.K):
                        pl += self.coca.PK(i + 1, t) * self.gamma[l][i]
                        pl1 += self.coca.PK_drv(i + 1, t, 1) * self.gamma[l][i]
                    temp_ = (self.thetal[l]) * pl1 / (1 - pl)
                    ans += temp_
                return ans - (self.x0 - sing_exposure)

            spt_hat1 = fsolve(__changeK_KL_ex_fir_x, self.est_spt)
            uhat1 = spt_hat1 * np.sqrt(self.coca.changeK_KL_sec(spt_hat1, idx_sector))
            what1 = np.sign(spt_hat1) * np.sqrt(2 * (spt_hat1 * self.x0 - self.coca.changeK_KL(spt_hat1, idx_sector)))

            lambda13 = self.coca.changeK_KL_thi(spt_hat1, idx_sector) / (self.coca.changeK_KL_sec(spt_hat1, idx_sector) ** 1.5)
            lambda14 = self.coca.changeK_KL_for(spt_hat1, idx_sector) / (self.coca.changeK_KL_sec(spt_hat1, idx_sector) ** 2)

            temp1 = (1 / (what1 ** 3)) - (1 / (uhat1 ** 3)) - (lambda13 / (2 * (uhat1 ** 2))) + (1 / uhat1) * (
                (lambda14 / 8) - (5 * (lambda13 ** 2) / 24))
            tp_hat1 = 1 - (norm.cdf(what1) + norm.pdf(what1) * (1 / what1 - 1 / uhat1 - temp1))

            sing_contribution = tp_hat1 * self.delta[idx_sector]

            # step 2
            # add all the common factor one by one
            for idx_common in np.arange(0, self.L):
                def __changeL_KL_ex_fir_x(t):
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
                        if l == idx_common:
                            temp_ = (self.thetal[l] + 1) * pl1 / (1 - pl)
                        else:
                            temp_ = (self.thetal[l]) * pl1 / (1 - pl)
                        ans += temp_
                    return ans - (self.x0 - sing_exposure)

                spt_hat2 = fsolve(__changeL_KL_ex_fir_x, self.est_spt)

                uhat2 = spt_hat2 * np.sqrt(self.coca.changeL_KL_sec(spt_hat2, idx_common))
                what2 = np.sign(spt_hat2) * np.sqrt(2 * (spt_hat2 * self.x0 - self.coca.changeL_KL(spt_hat2, idx_common)))

                lambda23 = self.coca.changeL_KL_thi(spt_hat2, idx_common) / (
                            self.coca.changeL_KL_sec(spt_hat2, idx_common) ** 1.5)
                lambda24 = self.coca.changeL_KL_for(spt_hat2, idx_common) / (
                            self.coca.changeL_KL_sec(spt_hat2, idx_common) ** 2)

                temp2 = (1 / (what2 ** 3)) - (1 / (uhat2 ** 3)) - (lambda23 / (2 * (uhat2 ** 2))) + (1 / uhat2) * (
                        lambda24 / 8 - 5 * (lambda23 ** 2) / 24)
                tp_hat2 = 1 - (norm.cdf(what2) + norm.pdf(what2) * (1 / what2 - 1 / uhat2 - temp2))
                sing_contribution += tp_hat2 * self.gamma[idx_common][idx_sector]

            # put into set
            set_contribution.append((sing_exposure * sing_pd * sing_contribution / tp)[0] )

        return set_contribution


if __name__ == '__main__':


    pf = portfolio_info()
    pf.init_rcobligor1()

    cbvpara = CBVmodel()
    cbvpara.CBV1()

    coca = cgf_calculation(pf, cbvpara)
    root = coca.QL_root()

    skew_value = coca.skewness()
    kurt_value = coca.kurtosis()
    coca.KL_sec(0)

    print("the mean of the portfolio loss:", coca.Lmean)
    print("the first order derivative of cgf at t=0:", coca.KL_fir(0))
    print("minimum upper bound of t inside the root of cgf:", root)


    ans_set = []
    spt_set = np.arange(0.01, root - 0.005, 0.05)
    for t in spt_set:
        ans = coca.KL_fir(t)
        ans_set.append(ans)
    result = pan.DataFrame()
    result["spt"] = spt_set
    result["fir"] = ans_set

    model1 = SPAcalculation(coca, est_spt= 1.2, est_var= 1.4 , tailprob=0.01)
    var, spt = model1.solver_tailprob_changeto_VaR_spt_2nd()
    es2 = model1.ES2()
    es3 = model1.ES3()

    init = np.array(1, dtype="float")
    es4 = minimize(model1.check_function, init, method='Nelder-Mead')

    # # spt = 0.9977, var = 1.7078

    model = RCcalculation( coca, est_spt = spt, x0 = var)

    esc1 = model.ESC_HUANG_first()
    esc2 = model.ESC_HUANG_second()

    varc1 = model.VARC_HUANG_first()
    varc2 = model.VARC_HUANG_second()

    print("ESC", sum(esc1))
    print("VARC", sum(varc1))




    #ans1 = model.VARC_Change_HUANG_first()

    martin = model.VARC_MARTIN()
    #esc_diff = model.ESC_diff()


    ans_result = pf.df
    ans_result["varc1"] = varc1
    ans_result["varc2"] = varc2
    ans_result["esc1"] = esc1
    ans_result["esc2"] = esc2
