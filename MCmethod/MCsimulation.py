import pandas as pan
import numpy as np

from DataPre.info_portfolio_CBV import portfolio_info, CBVmodel

from scipy.optimize import fsolve
import json

import os
from cgf_functions import cgf_calculation
import time


class MCsimulation():

    def __init__(self, coca, est_spt= 0.9, x0 = 1.72, level = 0.8277):

        self.coca = coca

        self.df = coca.df
        self.K = coca.K
        self.Lmean = coca.Lmean

        self.L = coca.L
        self.delta = coca.delta
        self.thetak = coca.thetak
        self.thetal = coca.thetal
        self.gamma = coca.gamma

        # for important sampling, change into loss x0
        self.x0 = x0
        self.est_spt = est_spt

        # the contribution level
        self.level = level

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

    def __MC_twist_para(self, t0):
        """
        this function is used to define the twist parameters in important sampling
        """
        twist = []  # the twisting terms
        for k in np.arange(0, self.K):
            twist.append(self.coca.PK(k + 1, t0))

        for l in np.arange(0, self.L):
            pl = 0
            for i in np.arange(0, self.K):
                pl += self.coca.PK(i + 1, t0) * self.gamma[l][i]
            twist.append(pl)

        # combine the theta,
        theta = self.thetak + self.thetal

        # changing the scale value
        scale = 1 / (1 -  np.array(twist))
        return theta, scale

    def MCIS(self, nsenario, nSim, PATH_MODEL, CONTRI = False):
        """
        this function is to generation the set of portfolio loss for every simulation
        :param nsenario: the number of senario
        :param nsimulate: the number of simulation in each senario
        :return: the set of loss
        """

        # the twisting parameter
        # t0 should in the range to make the scale be positive
        t0 = fsolve(self.__KL_fir_x, self.est_spt)
        # the twisting term for gamma distribution
        theta, scale = self.__MC_twist_para(t0)

        if any(scale)<0:
            print("negative value of scale")

        # for all senario data
        a = list(map(str, np.arange(1, nsenario + 1)))
        loss_senario_df = pan.DataFrame(columns=list(map(lambda x:"loss"+x, a))+list(map(lambda x:"likehood"+x, a)))

        if CONTRI:
            with open(PATH_MODEL+"IS_contri_data.json", 'r') as load_f:
                load_dict = json.load(load_f)

        senario = 0
        while senario < nsenario:
            senario += 1
            if senario % 2 == 0:
                print("Congs! ----------The senario is: ", senario)

            dfc = self.df.copy()

            # random gamma variable
            ssim = np.random.gamma(np.tile(theta,(nSim,1)), np.tile(scale,(nSim,1)))

            weig = ssim[:, 0:self.K] * self.delta
            for l in np.arange(0, self.L):
                weig += np.tile(self.gamma[l], (nSim, 1)) * ssim[:, self.K+l : self.K+l+1]


            tlambda = [ (np.transpose(np.tile( x*np.exp(y*t0), (nSim,1)))  * weig[:,w-1]).tolist()
                        for w in np.arange(1, self.K +1)    for x,y in
                        zip(dfc.PD[dfc.sector ==w].values, dfc.PL[dfc.sector ==w].values)]
            # shape: Num_obligor:1:nSim

            tlambda = np.array(tlambda)[:, 0, :]
            # shape: Num_obligor * nSim

            tilting_lambda = np.transpose(tlambda)

            # 这里按照了某顺序，for obligor 需要注意在算谁的obligor contri， 可以在dict再增加一列
            pl = np.array([x for w in np.arange(1, self.K +1) for x in dfc.PL[dfc.sector ==w].values])

            rv = np.random.poisson(lam = tilting_lambda)
            contri_loss = pl * rv

            loss_before = np.sum(contri_loss, axis=1)
            tilt_para = np.exp(-t0 * loss_before + self.coca.KL(t0))

            loss_senario_df["loss" + str(senario)] = loss_before
            loss_senario_df["likehood" + str(senario)] = tilt_para

            if CONTRI:
                # VaR contribution
                index_up = np.where(self.level - loss_before <=  0.01 * self.level)[0]
                index_down = np.where(self.level - loss_before >= - 0.01 * self.level)[0]
                index_varc = list(set(index_up) & set(index_down))

                print("the efficient number of VARC:", len(index_varc))

              #  index_varc = np.where(loss_before == self.x0)[0].tolist()

                load_dict[str(senario)]["vden"] += sum(tilt_para[index_varc])
                vnen_ = [contri_loss[ind,:] * tilt_para[ind] for ind in index_varc]
                vnen_ = np.sum(vnen_, axis=0)
                load_dict[str(senario)]["vnen"] = (load_dict[str(senario)]["vnen"] + vnen_).tolist()


                print("success save varc of senario in load_dict"+ str(senario))

                # ES contribution
                index_esc = np.where(loss_before >= self.level)[0]

                load_dict[str(senario)]["eden"] += sum(tilt_para[index_esc])
                enen_ = [ contri_loss[ind, :] * tilt_para[ind] for ind in index_esc]
                enen_ = np.sum(enen_, axis=0)
                load_dict[str(senario)]["enen"] = (load_dict[str(senario)]["enen"] + enen_).tolist()

                print("the efficient number of ESC:", len(index_esc))

                print("success save esc of senario in load_dict" + str(senario))

        loss_senario_df.to_csv( PATH_MODEL + "ismc_data.csv", mode='a+',header=False, index=False, sep=',')
        print("success write the portfolio loss data")

        if CONTRI:
            with open( PATH_MODEL + 'IS_contri_data.json', 'w') as json_file:
                json.dump(load_dict, json_file)
                print("success save the contribution data")
        return loss_senario_df

    def MCP(self, nsenario, nSim, PATH_MODEL, CONTRI = False):
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

        if CONTRI:
            with open(PATH_MODEL+"P_contri_data.json", 'r') as load_f:
                load_dict = json.load(load_f)

        while senario < nsenario:
            senario += 1
            if senario % 2 == 0:
                print("Congs! ----------The senario is: ", senario)

            dfc = self.df.copy()

            # random gamma variable
            ssim = np.random.gamma(np.tile(theta, (nSim, 1)), np.tile(scale, (nSim, 1)))

            # weight
            weig = ssim[:, 0:self.K] * self.delta
            for l in np.arange(0, self.L):
                weig += np.tile(self.gamma[l], (nSim, 1)) * ssim[:, self.K + l: self.K + l + 1]

            plambda = [ ( np.transpose(np.tile(x, (nSim, 1))) * weig[:, w - 1]).tolist()
                        for w in np.arange(1, self.K + 1) for x in dfc.PD[dfc.sector == w].values]
            plambda = np.array(plambda)[:, 0, :]

            possion_lambda = np.transpose(plambda)
            pl = np.array([x for w in np.arange(1, self.K + 1) for x in dfc.PL[dfc.sector == w].values])

            rv = np.random.poisson(lam=possion_lambda)
            contri_loss = pl * rv

            loss_sim = np.sum(contri_loss, axis=1)
            loss_senario_df[str(senario)] = loss_sim


            if CONTRI:
                # VaR contribution
                index_up = np.where( self.level - loss_sim <=  0.01 * self.level)[0]
                index_down = np.where(self.level - loss_sim >= - 0.01 * self.level)[0]
                index_varc = list(set(index_up) & set(index_down))

              #  index_varc = np.where(loss_before == self.x0)[0].tolist()

                load_dict[str(senario)]["vden"] += len(index_varc)
                vnen_ = [contri_loss[ind,:] for ind in index_varc]
                vnen_ = np.sum(vnen_, axis=0)
                load_dict[str(senario)]["vnen"] = (load_dict[str(senario)]["vnen"] + vnen_).tolist()

                print("the efficient number of VARC:", len(index_varc))

                print("success save varc of senario in load_dict"+ str(senario))

                # ES contribution
                index_esc = np.where(loss_sim >= self.level)[0]

                load_dict[str(senario)]["eden"] += len(index_esc)
                enen_ = [ contri_loss[ind, :] for ind in index_esc]
                enen_ = np.sum(enen_, axis=0)
                load_dict[str(senario)]["enen"] = (load_dict[str(senario)]["enen"] + enen_).tolist()

                print("the efficient number of ESC:", len(index_esc))

                print("success save esc of senario in load_dict" + str(senario))



        loss_senario_df.to_csv( PATH_MODEL + "pmc_loss.csv", mode='a+', header=False, index=False, sep=',')
        print("success write the portfolio loss data")

        if CONTRI:
            with open( PATH_MODEL + 'P_contri_data.json', 'w') as json_file:
                json.dump(load_dict, json_file)
                print("success save the contribution data")

        return loss_senario_df

if __name__ == '__main__':

    pf = portfolio_info()
    pf.init_obligor()

    cbvpara = CBVmodel()
    cbvpara.CBV3()

    coca = cgf_calculation(pf, cbvpara)
    model = MCsimulation(coca)

    pathdic = os.path.join(os.path.join(os.getcwd()), 'MCResult/obligor/CBV3/')

   # start_time = time.time()
   # isresult = model.MCIS(nsenario = 10, nSim = 100000, PATH_MODEL= pathdic, CONTRI= False)
   # print("--- %s seconds in MCIS---" % (time.time() - start_time))
    #478.81078

    start_time = time.time()
    presult = model.MCP(nsenario=10, nSim= 200000, PATH_MODEL=pathdic, CONTRI= False)
    print("--- %s seconds in MCP---" % (time.time() - start_time))
    #732.2329988479


"""
    pf = portfolio_info()
    pf.init_rcobligor()

    cbvpara = CBVmodel()
    cbvpara.CBV3()

    coca = cgf_calculation(pf, cbvpara)

    model = MCsimulation(coca)

    pathdic = os.path.join(os.path.join(os.getcwd()), 'MCResult/rc_obligor/CBV3/')

    start_time = time.time()
    isresult = model.MCIS(nsenario = 10, nSim = 10000, PATH_MODEL= pathdic, CONTRI= True)
    print("--- %s seconds in MCIS---" % (time.time() - start_time))
    #399.051338195

    start_time = time.time()
    presult = model.MCP(nsenario=10, nSim= 10000, PATH_MODEL=pathdic, CONTRI= True)
    print("--- %s seconds in MCP---" % (time.time() - start_time))
    #732.2329988479
"""