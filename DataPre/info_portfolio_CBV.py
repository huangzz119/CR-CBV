"""
Created on Wed Sep 4 17:43:41 2019

@author: hzz
"""
# path add into current package
import os
import sys
os.getcwd()
fileDir = os.path.dirname(os.getcwd())
sys.path.append(fileDir)

import pandas as pan
import numpy as np
import json

"""
portfolio information
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


class portfolio_info():

    def __init__(self):
        """
        K: the number of sector
        df: dataframe of portfolio information
        Lmean: mean of expected loss
        """

        self.K = 10
        self.df = 0
        self.Lmean = 0

    def init_obligor(self):

        df_origin = pan.read_excel(fileDir+"/DataPre/Testing.xls", header=1)
        df_info = df_origin.iloc[0:df_origin.shape[0], [2, 7, 19, 15, 6]]
        df_info.columns = ["obligor", "PD", "PL", "sector", "rating"]
        df_info["EL"] = df_info.apply(lambda x: x["PD"] * x["PL"], axis=1)

        df_obligor = df_info.iloc[range(0, len(df_info), 2)]
        plcopy = df_obligor.loc[:,"PL"].copy()
        df_obligor.loc[:, "PL"] = list(map(lambda x: x * 2 / 1000000000, plcopy))
        elcopy = df_obligor.loc[:,"EL"].copy()
        df_obligor.loc[:, "EL"] = list(map(lambda x: x * 2 / 1000000000, elcopy))

        # sort descending, replace the df and Lmean
        self.df = df_obligor.sort_values(by='PL', ascending=False)
        self.Lmean = sum(df_obligor.EL)

    def init_sector(self):

        df_origin = pan.read_excel(fileDir+"/DataPre/Testing.xls", header=1)
        df_info = df_origin.iloc[0:df_origin.shape[0], [2, 7, 19, 15, 6]]
        df_info.columns = ["obligor", "PD", "PL", "sector", "rating"]
        df_info["EL"] = df_info.apply(lambda x: x["PD"] * x["PL"], axis=1)

        df_sector = pan.DataFrame(columns=["sector","PL", "EL","PD"])
        df_sector["sector"] = np.arange(1, 11)
        df_sector["NumCP"] = list(map(lambda x: len(df_info[df_info["sector"] == x]) / 2, df_sector.sector.values))
        df_sector["PL"] = list(map(lambda x: sum(df_info[ df_info["sector"] == x ].PL)/ 1000000000, df_sector.sector.values))
        df_sector["EL"] = list(map(lambda x: sum(df_info[df_info["sector"] == x].EL)/ 1000000000, df_sector.sector.values))
        df_sector["PD"] = df_sector.EL.values/df_sector.PL.values
        df_sector["NumCP"] = list(map(lambda x: len(df_info[ df_info["sector"] == x ])/2, df_sector.sector.values))

        # replace the df and Lmean
        self.df = df_sector
        self.Lmean = sum(df_sector.EL)

    def init_rcobligor1(self):

        df_origin = pan.read_excel(fileDir+"/DataPre/Testing.xls", header=1)
        df_info = df_origin.iloc[0:df_origin.shape[0], [2, 7, 19, 15, 6]]
        df_info.columns = ["obligor", "PD", "PL", "sector", "rating"]
        df_info["EL"] = df_info.apply(lambda x: x["PD"] * x["PL"], axis=1)

        df_obligor = df_info.iloc[range(0, len(df_info), 2)]
        plcopy = df_obligor.loc[:, "PL"].copy()
        df_obligor.loc[:, "PL"] = list(map(lambda x: x * 2 / 100000000, plcopy))
        elcopy = df_obligor.loc[:, "EL"].copy()
        df_obligor.loc[:, "EL"] = list(map(lambda x: x * 2 / 100000000, elcopy))

        df_rc = df_obligor.loc[df_obligor["sector"] == 1]
        df_rc = df_rc.sort_values(by='PL', ascending=True).drop_duplicates(['EL'])[25:45]

        for i in np.arange(2, 11):
            df2 = df_obligor.loc[df_obligor["sector"] == i].sort_values(by='EL', ascending=True)
            tempdf = pan.concat([df_rc, df2], ignore_index=True)
            df2 = tempdf.drop_duplicates(['PL'])
            df2_select = df2.loc[df2["sector"] == i][25:45]
            df_rc = pan.concat([df_rc, df2_select], ignore_index=True)

        self.df = df_rc
        self.Lmean = sum(df_rc.EL)

    def init_rcobligor2(self):

        print("information for risk contribution")

        df_origin = pan.read_excel(fileDir+"/DataPre/Testing.xls", header=1)
        df_info = df_origin.iloc[0:df_origin.shape[0], [2, 7, 19, 15, 6]]
        df_info.columns = ["obligor", "PD", "PL", "sector", "rating"]
        df_info["EL"] = df_info.apply(lambda x: x["PD"] * x["PL"], axis=1)

        df_obligor = df_info.iloc[range(0, len(df_info), 2)]
        plcopy = df_obligor.loc[:, "PL"].copy()
        df_obligor.loc[:, "PL"] = list(map(lambda x: x * 2 / 100000000, plcopy))
        elcopy = df_obligor.loc[:, "EL"].copy()
        df_obligor.loc[:, "EL"] = list(map(lambda x: x * 2 / 100000000, elcopy))

        df_rc = df_obligor.loc[df_obligor["sector"] == 1]
        df_rc = df_rc.sort_values(by='PL', ascending=False).drop_duplicates(['PL']).iloc[[1,2,3,1]]

        df2 = df_obligor.loc[df_obligor["sector"] == 2].sort_values(by='PL', ascending=False)
        tempdf = pan.concat([df_rc, df2], ignore_index=True)
        df2 = tempdf.drop_duplicates(['EL'])
        df2_select = df2.loc[df2["sector"] == 2].iloc[[4,4, 4,4]]
        df_rc = pan.concat([df_rc, df2_select], ignore_index=True)

        df3 = df_obligor.loc[df_obligor["sector"] == 3].sort_values(by='PL', ascending=False)
        tempdf = pan.concat([df_rc, df3], ignore_index=True)
        df2 = tempdf.drop_duplicates(['EL'])
        df2_select = df2.loc[df2["sector"] == 3].iloc[[4, 4, 4, 4]]
        df_rc = pan.concat([df_rc, df2_select], ignore_index=True)

        self.df = df_rc
        self.Lmean = sum(df_rc.EL)


"""
Model parameter information:
there are five parameter sets. 
"""

class CBVmodel():

    datafile = open(fileDir+"/DataPre/test.txt", 'r')
    js = datafile.read()
    dic = json.loads(js)
    datafile.close()

    def __init__(self):
        self.L = 0
        self.delta = 0
        self.thetak = 0
        self.gamma = 0
        self.thetal = 0

    def CBV1(self):
        self.L = CBVmodel.dic["CBV1"]["L"]
        self.delta = CBVmodel.dic["CBV1"]["delta"]
        self.thetak = CBVmodel.dic["CBV1"]["thetak"]
        self.gamma = CBVmodel.dic["CBV1"]["gamma"]
        self.thetal = CBVmodel.dic["CBV1"]["thetal"]

    def CBV2(self):
        self.L = CBVmodel.dic["CBV2"]["L"]
        self.delta = CBVmodel.dic["CBV2"]["delta"]
        self.thetak = CBVmodel.dic["CBV2"]["thetak"]
        self.gamma = CBVmodel.dic["CBV2"]["gamma"]
        self.thetal = CBVmodel.dic["CBV2"]["thetal"]

    def CBV3(self):
        self.L = CBVmodel.dic["CBV3"]["L"]
        self.delta = CBVmodel.dic["CBV3"]["delta"]
        self.thetak = CBVmodel.dic["CBV3"]["thetak"]
        self.gamma = CBVmodel.dic["CBV3"]["gamma"]
        self.thetal = CBVmodel.dic["CBV3"]["thetal"]


if __name__ == '__main__':

    # read all portfolio information
    pf = portfolio_info()
    model = CBVmodel()



