"""
this file is to calculate the VaR and ES at given confidence level
"""
from SPAcalculator import SPA
import pandas as pan
import numpy as np
import pickle
import csv
import codecs
import json

import os
os.getcwd()
file = open('alpha_test_data', 'wb')

datafile = open('test.txt', 'r')
js = datafile.read()
dic = json.loads(js)
datafile.close()

L1 = dic["CBV1"]["L"]
delta1 = dic["CBV1"]["delta"]
thetak1 = dic["CBV1"]["thetak"]
gamma1 = dic["CBV1"]["gamma"]
thetal1 = dic["CBV1"]["thetal"]

L2 = dic["CBV2"]["L"]
delta2 = dic["CBV2"]["delta"]
thetak2 = dic["CBV2"]["thetak"]
gamma2 = dic["CBV2"]["gamma"]
thetal2 = dic["CBV2"]["thetal"]

L3 = dic["CBV3"]["L"]
delta3 = dic["CBV3"]["delta"]
thetak3 = dic["CBV3"]["thetak"]
gamma3 = dic["CBV3"]["gamma"]
thetal3 = dic["CBV3"]["thetal"]



if __name__ == '__main__':

    # calculating the corresponding VaR and ES given alpha"
    print("Here calculating the corresponding VaR and ES given alpha")
    alphaset = [0.9, 0.95, 0.99]

    ansset = []
    for md in np.arange(0, 3, 1):
        print("-----the CBV model with" + str(md+1) + " common factors-------")
        ansing = pan.DataFrame(columns = ["alpha", "spt", "x","ES1","ES2","ES3"])

        for ap in np.arange(0, len(alphaset)):
            if md == 0:
                temptest = SPA(L1, delta1, thetak1, gamma1, thetal1, alpha0 = alphaset[ap])
            if md == 1:
                temptest = SPA(L2, delta2, thetak2, gamma2, thetal2, alpha0 = alphaset[ap])
            if md == 2:
                temptest = SPA(L3, delta3, thetak3, gamma3, thetal3, alpha0 = alphaset[ap])

            print("-----------the alpha equals to "+str(alphaset[ap])+"------------------")

            tvar = temptest.VaR_given_alpha()
            tspt = temptest.spt_given_alpha(tvar)

            es1 = temptest.ES1(tvar, alphaset[ap])
            es2 = temptest.ES2(tvar, tspt, alphaset[ap])
            es3 = temptest.ES3(tvar, tspt, alphaset[ap])

            ansing = ansing.append([{"alpha": alphaset[ap], "spt": tspt, "x": tvar, "ES1": es1, "ES2": es2, "ES3": es3}])
        print(ansing)
        ansset.append(ansing)

    ans1 = ansset[0]
    ans2 = ansset[1]
    ans3 = ansset[2]

    pickle.dump(ansset, file)
    file.close()

    with open("alpha_result.csv", 'w') as csvfile:
        writer = csv.writer(csvfile)

    ans1.to_csv("alpha_result.csv",index=False,sep=',')
    ans2.to_csv("alpha_result.csv",mode='a', index=False,sep=',')
    ans3.to_csv("alpha_result.csv",mode='a', index=False,sep=',')

    data = pan.read_csv('alpha_result.csv')
