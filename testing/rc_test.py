"""
this file is to calculate the risk contribution for obligor i
"""
import os
os.getcwd()

from SPAcalculator import SPA
import pandas as pan
import numpy as np
from scipy.optimize import fsolve
import pickle
file = open('rctest_data', 'wb')

import json

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


df_origin = pan.read_excel("Testing.xls", header=1)
df_ = df_origin.iloc[0:df_origin.shape[0], [2, 7,  19, 15, 6]]
df_.columns = ["obligor", "PD",    "PL", "sector", "rating"]
df_["EL"] = df_.apply(lambda x: x["PD"] * x["PL"], axis=1)

# Sort Descending
df_ = df_.sort_values(by='PL', ascending=False)

df_new = df_.iloc[range(0, len(df_), 2)]
df_new.loc[:,"PL"] = list(map(lambda x: x * 2 / 1000000000, df_new["PL"]))
df_new.loc[:,"EL"] = list(map(lambda x: x * 2 / 1000000000, df_new["EL"]))

df = df_new


# the confidence level alpha = 99
var99 = 2


res = pan.DataFrame(columns = ["obligor", "PL","sector", "rating","varcm","varck","esck"])
for obligor in np.arange(1, 3, 1):
    #if obligor% == 0:
    print("--------the calculation is "+str(obligor)+"-th obligor")
    test1 = SPA(L1, delta1, thetak1, gamma1, thetal1, i0 = obligor, x0 = var99)
    test2 = SPA(L2, delta2, thetak2, gamma2, thetal2, i0 = obligor, x0 = var99)
    test3 = SPA(L3, delta3, thetak3, gamma3, thetal3, i0 = obligor, x0 = var99)
    test = [test1, test2, test3]

    for i in np.arange(0, 3):
        testing = test[i]
        varcm = testing.VARC_MARTIN()
        varck = testing.VARC_KIM()
        esck = testing.ESC_KIM()

        res = res.append([{"obligor": df.obligor.values[obligor-1], "PL": df.PL.values[obligor-1]*1000000000,
                           "sector": df.sector.values[obligor-1], "rating": df.rating.values[obligor-1],
                           "varcm": varcm*1000000000, "varck": varck*1000000000, "esck": esck*2000000000}])

print("the confidence level is alpha = 99")
print("print top 20 obligors's information")
print(res)

pickle.dump(res, file)
file.close()

