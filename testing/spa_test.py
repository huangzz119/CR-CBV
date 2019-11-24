"""
this file is to plot the VaR and ES
"""
from SPAcalculator import SPA
import pandas as pan
import numpy as np
import pickle
import csv

import json

import os
os.getcwd()

file = open('spa_test_data', 'wb')

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

test1 = SPA(L1, delta1, thetak1, gamma1, thetal1)
test2 = SPA(L2, delta2, thetak2, gamma2, thetal2)
test3 = SPA(L3, delta3, thetak3, gamma3, thetal3)
test = [test1, test2, test3]



with open("SPA_result.csv", 'w') as csvfile:
    writer = csv.writer(csvfile)
    head =  ["spt", "alpha","TailProb", "x","ES1","ES2","ES3"]
    writer.writerow(head)

# dataframe for plot
resset = []
for md in np.arange(0, 3, 1):
    print("-----the CBV model with " + str(md+1) + " common factors-------")
    testing = test[md]
    resing = pan.DataFrame(columns = ["spt", "alpha","TailProb", "x","ES1","ES2","ES3"])
    count = 0
    for t in np.arange(0.7, 1.2,0.01):
        count += 1
        if count%5 == 0:
            print("-----the calculation time is " +str(count)+ "---------")

        x = testing.KL_fir(t)
        alpha = testing.CDF(x, t)
        tailprob = 1-alpha
        es1 = testing.ES1(x, alpha)
        es2 = testing.ES2(x, t, alpha)
        es3 = testing.ES3(x, t, alpha)
        resing = resing.append([{"spt":t, "alpha":alpha, "TailProb": tailprob, "x":x, "ES1":es1, "ES2":es2,"ES3":es3}])

        with open("SPA_result.csv", 'a+', newline='') as csvfile:
            csv_write = csv.writer(csvfile)
            row = [t, alpha, tailprob, x, es1, es2, es3]
            csv_write.writerow(row)

    resset.append(resing)

res1 = resset[0]
res2 = resset[1]
res3 = resset[2]

pickle.dump(resset, file)

print(" Finish the calculation")





#load data
"""

with open('pickle_example.pickle', 'rb') as file:
    a_dict1 =pickle.load(file)

print(a_dict1)
"""
