from cgf_functions import cgf_calculation
from DataPre.info_portfolio_CBV import portfolio_info, CBVmodel
import pandas as pan
import time
from SPA.SPA_given_tailprob import SPAcalculation
from scipy.optimize import minimize
import numpy as np
import os
from MCmethod.MCimplement import pmc_result, ismc_result,mc_contri
import matplotlib.pyplot as plt
import json
from SPA.risk_contribution import Kimfunction, RCcalculation

pf = portfolio_info()
pf.init_obligor()
cbvpara = CBVmodel()
cbvpara.CBV1()

coca = cgf_calculation(pf, cbvpara)

# portfolio information
info = pan.DataFrame(columns = ["cbv", "mean","var", "skewness","kurtosis"])
mean =coca.KL_fir(0)
var = coca.KL_sec(0)
skew = coca.skewness()
kurt = coca.kurtosis()
info = info.append([{"cbv": "CBV2", "mean": mean, "var": var, "skewness": skew, "kurtosis": kurt}])
print("------------------------------------------------------------------------------------------------")
print("The portfolio information of CBV2 model is:")
print(info)

#calculation VaR and ES at different tail probability
print("tail probability is 0.01")
model = SPAcalculation(coca,  est_spt = 0.9, est_var = 2.7, tailprob= 0.01)

start_time = time.time()
var, spt = model.solver_tailprob_changeto_VaR_spt()
var2, spt2 = model.solver_tailprob_changeto_VaR_spt_2nd()
print(var)
print(var2)
print("--- %s seconds in ES1---" % (time.time() - start_time))
# var = 0.445, spt = 2.789
start_time = time.time()
es1 = model.ES1()
print(es1)
print("--- %s seconds in ES1---" % (time.time() - start_time))
# 0.7134
start_time = time.time()
es2 = model.ES2()
print(es2)
print("--- %s seconds in ES2---" % (time.time() - start_time))
# 0.800
start_time = time.time()
es3 = model.ES3()
print(es3)
print("--- %s seconds in ES3---" % (time.time() - start_time))
# 0.6548
start_time = time.time()
init = np.array(1.3, dtype="float")
es4 = minimize(model.check_function, init, method='Nelder-Mead')
print(es4)
print("--- %s seconds in ES4---" % (time.time() - start_time))
# 0.6546
print("------------------------------------------------------------------------------------------------")


pathdic = os.path.join(os.getcwd(), 'MCmethod/MCResult/obligor/CBV1/')

start_time = time.time()
datap = pan.read_csv(os.path.join(pathdic, 'pmc_loss.csv'))
tp = np.arange(0.01, 0.11, 0.01)
pmc_ans = pmc_result(datap, tp)
print("--- %s seconds in calculation VaR of PMC---" % (time.time() - start_time))

start_time = time.time()
datais = pan.read_csv(os.path.join(pathdic, 'ismc_data.csv'))
range90 = np.arange(1.31,1.35, 0.00005)
range95 = np.arange(1.71, 1.73, 0.00005)
ismc90 = ismc_result(datais, range90, SINGLE= True, level= 0.1)
print("--- %s seconds in calculation VaR of ISMC---" % (time.time() - start_time))


with open(os.path.join(pathdic,'IS_contri_data.json'), 'r') as load_f:
    load_dict_IS = json.load(load_f)

with open(os.path.join(pathdic,'P_contri_data.json'), 'r') as load_f:
    load_dict_P = json.load(load_f)


pmc_contri = mc_contri(load_dict_IS)
ismc_contri = mc_contri(load_dict_P)

kimca = Kimfunction(coca)
model = RCcalculation(kimca, est_spt = 2.50, x0 = 0.8277)

martin = model.VARC_MARTIN()
kim = model.VARC_KIM()
eskim = model.ESC_KIM()

df = pf.df
df["p_varc"] = pmc_contri["varc_mean"]
df["is_varc"] = ismc_contri["varc_mean"]
df["m_varc"] = martin
df["K_varc"] = kim
df["K_esc"] = eskim
df["p_esc"] = pmc_contri["esc_mean"]
df["is_esc"] = ismc_contri["esc_mean"]
df= df.sort_values(by='EL', ascending=True)


plt.figure(2, figsize=(12, 8))
plt.plot(df.EL[:10], df.p_varc[:10], 'ro', label="MC(P)-CBV3")
plt.plot(df.EL, df.is_varc, 'b--',label="MC(IS)-CBV3")
plt.plot(df.EL, df.m_varc, label="Mt-CBV3")
#plt.plot(df.EL[:10], df.K_varc[:10], label="Kim-CBV2")
plt.xlabel('EL',  fontsize=15)
plt.ylabel('var contribution', fontsize=15)
plt.legend(fontsize = 15)
plt.savefig("varc_mc.png")
plt.show()

plt.figure(0, figsize=(12, 8))
plt.plot(df.EL[:10], df.p_esc[:10], 'ro', label="MC(P)-CBV3")
plt.plot(df.EL[:10], df.is_esc[:10], 'b--', label="MC(IS)-CBV3")
#plt.plot(df.EL[92:], df.K_esc[92:], label="Kim-CBV")
plt.xlabel('EL ',  fontsize=15)
plt.ylabel('es contribution', fontsize=15)
plt.legend(fontsize = 15)
plt.savefig("esc_mc.png")
plt.show()
