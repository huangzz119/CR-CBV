from DataPre.info_portfolio_CBV import portfolio_info,CBVmodel
from MCmethod.MCsimulation import MC_simulation
import os
import sys

filename = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(filename)

import pandas as pan
import numpy as np
import matplotlib.pyplot as plt

pf = portfolio_info()
pf.init_obligor()
#pf.__dict__

cbvpara = CBVmodel()
cbvpara.CBV1()
#cbvpara.__dict__

METHOD1 = 'ISMC'
METHOD2 = 'PMC'
model = "CBV1"
PATH_MODEL1 = 'Result/obligor/{}/{}/'.format(METHOD1,model)
PATH_MODEL2 = 'Result/obligor/{}/{}/'.format(METHOD2,model)

nsenario = 10
nSim1 = 10000
nSim2 = 30000

testing = MC_simulation(pf, cbvpara)
a = testing.MCIS(nsenario, nSim1, PATH_MODEL1)
b = testing.MCP(nsenario, nSim2, PATH_MODEL2)


datais = pan.read_csv(PATH_MODEL1+'ismc_data.csv')
datap = pan.read_csv(PATH_MODEL2+'pmc_loss.csv')




tp = np.arange(0.005, 0.155,0.005)
losshood = np.arange(1, 2.8,0.02)
isvar_df = pan.DataFrame(columns=list(map(str, losshood)))
ises_df = pan.DataFrame(columns=list(map(str, losshood)))
pvar_df = pan.DataFrame(columns=list(map(str, tp)))
pes_df = pan.DataFrame(columns=list(map(str, tp)))


for ss in np.arange(1,nsenario+1):
    lossis = datais.iloc[:,ss-1]
    lhis = datais.iloc[:,ss+9]
    mask = list(map(lambda x: lossis>=x, losshood))
    tpis = list(map(lambda x: (sum(np.array(lhis)[x]))/len(lossis),mask))
    esis = list(map(lambda x: np.mean(np.array(lhis)[x]*np.array(lossis)[x]) ,mask))

    mcp = datap.iloc[:,ss-1]
    ttvar = np.percentile(mcp, list(map(lambda x: (1 - x) * 100, tp)))
    tmask = list(map(lambda x: mcp >= x, ttvar))
    ttes = list(map(lambda x: sum(np.array(mcp)[x]) / len(np.array(mcp)[x]), tmask))

    isvar_df.loc["tp" + str(ss)] = tpis
    ises_df.loc["es" + str(ss)] = esis
    pvar_df.loc["var" + str(ss)] = ttvar
    pes_df.loc["es" + str(ss)] = ttes


isvar_df.loc["mean_value"] = isvar_df.mean()
isvar_df.loc["variance"] = isvar_df[:-1:].var()
ises_df.loc["mean_value"] = ises_df.mean()
ises_df.loc["variance"] = ises_df[:-1:].var()

pvar_df.loc["mean_value"] = pvar_df.mean()
pvar_df.loc["variance"] = pvar_df[:-1:].var()
pes_df.loc["mean_value"] = pes_df.mean()
pes_df.loc["variance"] = pes_df[:-1:].var()


plt.figure(0, figsize=(12, 8))
plt.plot(losshood, isvar_df.iloc[-2,:], label="MC(IS)-CBV")
plt.plot(pvar_df.iloc[-2,:], tp, label="MC-CBV")
plt.xlabel('VaR x',  fontsize=15)
plt.ylabel('Tail probability', fontsize=15)
plt.legend(fontsize = 15)
plt.savefig('var.png')
plt.show()


