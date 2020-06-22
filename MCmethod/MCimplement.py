import os
import sys
os.getcwd()
fileDir = os.path.dirname(os.getcwd())
sys.path.append(fileDir)

import pandas as pan
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from DataPre.info_portfolio_CBV import portfolio_info, CBVmodel
from SPA.risk_contribution import Kimfunction, RCcalculation
from SPA.cgf_functions import cgf_calculation
from scipy.optimize import minimize
from SPA.SPA_given_tailprob import SPAcalculation

def pmc_result(datap, tp):
    """
    this function is to generate the var and es at the tail probability level
    :param datap:
    :param tp: the tail probability
    """
    nsenario = 10
    sena = np.arange(1, nsenario + 1)

    pmc_dataframe = pan.DataFrame()
    pmc_dataframe["tp"] = tp

    for ss in sena:

        mcp = datap.iloc[:, ss - 1]
        ttvar = np.percentile(mcp, list(map(lambda x: (1 - x) * 100, tp)))
        tmask = list(map(lambda x: mcp >= x, ttvar))
        ttes = list(map(lambda x: sum(np.array(mcp)[x]) / len(np.array(mcp)[x]), tmask))

        pmc_dataframe["var" + str(ss)] = ttvar
        pmc_dataframe["es" + str(ss)] = ttes

    pmc_dataframe["var_mean"] = pmc_dataframe.loc[:, list(map(lambda x: "var" + str(x), sena))].mean(axis=1)
    pmc_dataframe["var_std"] = pmc_dataframe.loc[:, list(map(lambda x: "var" + str(x), sena))].std(axis=1)


    pmc_dataframe["es_mean"] = pmc_dataframe.loc[:, list(map(lambda x: "es" + str(x), sena))].mean(axis=1)
    pmc_dataframe["es_std"] = pmc_dataframe.loc[:, list(map(lambda x: "es" + str(x), sena))].std(axis=1)


    return pmc_dataframe.loc[:,["tp", "var_mean", "var_std", "es_mean", "es_std"]]

def ismc_result(datais, losshood, SINGLE = False, level = .01):
    """
    this function has two kinds of output:
    when SIGNLE = TRUE, the output is the var and es at a certain loss level;
    when SIGNLE = FALSE, the output is the mean and st of tp and es at a range of var
    :param datais:
    :param losshood:
    :param SINGLE: choose to output a single value of a range of value
    :param level: the losshood when SINGLE is TRUE
    """
    nsenario = 10
    sena = np.arange(1, nsenario + 1)

    ismc_dataframe = pan.DataFrame()
    ismc_dataframe["var"] = losshood

    for ss in sena:
        lossis = datais.iloc[:, ss - 1]
        lhis = datais.iloc[:, ss + 9]

        mask = list(map(lambda x: lossis >= x, losshood))
        tpis = list(map(lambda x: (sum(np.array(lhis)[x])) / len(lossis), mask))
        esis = list(map(lambda x: np.mean(np.array(lossis)[x]), mask))
     #   esis = list(map(lambda x: np.mean( np.array(lhis)[x] * np.array(lossis)[x] ), mask))

        ismc_dataframe["tp" + str(ss)] = tpis
        ismc_dataframe["es" + str(ss)] = esis

    if SINGLE == True:
        ans = pan.DataFrame(columns=["tp","var_mean","var_std","es_mean","es_std" ])
        ans["tp"] = [level]

        for ss in sena:
            ismc_dataframe["tp" + str(ss)] = abs( ismc_dataframe["tp" + str(ss)].values  - level )

        var_list = list(map(lambda ss: ismc_dataframe["var"][ismc_dataframe["tp" + str(ss)] < 1e-5].mean(), sena))
        es_list =  list(map(lambda ss: ismc_dataframe["es"+str(ss)][ismc_dataframe["tp" + str(ss)] < 1e-5].mean(), sena))

        var_ls = [x for x in var_list if x is not np.nan]
        es_ls = [x for x in es_list if x is not np.nan]
        ans["var_mean"] = np.mean(var_ls)
        ans["var_std"] = np.std(var_ls)
        ans["es_mean"] = np.mean(es_ls)
        ans["es_std"] = np.std(es_ls)

    else:
        ismc_dataframe["tp_mean"] = ismc_dataframe.loc[:, list(map(lambda x: "tp"+str(x), sena))].mean(axis = 1)
        ismc_dataframe["tp_std"] = ismc_dataframe.loc[:, list(map(lambda x: "tp"+str(x), sena))].std(axis = 1) \
                                /ismc_dataframe.tp_mean.values
        ismc_dataframe["es_mean"] = ismc_dataframe.loc[:, list(map(lambda x: "es" + str(x), sena))].mean(axis=1)
        ismc_dataframe["es_std"] = ismc_dataframe.loc[:, list(map(lambda x: "es" + str(x), sena))].std(axis=1) \
                                / ismc_dataframe.es_mean.values
        ans = ismc_dataframe.loc[:,["var", "tp_mean", "tp_std", "es_mean", "es_std"]]

    return ans

def mc_contri(load_dict_final):

    nsenario = 10
    sena = np.arange(1, nsenario + 1)

    a = list(map(str, sena))
    contr_df = pan.DataFrame(columns=list(map(lambda x: "varc" + x, a)) + list(map(lambda x: "esc" + x, a)))

    for ss in sena:
        sena_dict = load_dict_final[str(ss)]

        varc = [x / sena_dict["vden"] for x in sena_dict["vnen"]]
        esc = [x / sena_dict["eden"] for x in sena_dict["enen"]]

        contr_df["varc" + str(ss)] = list(map(lambda x: x, varc))
        contr_df["esc" + str(ss)] = list(map(lambda x: x, esc))

    contr_df["varc_mean"] = contr_df.loc[:, list(map(lambda x: "varc" + str(x), sena))].mean(axis=1)
    contr_df["varc_std"] = contr_df.loc[:, list(map(lambda x: "varc" + str(x), sena))].std(axis=1) \
                           / contr_df.varc_mean.values

    contr_df["esc_mean"] = contr_df.loc[:, list(map(lambda x: "esc" + str(x), sena))].mean(axis=1)
    contr_df["esc_std"] = contr_df.loc[:, list(map(lambda x: "esc" + str(x), sena))].std(axis=1) \
                          / contr_df.esc_mean.values

    return contr_df.loc[:, ["varc_mean", "varc_std", "esc_mean", "esc_std"]]

if __name__ == '__main__':

    pathdic = os.path.join( fileDir + '/MCmethod/MCResult/obligor/CBV2/')
    tp = np.arange(0.01, 0.11, 0.01)
    datap = pan.read_csv(os.path.join(pathdic, 'pmc_loss.csv'))
    pmc_ans = pmc_result(datap, tp)
    losshood = np.arange(4.1, 4.35, 0.05)
    datais = pan.read_csv(os.path.join(pathdic, 'ismc_data.csv'))
    ismc_ans = ismc_result(datais, losshood, SINGLE = False)

    pf = portfolio_info()
    pf.init_obligor()
    cbvpara = CBVmodel()
    cbvpara.CBV2()
    coca = cgf_calculation(pf, cbvpara)
    root = coca.QL_root()
    tailprob_set = np.arange(0.001, 0.05, 0.001)
    ans_set = pan.DataFrame(columns=["var1", "var2", "var_check", "es1", "es2", "es3", "es_check"])
    init = np.array(2.5, dtype="float")
    for idx, i in enumerate(tailprob_set):
        model = SPAcalculation(coca, est_spt=1.2, est_var=2.5, tailprob=i)
        var1, spt1 = model.solver_tailprob_changeto_VaR_spt()
        var2, spt2 = model.solver_tailprob_changeto_VaR_spt_2nd()
        es1 = model.ES1()
        es2 = model.ES2()
        es3 = model.ES3()
        es4 = minimize(model.check_function, init, method='Nelder-Mead')
        ans_set.loc[idx] = [var1, var2, es4.x[0], es1, es2, es3, es4.fun]

    plt.figure(1, figsize=(12, 8))
    plt.xlim(0.0001, 0.01)
    plt.plot( pmc_ans.tp, pmc_ans.var_mean, "y", label="MC(P)")
    plt.plot( pmc_ans.tp, pmc_ans.var_mean + (1.96 * pmc_ans.var_std / np.sqrt(10)), "y--", label="MC(P)-95%CI")
    plt.plot( pmc_ans.tp, pmc_ans.var_mean - (1.96 * pmc_ans.var_std / np.sqrt(10)), "y--", label="MC(P)-95%CI")
    plt.plot( ismc_ans.tp_mean, ismc_ans["var"],  "c",  label="MC(IS)")
    plt.plot( tailprob_set, ans_set.var1,  label="SPA-first order")
    plt.plot( tailprob_set, ans_set.var2,  label="SPA-second order")
    plt.plot( tailprob_set, ans_set.var_check,  label="check function")
    plt.title("CBV2", fontsize=20)
    plt.ylabel('RD',  fontsize=15)
    plt.xlabel('Tail probability', fontsize=15)
    plt.legend(fontsize = 15)
    plt.grid(linestyle='-.')
    plt.savefig("var_new.png")
    plt.show()

    plt.figure(2, figsize=(12, 8))
    plt.ylim(3.6, 4.5)
    plt.xlim(0.002, 0.008)
    plt.plot( pmc_ans.tp, pmc_ans.es_mean,"y", label="MC(P)")
    plt.plot( pmc_ans.tp, pmc_ans.es_mean + (1.96 * pmc_ans.es_std / np.sqrt(10)), "y--", label="MC(P)-95%CI")
    plt.plot( pmc_ans.tp, pmc_ans.es_mean - (1.96 * pmc_ans.es_std / np.sqrt(10)), "y--", label="MC(P)-95%CI")
    plt.plot( tailprob_set,ans_set.es1, label="two-calls")
    plt.plot( tailprob_set,ans_set.es2, label="SPA-first order")
    plt.plot( tailprob_set,ans_set.es3,"o", label="SPA-second order")
    plt.plot( tailprob_set,ans_set.es_check, label="Check Function")
    plt.title("CBV1", fontsize=20)
    plt.ylabel('Expected Shortfall', fontsize=15)
    plt.xlabel('Tail probability', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(linestyle='-.')
    plt.savefig("es_new.png")
    plt.show()


    with open(os.path.join(pathdic,'IS_contri_data.json'), 'r') as load_f:
        load_dict_IS = json.load(load_f)
    with open(os.path.join(pathdic,'P_contri_data.json'), 'r') as load_f:
        load_dict_P = json.load(load_f)

    pmc_contri = mc_contri(load_dict_IS)
    ismc_contri = mc_contri(load_dict_P)
    kimca = Kimfunction(coca)
    model = RCcalculation(kimca, est_spt = 2.80, x0 = 0.44)
    martin = model.VARC_MARTIN()

    df = pf.df
    df["p_varc"] = pmc_contri["varc_mean"]
    df["is_varc"] = ismc_contri["varc_mean"]
    df["Martin_varc"] = martin
    df["p_esc"] = pmc_contri["esc_mean"]
    df["is_esc"] = ismc_contri["esc_mean"]
    df= df.sort_values(by='EL', ascending=True)

    plt.figure(0, figsize=(12, 8))
    plt.plot( df.EL, df.p_varc, "ro",  label="MC(P)-CBV2",)
    plt.plot( df.EL, df.is_varc, "b--", label="MC(IS)-CBV2")
    plt.plot( df.EL, df.Martin_varc, "b--", label="Martin-CBV2")
    plt.xlabel('Expected Loss ',  fontsize=15)
    plt.ylabel('Var Contribution', fontsize=15)
    plt.legend(fontsize = 15)
    plt.savefig("varc_mc.png")
    plt.show()





