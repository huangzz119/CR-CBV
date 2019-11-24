import pandas as pan
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from DataPre.info_portfolio_CBV import portfolio_info, CBVmodel
from SPA.risk_contribution import Kimfunction, RCcalculation
from cgf_functions import cgf_calculation

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
        esis = list(map(lambda x: np.mean(np.array(lhis)[x] * np.array(lossis)[x]), mask))

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
    pathdic = os.path.join(os.getcwd(), 'MCResult/rc_obligor/CBV2/')

    tp = np.arange(0.001, 0.35, 0.002)
    datap = pan.read_csv(os.path.join(pathdic, 'pmc_loss.csv'))

    losshood = np.arange(0.25, 1.1, 0.005)
    datais = pan.read_csv(os.path.join(pathdic, 'ismc_data.csv'))

    # range90 = np.arange(0.33, 0.335,0.00002)
    # range95 = np.arange(0.455,0.46, 0.00002)
    # range99 = np.arange(0.795, 0.8, 0.00002)

    pmc_ans = pmc_result(datap, tp)
    ismc_ans = ismc_result(datais, losshood, SINGLE = False)

    plt.figure(0, figsize=(12, 8))
    plt.plot( pmc_ans.var_mean,pmc_ans.tp, label="MC(P)-CBV2")
    plt.plot( ismc_ans.var,ismc_ans.tp_mean, label="MC(IS)-CBV2")
    plt.xlabel('VaR x',  fontsize=15)
    plt.ylabel('Tail probability', fontsize=15)
    plt.legend(fontsize = 15)
    plt.savefig("var_mc.png")
    plt.show()

    plt.figure(0, figsize=(12, 8))
    plt.plot(pmc_ans.es_mean, pmc_ans.tp, label="MC(P)-CBV2")
    plt.plot(ismc_ans.es_mean, ismc_ans.tp_mean, label="MC(IS)-CBV2")
    plt.xlabel('ES ', fontsize=15)
    plt.ylabel('Tail probability', fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig("es_mc.png")
    plt.show()



    with open(os.path.join(pathdic,'IS_contri_data.json'), 'r') as load_f:
        load_dict_IS = json.load(load_f)

    with open(os.path.join(pathdic,'P_contri_data.json'), 'r') as load_f:
        load_dict_P = json.load(load_f)


    pmc_contri = mc_contri(load_dict_IS)
    ismc_contri = mc_contri(load_dict_P)

    pf = portfolio_info()
    pf.init_rcobligor()
    cbvpara = CBVmodel()
    cbvpara.CBV2()

    coca = cgf_calculation(pf, cbvpara)
    kimca = Kimfunction(coca)
    model = RCcalculation(kimca, est_spt = 2.80, x0 = 0.45)

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


    plt.figure(0, figsize=(12, 8))
    plt.plot(df.EL, df.p_varc, label="MC(P)-CBV2")
    plt.plot(df.EL, df.is_varc, label="MC(IS)-CBV2")
    plt.plot(df.EL, df.m_varc, label="Mt-CBV2")
    plt.plot(df.EL, df.K_varc, label="Kim-CBV2")
    plt.xlabel('EL ',  fontsize=15)
    plt.ylabel('var contribution', fontsize=15)
    plt.legend(fontsize = 15)
    plt.savefig("varc_mc.png")
    plt.show()

    plt.figure(0, figsize=(12, 8))
    plt.plot(df.EL, df.p_esc, label="MC(P)-CBV")
    plt.plot(df.EL, df.is_esc, label="MC(IS)-CBV")
    plt.plot(df.EL[92:], df.K_esc[92:], label="Kim-CBV")
    plt.xlabel('EL ',  fontsize=15)
    plt.ylabel('es contribution', fontsize=15)
    plt.legend(fontsize = 15)
    plt.savefig("esc_mc.png")
    plt.show()
