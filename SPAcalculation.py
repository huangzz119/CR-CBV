import pandas as pan
import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import root, fsolve
import matplotlib.pyplot as plt


"""
credit capital model
Version 1

1. all tenors are set to one year
2. obligors are given by peers(counterparty)
3. each obligor was given a hundred percent weight to the sector which it belongs;
so the correlation of sectors are connected by common background factors
4. neglect the assumption of normalized sector 
"""

# portfolio information
"""
3000 obligors 2 term loans each
counterparty number: 3000
sector number: 10
rating number: 7

PD: probability of default
SD(PD): the standard derivation of probability of default
EAD: exposure at default
LGD: loss given default
PL: potential loss, PL = EAD * LGD
EL: expected loss, EL = PL * PD
"""

df_origin = pan.read_excel("Testing.xls",header =1)
df = df_origin.iloc[0:df_origin.shape[0], [2,7,8,18, 22,19,15,6]]
df.columns = ["obligor","PD","SD(PD)","EAD","LGD","PL", "sector","rating"]
df["PL"] = df["PL"].map(lambda x: x/1000000000)
df["EAD"] = df["EAD"].map(lambda x: x/1000000)
df["EL"] = df.apply(lambda x: x["PD"] * x["PL"],axis = 1 )

portfolio_EAD = df["EAD"].sum()
portfolio_PL = df["PL"].sum()
portfolio_EL = df["EL"].sum()
average_PD = df["PD"].mean()

#sector information
"""
there are ten sectors
mean default rate: EL/PL (don't need to use)
variance
covariance
"""
st_origin = pan.read_excel("Sectors.xls")
st_origin["mean"]=0
st_mean = []
for i in np.arange(0,10):
    st = df.loc[df["sector"] == i+1]
    st_mean.append(st["EL"].sum()/st["PL"].sum())
st_origin["mean"]=st_mean
st_var = st_origin["var"]
st_cov = st_origin.iloc[0:9, 0:9]

# estimate parameters
"""
First example: only one common factor
"""
"""
def opt(a, b, c, d):
    for i in np.arange(0,10):
        a[i]*b[i] +c[i]* d[i] = 1
        (a[i]**2)*b[i] + (c[i]**2)* d = st_var[0]
        c[i]*c[i+1]*d = st_cov.iloc[0,1]
    return a, b, c, d
"""

# CBV gamma factors
K = 10
L = 1

delta = [3.10470, 0.32719, 0.24383, 0.84626, 0.74251, 1.50169, 1.37805, 1.34703, 2.41331, 5.33036]
thetak = [0.17295, 1.09437, 1.20121, 0.52224, 0.73185, 0.41985, 0.47681, 0.26698, 0.16698, 0.05573]
gamma = [[0.88492, 1.22682, 1.35137, 1.06651, 0.87261, 0.70619, 0.65540, 1.22383, 1.14098, 1.34346]]
thetal = [0.52325]


# the minimum upper boundary of saddlepoint
up_bound = []
for k in np.arange(0, K):
    st = df.loc[df["sector"] == k + 1]
    muk = sum (st.PD )
    up = math.log( 1/delta[k]+ muk) / sum ( st.PD * st.PL)
    up_bound.append(up)
for l in np.arange(0, L):
    mul = 0
    den = 0
    for i in np.arange(0, len(thetak)):
        st = df.loc[df["sector"] == i + 1]
        temp_mu = sum( st.PD*gamma[l][i] )
        mul = mul + temp_mu
        temp_den = sum ( st.PD* gamma[l][i]* st.PL )
        den = den + temp_den
    up =  math.log( 1+ mul )/den
    up_bound.append(up)
min_up = min(up_bound)

# define minimum loss and maximum loss
Lmax = sum(df.PL)
Lmin = sum(df.EL)

# the CGF
def KL(t):
    ans = 0
    for i in np.arange(0, K):
        st = df.loc[df["sector"] == i+1]  # sectors
        Qk = sum( st.PD * (np.exp(t * st.PL)-1) )
        temp = - thetak[i]*math.log(1-delta[i]*Qk)
        ans = ans + temp
    for l in np.arange(0, L):
        Ql = 0
        for i in np.arange(0, K):
            st = df.loc[df["sector"] == i+1]
            tempQl = sum( st.PD * (np.exp(t * st.PL) - 1) * gamma[l][i])
            Ql = Ql + tempQl
        temp = - thetal[l] * math.log(1 -  Ql)
        ans = ans + temp
    return ans

def KL_fir(t):
    ans = 0
    for i in np.arange(0, K):
        st = df.loc[df["sector"] == i + 1]
        Qk = sum( st.PD * ( np.exp(t * st.PL) - 1))
        Qk1 = sum(st.PL * st.PD * (np.exp(t * st.PL)))
        Nk1 = Qk1 / (1 - delta[i] * Qk)
        temp = delta[i] * thetak[i] * Nk1
        ans = ans + temp
    for l in np.arange(0, L):
        Ql = 0
        Ql1 = 0
        for i in np.arange(0, K):
            st = df.loc[df["sector"] == i + 1]
            tempQl = sum(st.PD * (np.exp(t * st.PL) - 1) * gamma[l][i])
            tempQl1 = sum(st.PL * st.PD * gamma[l][i] * np.exp(t * st.PL))
            Ql = Ql + tempQl
            Ql1 = Ql1 + tempQl1
        Nl1 = Ql1 / (1 - Ql)
        temp = thetal[l] * Nl1
        ans = ans + temp
    return ans


def KL_sec(t):
    ans = 0
    for i in np.arange(0, K):
        st = df.loc[df["sector"] == i + 1]
        Qk = sum(st.PD * (np.exp(t * st.PL) - 1))
        Qk1 = sum(st.PL * st.PD * (np.exp(t * st.PL)))
        Qk2 = sum(st.PL**2 * st.PD * (np.exp(t * st.PL)))
        Nk1 = Qk1 / ( 1 - delta[i]* Qk)
        Nk2 = Qk2 / ( 1 - delta[i] * Qk)
        temp = delta[i]* thetak[i] * ( Nk2 + delta[i]*(Nk1**2) )
        ans = ans + temp
    for l in np.arange(0, L):
        Ql = 0
        Ql1 = 0
        Ql2 = 0
        for i in np.arange(0, K):
            st = df.loc[df["sector"] == i + 1]
            tempQl = sum(st.PD * (np.exp(t * st.PL) - 1) * gamma[l][i])
            tempQl1 = sum(st.PL * st.PD * gamma[l][i] * np.exp(t * st.PL))
            tempQl2 = sum((st.PL**2) * st.PD * gamma[l][i] * np.exp(t * st.PL))
            Ql = Ql + tempQl
            Ql1 = Ql1 + tempQl1
            Ql2 = Ql2 + tempQl2
        Nl1 = Ql1 / (1-Ql)
        Nl2 = Ql2 / (1-Ql)
        temp = thetal[l] * (Nl2 + (Nl1**2))
        ans = ans + temp
    return ans
# CDF
def FL(var, spt):
    what = np.sign(spt) * np.sqrt(2*(spt*var - KL(spt)))
    uhat = spt * np.sqrt(KL_sec(spt))
    ans = norm.cdf(what) + norm.pdf(what)(1/what - 1/uhat)
    return ans

#spt = fsolve(KL_fir, 0.1)

spt_set = np.arange(0, min_up,0.5)
var_set = []
for i in np.arange(0,len(spt_set)):
    spt = spt_set[i]
    var = KL_fir(spt)
    var_set.append(var)



plt.figure(0, figsize=(20, 15))
plt.plot(spt_set, var_set)
plt.xlabel('spt',  fontsize=20)
plt.ylabel('var', fontsize=20)
#plt.legend(fontsize = 20)
#plt.savefig("First_Deriv_Plot_for Example " +str ( temp + 1 ) + ".png")
plt.show()




