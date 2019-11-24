import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve,minimize
from scipy import special
import matplotlib.pyplot as plt

# negative binomial distribution

def function_tailprob_changeto_VaR(var):
    """
    :return: given the var, return the corresponding spt, then return the CDF
    """
    def KL_fir_changing(t):
        """
        this function is for given a var, solve out t.
        """
        ans = r*p*np.exp(t)/(1-p*np.exp(t))
        return ans - var

    troot = fsolve(KL_fir_changing, 2)

    KL = r*np.log(1-p) - r*np.log(1-p*np.exp(troot))
    KL_sec = r*p*np.exp(troot)/((1-p*np.exp(troot))**2)

    uhat = troot * np.sqrt(KL_sec)
    what = np.sign(troot) * np.sqrt(2 * (troot * var - KL))
    ans = norm.cdf(what) + norm.pdf(what) * (1 / what - 1 / uhat)
    return (1 - ans) - tailprob

def solver_tailprob_changeto_VaR_spt():
    """
    :return: return the corresponding VaR and spt given the tail probability
    """
    var = fsolve(function_tailprob_changeto_VaR, 2)
    def KL_fir_changing(t):
        """
        this function is for given a var, solve out t.
        """
        ans = r * p * np.exp(t) / (1 - p * np.exp(t))
        return ans - var
    spt = fsolve(KL_fir_changing, 1)
    return var, spt

def ES3(var, spt):

    KL = r * np.log(1 - p) - r * np.log(1 - p * np.exp(spt))
    KL_sec = r * p * np.exp(spt) / ((1 - p * np.exp(spt)) ** 2)

    uhat = spt * np.sqrt(KL_sec)
    what = np.sign(spt) * np.sqrt(2 * (spt * var - KL))

    ans = Lmean * (1 - norm.cdf(what)) + \
          norm.pdf(what) * ((var / uhat) - (Lmean / what) + (Lmean - var) / what ** 3 + 1 / (spt * uhat))

    ans = ans / tailprob

    return ans

def ES_heavytail(var):

        def g_function_solve_y2(y2):
            t2 = var * (y2 ** (-2)) * np.log(var / y2)
            ans = alpha*beta /(1-beta*t2) - y2
            return ans

        y2 = fsolve(g_function_solve_y2, 0.5)
        t1 = np.log(var / y2)
        t2 = var * (y2 ** (-2)) * np.log(var / y2)

        # the Lmean
        alpha0 = alpha*beta
        alpha1 = 0

        kXX = np.exp(t1) - 1 - alpha * np.log( 1-beta*t2 )
        wc = np.sign((var - alpha0)) * np.sqrt(2 * (t1 * (var / y2) + t2 * y2 - kXX))

        hessian = np.array([[np.exp(t1), 0], [0, alpha*(beta**2)/((1-beta*t2)**2)]]).astype(float)
        invhessian = np.linalg.inv(hessian)

        jacobian = np.array([[1 / y2[0], -var * (y2 ** (-2))[0]], [0, 1]]).astype(float)
        gy2 = jacobian[:, 1]
        gy1 = jacobian[:, 0]
        bmt = np.array([t1[0], t2[0]])

        kc = (np.linalg.det(hessian) / (np.linalg.det(jacobian) ** 2)) * (
                np.dot(np.dot(gy2, invhessian), gy2) + np.dot(bmt, gy2 ** 2))

        uc = np.dot(bmt, gy1) * np.sqrt(kc)

        temp1 = (1 - norm.cdf(wc)) * (alpha0+alpha1 - var)

        temp2 = norm.pdf(wc) * ((alpha0+alpha1 - var) / wc - (alpha0 - var) / (wc ** 3)
                                - 1 / (np.dot(bmt, gy1) * uc))

        ans = var + 1 / tailprob * (temp1 - temp2)

        return ans

def ES_log_heavytail(var):

        def g_function_solve_y2(y2):
            t2 = var * (y2 ** (-2)) * np.log(var / y2)
            ans = np.log(beta) + special.digamma(alpha + t2) - y2
            return ans

        y2 = fsolve(g_function_solve_y2, 0.5)
        t1 = np.log(var / y2)
        t2 = var * (y2 ** (-2)) * np.log(var / y2)

        # the Lmean
        alpha0 = np.exp( np.log(beta) + special.digamma(alpha) )
        alpha1 = special.polygamma(1, alpha) * np.exp(np.log(beta) + special.digamma(alpha))/2

        hessian = np.array([[np.exp(t1), 0], [0, special.polygamma(1, alpha + t2)]]).astype(float)
        invhessian = np.linalg.inv(hessian)

        jacobian = np.array([[1 / y2[0], -var * (y2 ** (-2))[0]], [0, 1/y2]]).astype(float)
        gy2 = jacobian[:, 1]
        gy1 = jacobian[:, 0]

        gy = np.array([var/y2, np.log(y2)])
        bmt = np.array([t1[0], t2[0]])

        kXX = np.exp(t1) - 1 + np.log(special.gamma(alpha + t2)) - np.log(special.gamma(alpha)) + t2 * np.log(beta)
        wc = np.sign((var - alpha0)) * np.sqrt(2 * ( np.dot(bmt, gy) - kXX))

        kc = (np.linalg.det(hessian) / (np.linalg.det(jacobian) ** 2)) * (
                np.dot(np.dot(gy2, invhessian), gy2) + np.dot(bmt, gy2 ** 2))
        uc = np.dot(bmt, gy1) * np.sqrt(kc)


        temp1 = (1 - norm.cdf(wc)) * (alpha0 + alpha1 - var)

        temp2 = norm.pdf(wc) * ( (alpha0 + alpha1 - var) / wc - (alpha0 - var) / (wc ** 3)
                                - 1 / (np.dot(bmt, gy1) * uc))

        answer = var + 1 / tailprob * (temp1 - temp2)

        return answer




if __name__ == '__main__':

    r = 100
    p = 0.01

    Lmean = r * p / (1 - p)
    alpha = r
    beta = p / (1 - p)

    tail = np.arange(0.1, 4, 0.1)

    VAR3_set = []
    ES3_set = []

    VARHV_set = []
    ESHV_set = []
    for tailprob in np.arange(0.1, 4, 0.1):

        print("---------"+str(tailprob))

        var, spt = solver_tailprob_changeto_VaR_spt()
        es_ans = ES3(var,spt)
        ES3_set.append(es_ans)
        VAR3_set.append(var)

        init = np.array(3, dtype="float")
        ans = minimize(ES_heavytail, init, method='Nelder-Mead')

        ESHV_set.append(ans.fun)
        VARHV_set.append(ans.x[0])

    plt.figure(0, figsize=(12, 8))
    plt.plot(tail, ES3_set, label="SPA: 2-order", linestyle=':', marker='|', color = "darkcyan")
    plt.plot(tail, ESHV_set, label="SPA: Y=X1X2", color = 'orange')
    plt.xlabel('Tail Probability', fontsize=20)
    plt.ylabel('ES', fontsize=20)
    plt.legend(fontsize=20)
    plt.title("Negative Binomial Distribution", fontsize = 30)
    plt.show()





    #a = minimize(ES_log_heavytail, init, method='Nelder-Mead')
