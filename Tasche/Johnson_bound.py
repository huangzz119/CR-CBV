import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

gamma = 15.3323
delta = 5.2397
xi = -0.3962
lambda1 = 73.4615

alpha = 0.001
Z = norm.ppf( 1-alpha )
var = xi + lambda1 * 1 / ( 1 + np.exp((gamma-Z)/delta))
print(var)


def integrand(y):
    """
    this function is to calculate the ES, for numerical integration
    :param y:
    :return:
    """
    term1 = xi + lambda1 / ( 1 + np.exp( (gamma-y)/delta ))
    ans = term1 * norm.pdf(y)
    return ans

L0 = var
K = gamma + delta * np.log((L0 - xi)/(xi - L0 + lambda1))
es = quad(integrand,  K, np.inf)[0] / alpha
print(es)
