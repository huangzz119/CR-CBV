import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import time
from scipy.stats import norm

N = 1
kappa_1 = 0.6060
kappa_2 = 0.33
skewness = 2.6633
kappa_3 = skewness * (kappa_2 ** 1.5)
kurtosis = 14.8809
kappa_4 = (kurtosis -3) * (kappa_2 ** 2)

x = 1.2913
an = norm.cdf(x) - kappa_3/6 * norm.pdf(x) *(x**2 - 1) - (kappa_4-3)/24 * norm.pdf(x) * x *(x**2 - 3)

kappa_1 + np.sqrt(kappa_2) * x

def GCVAR(x):
    """
    :param x: x
    :return: cumulative function
    """
    ans = norm.cdf(x) - kappa_3/6 * (norm.pdf(x) *(x**2 - 1)) - (kappa_4-3)/24 * (norm.pdf(x) * x *(x**2 - 3))
    return ans - 0.999
init = np.array(1.3 , dtype="float")
ans = fsolve(GCVAR, init )
kappa_1 + np.sqrt(kappa_2) * ans


# The Pearson coefficients of skewness^2 and kurtosis
beta_1 = kappa_3 ** 2
beta_2 = kappa_4
#  the lower bound omega_1
D = (3 + beta_2) * (16 * (beta_2 ** 2) + 87 * beta_2 + 171) / 27
d = -1 + (7 + 2 * beta_2 + 2 * np.sqrt(D)) ** (1 / 3) - (2 * np.sqrt(D) - 7 - 2 * beta_2) ** (1 / 3)
omega_1 = 0.5 * (-1 + np.sqrt(d) + np.sqrt( 4 / np.sqrt(d) - d - 3))
omega_0 = 0.5 * (8 + 4 * beta_1 + 4 * np.sqrt(4 * beta_1 + beta_1 ** 2)) ** (1 / 3) \
          + 2 * (8 + 4 * beta_1 + 4 * np.sqrt(4 * beta_1 + beta_1 ** 2)) ** (-1 / 3) - 1
omega_min = max(omega_0, omega_1)
# the upper bound
omega_2 = np.sqrt(-1 + np.sqrt(2 * (beta_2 - 1)))
omega_max = omega_2
def fun(x):
    m = -2 + np.sqrt(4 + 2 * (x ** 2 - (beta_2 + 3) / (x ** 2 + 2 * x + 3)))
    ans = (x - 1 - m )* (( x + 2 + m/2)**2)
    return ans
# verify
f_omega_min = fun(omega_min)
# f(w_2) = 0
f_omega_max = fun(omega_max)
# f(w_1) = (w_1 -1 )* (w_1 + 2)**2
f_omega_1a = fun(omega_1)
f_omega_1b = (omega_1-1) * (omega_1 + 2)**2

interval = np.arange(omega_min, omega_max, 0.02)
fun_set = []
for interval_value in interval:
    fun_value = fun(interval_value)
    fun_set.append(fun_value)
plt.figure(1, figsize=(12, 8))
plt.plot(interval, fun_set)
plt.title("CBV(1)")
plt.xlabel('omega',  fontsize=15)
plt.ylabel('f(omega)', fontsize=15)
plt.savefig("jenson.png")
plt.show()

def fun_solve(x):
    m = -2 + np.sqrt(4 + 2 * (x ** 2 - (beta_2 + 3) / (x ** 2 + 2 * x + 3)))
    ans = (x - 1 - m) * ((x + 2 + m / 2) ** 2)
    return ans - beta_1

init = np.array((omega_max - omega_min)/2 + omega_min , dtype="float")
omega_root = fsolve(fun_solve, init)[0]
omega = max(omega_root, omega_min)
# without using obsolute value
m = -2 + np.sqrt(4 + 2 * (omega ** 2 - (beta_2 + 3) / (omega ** 2 + 2 * omega + 3)))
#verify beta_1 and beta_2
beta_1_test = (omega - 1 - m) * (omega + 2 + 0.5 * m) ** 2
beta_2_test = (omega ** 2 + 2 * omega + 3) * (omega ** 2 - 0.5 * m ** 2 - 2 * m) - 3
Omega = - np.sign(kappa_3) * np.arcsinh( np.sqrt( (omega + 1)/(2 * omega) * ((omega - 1) / m - 1)))
delta_J = 1 / np.sqrt(np.log(omega))
gamma_J = Omega * delta_J
lambda_J = np.sqrt(kappa_2) / (omega - 1) * np.sqrt(2 * m / (omega + 1))
#xi_J = kappa_1 + np.sign(kappa_3) * np.sqrt(kappa_2) / (omega - 1) * np.sqrt(omega - 1 - m)
xi_J = kappa_1 + lambda_J * np.sqrt(omega) * np.sinh(gamma_J / delta_J)


#Z = np.random.randn(N, 1)
alpha = 0.05
Z = norm.ppf( 1-alpha )
X = (xi_J + lambda_J *( np.exp( (Z-gamma_J)/delta_J) - np.exp((gamma_J-Z)/delta_J))/2)
L0 = X
K = gamma_J + delta_J * np.log(((L0 - xi_J)/lambda_J) + np.sqrt(((L0 - xi_J)/lambda_J)**2 + 1))
ES = (xi_J*(1-norm.cdf(K)) +
      lambda_J/2 * np.exp( - gamma_J/delta_J + 1/(2*delta_J**2))* (1-norm.cdf(K - 1/delta_J))
      - lambda_J/2 *  np.exp( gamma_J/delta_J + 1/(2*delta_J**2))* (1-norm.cdf(K + 1/delta_J))) * 1/0.05



def coefficients(kappa_1, kappa_2, kappa_3, kappa_4, alpha):
    # The Pearson coefficients of skewness^2 and kurtosis
    beta_1 = kappa_3 ** 2
    beta_2 = kappa_4
    #  the lower bound omega_1
    D = (3 + beta_2) * (16 * (beta_2 ** 2) + 87 * beta_2 + 171) / 27
    d = -1 + (7 + 2 * beta_2 + 2 * np.sqrt(D)) ** (1 / 3) - (2 * np.sqrt(D) - 7 - 2 * beta_2) ** (1 / 3)
    omega_1 = 0.5 * (-1 + np.sqrt(d) + np.sqrt(4 / np.sqrt(d) - d - 3))
    omega_0 = 0.5 * (8 + 4 * beta_1 + 4 * np.sqrt(4 * beta_1 + beta_1 ** 2)) ** (1 / 3) \
              + 2 * (8 + 4 * beta_1 + 4 * np.sqrt(4 * beta_1 + beta_1 ** 2)) ** (-1 / 3) - 1
    omega_min = max(omega_0, omega_1)
    # the upper bound
    omega_2 = np.sqrt(-1 + np.sqrt(2 * (beta_2 - 1)))
    omega_max = omega_2

    def fun_solve(x):
        m = -2 + np.sqrt(4 + 2 * (x ** 2 - (beta_2 + 3) / (x ** 2 + 2 * x + 3)))
        ans = (x - 1 - m) * ((x + 2 + m / 2) ** 2)
        return ans - beta_1

    init = np.array((omega_max - omega_min) / 2 + omega_min, dtype="float")
    omega_root = fsolve(fun_solve, init)[0]
    omega = max(omega_min, omega_root)
    m = -2 + np.sqrt(4 + 2 * (omega ** 2 - (beta_2 + 3) / (omega ** 2 + 2 * omega + 3)))
    Omega = - np.sign(kappa_3) * np.arcsinh(np.sqrt((omega + 1) / (2 * omega) * ((omega - 1) / m - 1)))

    delta_J = 1 / np.sqrt(np.log(omega))
    gamma_J = Omega * delta_J
    lambda_J = np.sqrt(kappa_2) / (omega - 1) * np.sqrt(2 * m / (omega + 1))
    xi_J = kappa_1 + lambda_J * np.sqrt(omega) * np.sinh(gamma_J / delta_J)

    Z = norm.ppf(1 - alpha)
    VaR = (xi_J + lambda_J * (np.exp((Z - gamma_J) / delta_J) - np.exp((gamma_J - Z) / delta_J)) / 2)
    L0 = VaR
    K = gamma_J + delta_J * np.log(((L0 - xi_J) / lambda_J) + np.sqrt(((L0 - xi_J) / lambda_J) ** 2 + 1))
    ES = (xi_J * (1 - norm.cdf(K)) +
          lambda_J / 2 * np.exp(- gamma_J / delta_J + 1 / (2 * delta_J ** 2)) * (1 - norm.cdf(K - 1 / delta_J))
          - lambda_J / 2 * np.exp(gamma_J / delta_J + 1 / (2 * delta_J ** 2)) * (
                  1 - norm.cdf(K + 1 / delta_J))) * (1 / alpha)
    return VaR, ES


alpha = 0.001
start_time = time.time()
a, b = coefficients(kappa_1, kappa_2, kappa_3, kappa_4, alpha)
print(a, b)
print("--- %s seconds in calculation VaR of ISMC---" % (time.time() - start_time))
alpha = 0.01
start_time = time.time()
a, b = coefficients(kappa_1, kappa_2, kappa_3, kappa_4, alpha)
print(a, b)
print("--- %s seconds in calculation VaR of ISMC---" % (time.time() - start_time))
alpha = 0.05
start_time = time.time()
a, b = coefficients(kappa_1, kappa_2, kappa_3, kappa_4, alpha)
print(a, b)
print("--- %s seconds in calculation VaR of ISMC---" % (time.time() - start_time))
alpha = 0.1
start_time = time.time()
a, b = coefficients(kappa_1, kappa_2, kappa_3, kappa_4, alpha)
print(a, b)
print("--- %s seconds in calculation VaR of ISMC---" % (time.time() - start_time))
