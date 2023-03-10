import numpy as np
import matplotlib.pyplot as plt

N = 100 # No. of observations
x_0 = np.random.normal(1, np.sqrt(2)) # initial state has mean 1 and variance 2

#ground-truth value of a that is to be estimated
a = -1 
y_0 = np.sqrt(x_0**2 + 1) + np.random.normal(0, np.sqrt(1/2))
#  dynamical system includes states predicted x and predicted observations y
x = np.zeros(N+1) 
y = np.zeros(N+1)
x[0] = x_0
y[0] = y_0
for k in range(N):
  # epsilon_k and nu_k are zero-mean scalar Gaussian noises uncorrelated across time k
  epsilon_k = np.random.normal(0, np.sqrt(1))
  nu_k = np.random.normal(0, np.sqrt(1/2)) 
  x[k+1] = a*x[k] + epsilon_k
  y[k] = np.sqrt(x[k+1]**2 + 1) + nu_k

# EKF equations that will use the collected dataset D = {yk : k = 1,...,} to estimate the constant a
R_t = np.identity((2))
R_t[1, 1] = 0

# Initial covariance Sigma_0|0 and initial mean mu
# Assuming they are not correlated
sigma_k = np.identity((2))   
mu_k = np.array([1, -1])
I = np.identity(2)
Q = 1/2
a_means = []
a_means.append(mu_k[1])
a_covariances = []
a_covariances.append(sigma_k[1,1])

# Covariance Sigma_k|k
for i in range(len(y)):
  # Propagation step
  mu_k_plus_1_k = np.array([mu_k[0] * mu_k[1], mu_k[1]])
  A = np.vstack((np.hstack((mu_k[1], mu_k[0])),np.array([0, 1])))
  cov_k_plus_1_k = np.matmul(np.matmul(A, sigma_k), (A.T)) + R_t

  # Update step
  # Jacobian
  C = np.array([mu_k_plus_1_k[0]/(np.sqrt(mu_k_plus_1_k[0]**2 + 1)), 0])
  
  # Kalman gain to incoporate the fake linear observation
  K = np.matmul(cov_k_plus_1_k, C.T) / (np.matmul(np.matmul(C, cov_k_plus_1_k), C.T) + Q)
  
  # Resubstituting the fake observation in terms of thr actual observation
  mu_k_plus_1_k_plus_1 = mu_k_plus_1_k + (K * (y[i] - np.sqrt(mu_k_plus_1_k[0]**2 + 1)))
  sigma_k_plus_1_k_plus_1 = np.matmul((I - np.matmul(K.reshape((2,1)),C.reshape((1,2)))),cov_k_plus_1_k)

  a_means.append(mu_k[1])
  a_covariances.append(sigma_k_plus_1_k_plus_1[1,1])
  sigma_k = sigma_k_plus_1_k_plus_1
  mu_k = mu_k_plus_1_k_plus_1

a_means = np.array(a_means)
a_covariances = np.array(a_covariances)

time = np.arange(N+2)
aa = -np.ones(time.shape)


plt.plot(time, aa, label='a = -1', color='deeppink')
plt.plot(time, a_means + np.sqrt(a_covariances), color='gold', label='$\mu_k + \sigma_k$')
plt.plot(time, a_means - np.sqrt(a_covariances), color='purple', label='$\mu_k - \sigma_k$')
plt.plot(time, a_means, label='$\mu_k$', color='royalblue')
plt.legend()
plt.xlabel('Time steps')
plt.ylabel('a') 
plt.title('Estimation of a')
plt.show()