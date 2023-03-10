# Extended-Kalman-Filter-
Using filtering to estimate an unknown system parameter of a dynamical system

The dynamical system given by
xk+1 = axk + εk
y k = 􏰅 x 2k + 1 + ν k
 where xk, yk ∈ R are scalars, εk ∼ N(0, 1) and νk ∼ N(0, 1/2) are zero-mean scalar Gaussian noise uncorrelated across time k. The constant a is unknown and we would like to estimate its value. If we know that our initial state has mean 1 and variance 2
