# Extended-Kalman-Filter-
Using filtering to estimate an unknown system parameter of a dynamical system

The dynamical system given by
$x_{k+1} = ax_k + \epsilon_k$
$y_k = \sqrt{(x_k)^2  + 1} + \nu_k$
where $x_k, y_k $∈ R are scalars, $ε_k$ ∼ N(0, 1) and $ν_k$ ∼ N(0, 1/2) are zero-mean scalar Gaussian noise uncorrelated across time k. The constant a is unknown and its value is estimated using this filter.

First dataset D is simulated with a = −1 for 100 observations. This is the ground-truth value of a that we would like to estimate. 
Then the EKF equations are developed that will use the collected dataset D to estimate the constant a. 

 ![image](https://user-images.githubusercontent.com/68454938/224437594-ac08337b-77d3-4add-86d8-6e65b10c33aa.png)
