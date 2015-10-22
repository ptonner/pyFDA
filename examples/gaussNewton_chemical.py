"""
Example of Gauss Newton method on chemical production rate equation.

Data and details are from the Gauss Netwon wikipedia page.

Author: Peter Tonner
"""

import numpy as np
import matplotlib.pyplot as plt
from pyFDA.gaussNewton import GaussNewton

S = np.array([0.038,0.194,.425,.626,1.253,2.500,3.740])[:,None]
rate = np.array([0.050,0.127,0.094,0.2122,0.2729,0.2665,0.3317])

def f(x,theta):
	return (theta[0]*x)/(x+theta[1])

def resid(y,x,theta):
	return y - theta[0]*x/(x+theta[1])

def partial1(x,theta):
	return x/(theta[1]+x)

def partial2(x,theta):
	return -theta[0]*x/(theta[1]+x)**2

thetaInit = np.array([.1,.6])
xpred = np.linspace(.1,4)
yhatInit = f(xpred,thetaInit)

gn = GaussNewton(rate,S,thetaInit,resid,[partial1,partial2])
print "Initial error: %.5lf" % gn.SRS()

for i in range(5):
	gn.run()
	print "Iteration %d error: %.5lf" % (i,gn.SRS())

yhatFinal = f(xpred,gn.thetaCurrent)
print gn.thetaCurrent

plt.plot(xpred,yhatInit,label="initial estimate")
plt.plot(xpred,yhatFinal,label="final estimate")
plt.scatter(S,rate,label="raw data")

plt.legend(loc="best")
plt.show()