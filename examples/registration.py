"""
Example of function registration using a local regression framework

Method adapted from "Curve registration by local regression", Kneip et al., 2000

Author: Peter Tonner
"""

import numpy as np
import scipy.interpolate
from pyFDA import bspline

def binarySearch_inverseMonotonic(t0,t1,f,ft,i):
	"""Binary search on a monotone increasing function for the inverse of f at point t"""
	tm = (t0+t1)/2
	if i > 100:
		return tm
	f0 = f(t0) #scipy.interpolate.splev(t0,f)
	f1 = f(t1) #scipy.interpolate.splev(t1,f)
	if np.abs(f0-ft) < 1e-3:
		return t0
	if np.abs(f1-ft) < 1e-3:
		return t1
	fm = f(tm)#scipy.interpolate.splev(tm,f)
	if fm < ft:
		return binarySearch_inverseMonotonic(tm,t1,f,ft,i+1)
	else:
		return binarySearch_inverseMonotonic(t0,tm,f,ft,i+1)

def g(t,theta):
	return np.exp(theta[0]) *(t+theta[1])

def resid(y,t,theta):
	return y - np.exp(theta[2]) * xspline(g(t,theta))

def partial1(t,theta):
	return np.exp(theta[2]) * xspline(g(t,theta),deriv=1) * np.exp(theta[0]) * (t+theta[1])

def partial2(t,theta):
	return np.exp(theta[2]) * xspline(g(t,theta),deriv=1) * np.exp(theta[0])

def partial3(t,theta):
	return np.exp(theta[2]) * xspline(g(t,theta),deriv=0)


n = 200
t = np.linspace(0,6,200)
y = np.cos(t**3/(np.pi**2))

# define the registration for curve x
# hpoints = np.array([(0,0),(2,1),(4,3.75),(6,6)])
# hspline = bspline.Bspline(hpoints[:,0],hpoints[:,1])
# h = hspline(t)

theta1points = np.array([(0,0),(2,-.05),(4,-.03),(6,0)])
theta1 = bspline.Bspline(theta1points[:,0],theta1points[:,1])

theta2points = np.array([(0,0),(1,.2),(5,-.2),(6,0)])
theta2 = bspline.Bspline(theta2points[:,0],theta2points[:,1])

h = g(t,[theta1(t),0,0])
hspline = bspline.Bspline(t,h)
hinv = np.array([binarySearch_inverseMonotonic(0.,6.,hspline,z,0) for z in t])
hinvspline = bspline.Bspline(t,hinv)

# x = np.cos(hinv**3/(np.pi**2))
x = np.cos(hinvspline(t)**3/(np.pi**2))

# add some amplitude variation
ampPoints = np.array([(0,0),(1,.05),(2,-.2),(4,-.05),(5,0.05)])
amp = scipy.interpolate.splrep(ampPoints[:,0],ampPoints[:,1])

x = (1+scipy.interpolate.splev(t,amp)) * x

theta1points = .05*((t-3)**2 - 9)
theta1 = bspline.Bspline(t,theta1points)
h = g(t,[theta1(t),0,0])
hspline = bspline.Bspline(t,h)

x = np.cos(t**3/(np.pi**2))
y = np.cos(hspline(t)**3/(np.pi**2))

y = (t)/(1+t)
x = (g(t,[.1,0,0]))/(1+g(t,[.1,0,0]))



bandwidth = np.pi/2
variance = 1

xspline = bspline.Bspline(t,x)
xhats = [xspline]
ghats = []
thetas = []
for i in range(3):
	ghat = []
	thetas.append([])
	for j in range(n):
		decay = 2**(-i)
		w = 1 - variance * ((t - t[j])**2)/((bandwidth*decay)**2)
		w = np.max((w,np.zeros(x.shape[0])),0)

		# gn = gaussNewton.GaussNewton(y,x[:,None],np.array([0,0,0]),resid,[partial1,partial2,partial3],w,n/2)
		# gn = gaussNewton.GaussNewton(y,x[:,None],np.array([0,0,0]),resid,[partial1,partial2],w,n)
		gn = gaussNewton.GaussNewton(y,x[:,None],np.array([0,0,0]),resid,[partial1],w,1)
		# gn = gaussNewton.GaussNewton(y,x[:,None],np.array([0,0,0]),resid,[partial1],w)
		gn.run()
		thetas[-1].append(gn.thetaCurrent)

		ghat.append(g(t[j],gn.thetaCurrent))

	gspline = bspline.Bspline(t,ghat)
	ginv = np.array([binarySearch_inverseMonotonic(0.,6.,gspline,z,0) for z in t])
	ginvspline = bspline.Bspline(t,ginv)

	xhat = xspline(gspline(t))
	# xhat = xspline(ginvspline(t))

	xspline = bspline.Bspline(t,xhat)
	
	xhats.append(xspline)
	ghats.append(gspline)
thetas = np.array(thetas)
# 	plt.plot(t+t[j],g(t,gn.thetaCurrent))
# plt.show()