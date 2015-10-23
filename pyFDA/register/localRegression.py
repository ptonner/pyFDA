import numpy as np
from .. import bspline, gaussNewton
from ..util import binarySearch_inverseMonotonic

def g(t,theta):
	return np.exp(theta[0]) *(t+theta[1])

class RegisterLocalRegression(object):

	def __init__(self,x,y,t,bandwidth=None,ridge=None,variance=None):
		self.x = x
		self.xspline = bspline.Bspline(t,x)
		self.y = y
		self.yspline = bspline.Bspline(t,y)
		self.n = x.shape[0]
		self.t = t

		self.bandwidth = bandwidth
		self.ridge = ridge
		self.variance = variance

		self.xhats = [self.xspline]
		self.ghats = []
		self._thetas = []

	def thetas(self,):
		return np.array(self._thetas)

	def run(self,iter=None):
		if iter is None:
			iter = 1

		partials = [self.partial1,self.partial2,self.partial3]
		xhat = self.xspline(self.t)

		for i in range(iter):
			ghat = []
			self._thetas.append([])
			for j in range(self.n):

				variance = self.variance
				if variance is None:
					variance = 1

				if self.bandwidth is None:
					w = None
				else:
					decay = 2**(-i)
					w = 1 - variance * ((self.t - self.t[j])**2)/((self.bandwidth*decay)**2)
					w = np.max((w,np.zeros(self.n)),0)

				gn = gaussNewton.GaussNewton(self.y,xhat[:,None],np.array([0,0,0]),self.residual,partials,w,self.ridge)
				gn.run()

				self._thetas[-1].append(gn.thetaCurrent)
				ghat.append(self.g(self.t[j],gn.thetaCurrent))

			gspline = bspline.Bspline(self.t,ghat)
			xhat = self.xspline(gspline(self.t))

			# ginv = np.array([binarySearch_inverseMonotonic(0.,6.,gspline,z,0) for z in self.t])
			# ginvspline = bspline.Bspline(t,ginv)
			# xhat = xspline(ginvspline(t))

			self.xspline = bspline.Bspline(self.t,xhat)
			
			self.xhats.append(self.xspline)
			self.ghats.append(gspline)

	def h(self,):
		htemp = self.ghats[0](self.t)
		for i in range(1,len(self.ghats)):
			htemp = self.ghats[i](htemp)

		return bspline.Bspline(self.t,htemp)


	def g(self,t,theta):
		return np.exp(theta[0]) *(t+theta[1])

	def residual(self,y,t,theta):
		return y - np.exp(theta[2]) * self.xspline(self.g(t,theta))

	def partial1(self,t,theta):
		return np.exp(theta[2]) * self.xspline(g(t,theta),deriv=1) * np.exp(theta[0]) * (t+theta[1])

	def partial2(self,t,theta):
		return np.exp(theta[2]) * self.xspline(g(t,theta),deriv=1) * np.exp(theta[0])

	def partial3(self,t,theta):
		return np.exp(theta[2]) * self.xspline(g(t,theta),deriv=0)