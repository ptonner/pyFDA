import numpy as np
from .. import bspline, gaussNewton
from ..util import binarySearch_inverseMonotonic

def g(t,theta):
	return np.exp(theta[0]) *(t+theta[1])

class RegisterLocalRegression(object):

	def __init__(self,x,y,t,bandwidth=None,ridge=None,decay=True,variance=None):
		self.x = x
		self.xspline = bspline.Bspline(t,x)
		self.y = y
		self.yspline = bspline.Bspline(t,y)
		self.n = x.shape[0]
		self.t = t

		self.bandwidth = bandwidth
		self.ridge = ridge
		self.decay = decay
		self.variance = variance

		self.xhats = [self.xspline]
		self.ghats = []
		self._thetas = []
		self.error = [self.SRS()]

	def thetas(self,):
		return np.array(self._thetas)

	def partials(self,):
		return [self.partial1,self.partial2,self.partial3]

	def gaussNewton(self,timeInd,decay=None):
		partials = [self.partial1,self.partial2,self.partial3]

		variance = self.variance
		if variance is None:
			variance = 1

		if self.bandwidth is None:
			w = None
		else:
			if decay is None:
				decay = 1

			# w = 1 - variance * ((self.t - self.t[j])**2)/((self.bandwidth*decay)**2)
			w = variance * np.exp((-(self.t - self.t[timeInd])**2)/((self.bandwidth*decay)))
			# w = np.max((w,np.zeros(self.n)),0)

		# gn = gaussNewton.GaussNewton(self.y,self.xspline(self.t)[:,None],np.array([0,0,0]),self.residual,self.partials(),w,self.ridge)
		gn = gaussNewton.GaussNewton(self.y,self.t[:,None],np.array([0,0,0]),self.residual,self.partials(),w,self.ridge)

		return gn

	def run(self,iter=None):
		if iter is None:
			iter = 1

		partials = [self.partial1,self.partial2,self.partial3]
		xhat = self.xspline(self.t)
		self.deltas = []

		for i in range(iter):
			ghat = []
			self._thetas.append([])
			self.deltas.append([])
			for j in range(self.n):

				variance = self.variance
				if variance is None:
					variance = 1

				if self.bandwidth is None:
					w = None
				else:
					if self.decay:
						decay = 2**(-i)
					else:
						decay = 1

					# w = 1 - variance * ((self.t - self.t[j])**2)/((self.bandwidth*decay)**2)
					w = variance * np.exp((-(self.t - self.t[j])**2)/((self.bandwidth*decay)))
					# w = np.max((w,np.zeros(self.n)),0)

				# TEMP REMOVE!
				# w = 1 - ((self.t - self.t[j])**2)/((self.bandwidth)**2)

				# gn = gaussNewton.GaussNewton(self.y,xhat[:,None],np.array([0,0,0]),self.residual,partials,w,self.ridge)
				gn = gaussNewton.GaussNewton(self.y,self.xspline(self.t)[:,None],np.array([0,0,0]),self.residual,partials,w,self.ridge)

				gn = self.gaussNewton(j,decay=decay)
				gn.run()

				self._thetas[-1].append(gn.thetaCurrent)
				ghat.append(self.g(self.t[j],gn.thetaCurrent))
				self.deltas[-1].append(gn.deltas[-1])

			gspline = bspline.Bspline(self.t,ghat)
			xhat = self.xspline(gspline(self.t))
			self.xspline = bspline.Bspline(self.t,xhat)

			# uncomment to update x with the inverse of g, as described in the paper (is this correct?)
			# ginv = bspline.Bspline(ghat,self.t)
			# xhat = self.xspline(ginv(self.t))
			# self.xspline = bspline.Bspline(self.t,xhat)		
			
			self.xhats.append(self.xspline)
			self.ghats.append(gspline)

			self.error.append(self.SRS())

	def h(self,):
		htemp = self.ghats[0](self.t)
		for i in range(1,len(self.ghats)):
			htemp = self.ghats[i](htemp)

		return bspline.Bspline(self.t,htemp)

	def SRS(self,):
		if len(self._thetas) > 0:
			return np.sum((self.y-np.exp(self.thetas()[-1,:,2]) * self.xspline(self.t))**2)
		else:
			return np.sum((self.y- self.xspline(self.t))**2)

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