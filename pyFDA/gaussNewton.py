import numpy as np

class GaussNewton(object):

	def __init__(self,y,x,thetaInit,residual,partials):
		"""Minimized the residual sum of squares for a function f using the Gauss Newton method.

		Parameters:
		-----------
		y : array_like
			array of functions values, shape (n,1)
		x : array_like
			array of independent values, shape (n,k)
		thetaInit : array_like
			array of initial parameter estimates (p,1)
		residual : function
			function defining the residual, r(y,x,theta)
		partials: list of functions
		functions defining the partial derivative of the residual for each parameter, p(x,theta). len(partials) = p"""
		self.y = y
		self.x = x
		self.n = y.shape[0]
		assert x.shape[0] == self.n, "x and y must be same size!"

		self.iteration = 0
		self.thetaInit = self.thetaCurrent = thetaInit
		self.p = thetaInit.shape[0]
		self.residual = residual
		self.partials = partials
		assert len(partials) == self.p, "must provide partial function for each parameter!"

	def jacobian(self,):
		return np.array([[self.partials[j](self.y[i],self.x[i,:],self.thetaCurrent)[0] for j in range(self.p)] for i in range(self.n)])

	def resid(self,):
		return np.array([self.residual(self.y[i],self.x[i,:],self.thetaCurrent)[0] for i in range(self.n)])

	def SRS(self,):
		return np.sum(self.resid()**2)

	def run(self,iterations = None):
		if iterations is None:
			iterations = 1
		for i in range(iterations):
			self._iteration()

	def _iteration(self,):

		J = self.jacobian()
		proj = np.dot(np.linalg.inv(np.dot(J.T,J)),J.T)
		r = self.resid()
		self.thetaCurrent = self.thetaCurrent + np.dot(proj,r)
		
		self.iteration += 1