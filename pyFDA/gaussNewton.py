import numpy as np

class GaussNewton(object):

	def __init__(self,y,x,thetaInit,residual,partials,weights=None,ridge=None):
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
			function defining the residual, r(y,x,theta), should be of the form y - f(x,theta)
		partials: list of functions
		functions defining the partial derivative of the residual for each parameter, p(x,theta). len(partials) = p"""
		self.y = y
		self.x = x
		self.n = y.shape[0]
		assert x.shape[0] == self.n, "x and y must be same size!"

		self.iteration = 0
		self.thetaInit = self.thetaCurrent = thetaInit
		self.thetaHistory = [self.thetaInit]
		self.p = len(partials)
		self.residual = residual
		self.partials = partials
		self.deltas = []
		# assert len(partials) == self.p, "must provide partial function for each parameter!"

		self.weights = weights
		self.ridge = ridge

	def jacobian(self,):
		return np.array([[self.partials[j](self.x[i,:],self.thetaCurrent)[0] for j in range(self.p)] for i in range(self.n)])

	def resid(self,):
		return np.array([self.residual(self.y[i],self.x[i,:],self.thetaCurrent)[0] for i in range(self.n)])

	def SRS(self,):
		return np.sum(self.resid()**2)

	def run(self,iterations = None):
		if iterations is None:
			iterations = 1
		for i in range(iterations):
			self._iteration()

	def w(self):
		w = np.eye(self.n)
		if not self.weights is None:
			w = np.diag(self.weights)

		return w

	def _iteration(self,):

		w = np.eye(self.n)
		if not self.weights is None:
			w = np.diag(self.weights)

		ridge = 0
		if not self.ridge is None:
			ridge = self.ridge
		ridge = np.eye(self.p)*ridge

		J = self.jacobian()
		proj = np.dot(np.linalg.inv(np.dot(J.T,J)+ridge),J.T)
		r = np.dot(w,self.resid())
		
		temp = np.zeros(self.thetaCurrent.shape[0]) 
		temp[:self.p] = temp[:self.p] + self.thetaCurrent[:self.p]
		temp[:self.p] = temp[:self.p] + np.dot(proj,r)
		self.thetaCurrent = temp
		# print self.thetaCurrent, temp, np.dot(proj,r)
		# self.thetaCurrent = self.thetaCurrent + np.dot(proj,r)

		self.deltas.append(np.dot(proj,r))
		
		self.thetaHistory.append(self.thetaCurrent)
		self.iteration += 1