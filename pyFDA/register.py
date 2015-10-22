import numpy as np

class Register(object):

	def __init__(self,x,funcs,functionalModel,meanFunction):
		self.n = len(funcs)
		self.delta = np.zeros(n)
		self.funcs = funcs
		self.functionalModel = functionalModel
		self.meanFunction = meanFunction

	def compute(self,alpha=.1,tol=1e-1,maxiter=1000):
		z = 0
		diff = 1e10
		lastErr = 1e10
		mu = meanFunction(self.x,self.funcs,self.delta)

	    while diff > tol:

			for i in range(self.n):
				mu_pred = mu.predict(self.x)
				x_shift = self.func[i].predict(self.x-self.delta[i])
				mu_deriv = mu.predict(self.x,der=1)
				
				delta_d1 = np.sum((mu_pred-x_shift)*mu_deriv)
				delta_d2 = np.sum(mu_deriv**2)

				self.delta[i] -= alpha * delta_d1/delta_d2

			mu = meanFunction(time,self.func,self.delta)

			mu_pred = mu.predict(self.x)
			err = 0
			for i in range(self.n):
				x_shift = self.func[i].predict(self.x-self.delta[i])
				err += np.sum((mu_pred-x_shift)**2)
			diff = err - lastErr
			lastErr = err			

			z+=1
			if z > maxiter:
				break

		return err

