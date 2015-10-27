from function import Function
from scipy import interpolate

class Bspline(Function):

	def __init__(self,x,y,*args,**kwargs):
		Function.__init__(self,"B-spline",x,y,*args,**kwargs)

	def _fit(self,x,y,*args,**kwargs):
		self.knots,self.coeff,self.degree = interpolate.splrep(x,y,*args,**kwargs)

	def predict(self,x,deriv=None,*args,**kwargs):
		if deriv is None:
			deriv = 0
		if deriv == -1:
			return interpolate.splev(x,interpolate.splantider((self.knots,self.coeff,self.degree)))
		return interpolate.splev(x,(self.knots,self.coeff,self.degree),der=deriv)