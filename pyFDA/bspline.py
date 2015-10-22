from function import Function
from scipy import interpolate

class Bspline(Function):

	def __init__(self,x,y,*args,**kwargs):
		Function.__init__(self,"B-spline",x,y,*args,**kwargs)

	def _fit(self,x,y,*args,**kwargs):
		self.knots,self.coeff,self.degree = interpolate.splev(x,y,*args,**kwargs)