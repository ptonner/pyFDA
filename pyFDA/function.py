

class Function(object):
	"""A wrapper for function representations in functional data analysis"""

	def __init__(self,name=None,x=None,y=None,*args,**kwargs):
		self.name = name
		self.fit(x,y,*args,**kwargs)

	def fit(self,x=None,y=None,*args,**kwargs):
		if x is None or y is None:
			return
		self._fit(x,y,*args,**kwargs)

	def _fit(self,x,y,*args,**kwargs):
		raise NotImplementedError("")

	def predict(self,x,*args,**kwargs):
		raise NotImplementedError("")

	def __call__(self,x,*args,**kwargs):
		return self.predict(x,*args,**kwargs)