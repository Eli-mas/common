class Functive:
	"""
	Useful for removing simple lambda statements from expressions, e.g.
		lambda v: v+1
		lambda obj: obj.attr
		lambda v
	
	Setting f = Functive(), you can use this as
		map(lambda v: v+1, sequence) --> map(f+1, sequence)
		map(lambda v: v>1, sequence) --> map(f>1, sequence)
		map(lambda v: v==1, sequence) --> map(f==1, sequence)
		map(lambda v: v[0], sequence) --> map(f[0])
		map(lambda v: v[1:3], sequence) --> map(f[1:3])
	
	So, e.g.,
		>>> f = Functive() # singleton instance
		>>> list(map( f*100, range(5) )) # [0, 100, 200, 300, 400]
		>>> list(map( 1/f, range(1, 5) )) # [1.0, 0.5, 0.3333333333333333, 0.25]
		>>> list(map( round(f,0), [0.1, 1.1, 2.1, 3.1])) # [0.0, 1.0, 2.0, 3.0]
	
	Non-usage:
		map(bool(f), sequence): use map(bool, sequence)
	
	Currently does not handle compound lambda statements, such as
		lambda v: 1 + v/2
	But this is something I might implement.
	"""
	_instance = None
	def __new__(cls):
		if cls._instance is None:
			cls._instance = super().__new__(cls)
		return cls._instance
	
# 	def __abs__(self):
# 		return lambda v: v.__abs__()
	
	def __add__(self, value):
		return lambda v: v.__add__(value)
	
	def __and__(self, value):
		return lambda v: v.__and__(value)
	
# 	def __bool__(self):
# 		return lambda v: v.__bool__()
	
	def __ceil__(self):
		return lambda v: v.__ceil__()
	
# 	def __class__(self):
# 		return lambda v: v.__class__()
	
# 	def __delattr__(self, *value):
# 		return lambda v: v.__delattr__(*value)
	
# 	def __dir__(self):
# 		return lambda v: v.__dir__()
	
	def __divmod__(self, value):
		return lambda v: v.__divmod__(value)
	
# 	def __doc__(self, value):
# 		return lambda v: v.__doc__(value)
	
	def __eq__(self, value):
		return lambda v: v.__eq__(value)
	
	def __float__(self):
		return lambda v: v.__float__()
	
	def __floor__(self):
		return lambda v: v.__floor__()
	
	def __floordiv__(self, value):
		return lambda v: v.__floordiv__(value)
	
# 	def __format__(self, value):
# 		return lambda v: v.__format__(value)
	
	def __ge__(self, value):
		return lambda v: v.__ge__(value)
	
	def __getattribute__(self, value):
		return lambda v: v.__getattribute__(value)
	
	def __getitem__(self, *a):
		return lambda v: v.__getitem__(*a)
	
# 	def __getnewargs__(self):
# 		return lambda v: v.__getnewargs__()
	
	def __gt__(self, value):
		return lambda v: v.__gt__(value)
	
# 	def __hash__(self):
# 		return lambda v: v.__hash__()
	
# 	def __index__(self):
# 		return lambda v: v.__index__()
	
# 	def __init__(self, value):
# 		return lambda v: v.__init__(value)
# 	
# 	def __init_subclass__(self, value):
# 		return lambda v: v.__init_subclass__(value)
	
	def __int__(self):
		return lambda v: v.__int__()
	
	def __invert__(self):
		return lambda v: v.__invert__()
	
	def __le__(self, value):
		return lambda v: v.__le__(value)
	
	def __lshift__(self, value):
		return lambda v: v.__lshift__(value)
	
	def __lt__(self, value):
		return lambda v: v.__lt__(value)
	
	def __mod__(self, value):
		return lambda v: v.__mod__(value)
	
	def __mul__(self, value):
		return lambda v: v.__mul__(value)
	
	def __ne__(self, value):
		return lambda v: v.__ne__(value)
	
	def __neg__(self, value):
		return lambda v: v.__neg__(value)
	
# 	def __new__(cls, *a, **kw):
# 		return lambda v: v.__new__(*a, **kw)
	
	def __or__(self, value):
		return lambda v: v.__or__(value)
	
	def __pos__(self, value):
		return lambda v: v.__pos__(value)
	
	def __pow__(self, value):
		return lambda v: v.__pow__(value)
	
	def __radd__(self, value):
		return lambda v: v.__radd__(value)
	
	def __rand__(self, value):
		return lambda v: v.__rand__(value)
	
	def __rdivmod__(self, value):
		return lambda v: v.__rdivmod__(value)
	
# 	def __reduce__(self, value):
# 		return lambda v: v.__reduce__(value)
	
# 	def __reduce_ex__(self, value):
# 		return lambda v: v.__reduce_ex__(value)
	
# 	def __repr__(self):
# 		return lambda v: v.__repr__()
	
	def __rfloordiv__(self, value):
		return lambda v: v.__rfloordiv__(value)
	
	def __rlshift__(self, value):
		return lambda v: v.__rlshift__(value)
	
	def __rmod__(self, value):
		return lambda v: v.__rmod__(value)
	
	def __rmul__(self, value):
		return lambda v: v.__rmul__(value)
	
	def __ror__(self, value):
		return lambda v: v.__ror__(value)
	
	def __round__(self, value):
		return lambda v: v.__round__(value)
	
	def __rpow__(self, value):
		return lambda v: v.__rpow__(value)
	
	def __rrshift__(self, value):
		return lambda v: v.__rrshift__(value)
	
	def __rshift__(self, value):
		return lambda v: v.__rshift__(value)
	
	def __rsub__(self, value):
		return lambda v: v.__rsub__(value)
	
	def __rtruediv__(self, value):
		return lambda v: v.__rtruediv__(value)
	
	def __rxor__(self, value):
		return lambda v: v.__rxor__(value)
	
	def __setattr__(self, attr, value):
		return lambda v: v.__setattr__(attr, value)
	
# 	def __sizeof__(self):
# 		return lambda v: v.__sizeof__()
	
# 	def __str__(self):
# 		return lambda v: v.__str__()
	
	def __sub__(self, value):
		return lambda v: v.__sub__(value)
	
# 	def __subclasshook__(self):
# 		return lambda v: v.__subclasshook__(value)
	
	def __truediv__(self, value):
		return lambda v: v.__truediv__(value)
	
	def __trunc__(self, value):
		return lambda v: v.__trunc__(value)
	
	def __xor__(self, value):
		return lambda v: v.__xor__(value)
