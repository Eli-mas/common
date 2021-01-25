from abc import ABC, abstractmethod

class CallTrackerBase:
	"""
	Base class for classes that track the calls made to functions.
	
	Use the `count` method as a decorator to mark that a callable
	should be tracked by an instance of this class. E.g.
		>>> c = CallCounter()
		... @c.count
		... def f(...):
		...	    ...
	
	To get the number of calls made thus far, use item access
	directly on the decorated function (raises a KeyError if not tracked
	by this instance):
		>>> num_calls = c[f]
	
	To get the original function back and stop counting it (and
	remove its count data), use the `pop` method. The return value
	of `pop` is a tuple (original function, # calls made). The sole
	argument is the decorated function:
		>>> original_f, num_calls = c.pop(f)
	
	You cannot call the decorated function after this; attempts to
	do so will raise a KeyError.
	
	Note, if you lose access to an instance of this class tracking
	a decorated function, but keep calling that function, the instance
	will still be spending resources tracking it. For this reason,
	on the decorated function, the '_original' attribute contains
	the original reference function.
	"""
	
	def __init__(self):
		self.data = {}
		self.original_to_decorated = {}
		self.decorated_to_original = {}
	
	def __getitem__(self, decorated):
		"""Return the # of calls made to a decorated function.
		This function receives the decorated function, not the
		original."""
		return self.data[self.__resolve(decorated)]
		
	
	def __resolve(self, decorated):
		# these try-except blocks allow for arbitarily deep stacking
		# of decorators from this class
		original = decorated
		try:
			while True:
				try:
					self.data[decorated]
					break
				except KeyError:
					decorated = decorated._original
		except AttributeError:
			raise KeyError(
				f"{original} is not tracked by this instance, nor is"
				" an antecendent of it"
			)
		return decorated
		
	
	def pop(self, decorated):
		"""Return (original function, # of calls made to decorated function).
		Also clear internal data related to the function.
		This function receives the decorated function, not the
		original."""
		decorated = self.__resolve(decorated)
		original = self.decorated_to_original.pop(decorated)
		self.original_to_decorated.pop(original)
		return (
			original,
			self.data.pop(decorated)
		)
	
	def __del__(self):
		self.data.clear()
		self.original_to_decorated.clear()
		self.decorated_to_original.clear()
		del self.data, self.original_to_decorated, self.decorated_to_original
	
	@abstractmethod
	def track(self, func):
		"""Override this to control what information is tracked and
		how this information is handled when a function is called.
		This should be implemented as a decorator on the original
		function `func`."""
		pass

class CallCounterBase(ABC, CallTrackerBase):
	"""Base class for classes that aggregate a count over successive calls
	made to individual functions. The count can draw on any parameters
	passed to those functions; its behavior is controlled by the abstract
	method `_eval`."""
	
	@abstractmethod
	def _eval(self, f, a, kw):
		"""Use this to determine what information contributes to the count
		for a given function. `a` and `kw` represent the arguments (as a tuple)
		and keywords (as a dictionary) passed to the function. The original
		function is given by `f`."""
		pass
	
	def count(self, func):
		"""Decorator: apply this to a function to begin tracking its calls.
		Applying this to a function already being tracked by this instance
		(the original or the decorated version) has no effect."""
		if func in self.decorated_to_original:
			return func
		
		inner = self.original_to_decorated.get(func)
		
		if inner is None:
			def inner(*a, **kw):
				# it will be left to the reader to understand why
				# this works with Python's scoping rules
				self.data[inner] += self._eval(func, a, kw)
				return func(*a ,**kw)
			
			self.data[inner] = 0
			self.original_to_decorated[func] = inner
			self.decorated_to_original[inner] = func
		
		inner._original = func
		return inner
	
	track = count

class CallCounter(CallCounterBase):
	"""Counts the number of calls made to a given callable.
	
	Example usage:
		>>> c = CallCounter()
		>>> @c.count
		... def f(): pass
		>>> c[f] # 0
		>>> f()
		>>> c[f] # 1
		>>> original_f, count = c.pop(f)
		>>> f() # KeyError
		>>> c[f] # KeyError
		>>> count # 1
	
	Using the `_original` attribute:
		>>> @c.count
		... def g(): pass
		>>> del c # leaves dangling references to c's internal dicts in memory
		>>> g = g._original # now that data can be garbage collected
	"""
	def _eval(self, f, a, kw):
		"""Increment the internal counter for a function by 1.
		This happend regardless of what parameters (if any) the function
		receives on each call. Information about what parameters were passed
		is not retained.
		
		NOTE: The signature `_eval(self, f, a, kw)` must be retained."""
		return 1

class ArgumentCounter(CallCounterBase):
	"""Counts the number of variadic arguments passed across function calls."""
	def _eval(self, f, a, kw):
		return len(a)

class VariadicArgumentCounter(CallCounterBase):
	"""Counts the number of variadic arguments passed across function calls."""
	def _eval(self, f, a, kw):
		NotImplemented

__all__ = ('CallCounter', 'ArgumentCounter')