__getter_doc__ = \
"""Call this decorator on a class object to make it a directed getter:
the chain of calls specified in `name_to_callable_mapping` implicity
form a directed acyclic graph that allow the values to be computed on
the spot without having to specify a fixed order of computing the
variables.

E.g. say we have this:
>>> def attr1(self): return self.attr2 + 1
>>> def attr2(self): return 1

We can use this on a class as follows:
>>> @Getter('attr1': attr1, 'attr2': attr2)
>>> class A: pass

And we can now call:
>>> a.attr1 # prints '2'

without having to explicitly instantiate a.attr2 first.

The functionality provided also automatically detects recursions
if the `detect_recursion` parameter is set to True (False by default).
For example if we had defined this:
>>> def attr3(self): return self.attr4
>>> def attr4(self): return self.attr3
>>> @Getter('attr3': attr4, 'attr4': attr3)
>>> class A: pass

we would get these errors upon trying to get either of these attributes:
>>> a.attr3 # ImplicitRecursionError('unresolved recursion: attr3->attr4->attr3')
>>> a.attr4 # ImplicitRecursionError('unresolved recursion: attr4->attr3->attr4')

where 'ImplicitRecursionError' is a type defined at the module level.

Note if a recursion exists in the code but detect_recursion=False,
you will get Python's standard RecursionError if it is called.
"""

__Getter_doc__ = \
"""Getter: wrapper that allows for making a class into a directed getter,
where attributes not yet defined on an instance can be set dynamically
without hassle."""
# make this accessible at module level
__Getter_doc__ = __doc__ = \
	f'{__Getter_doc__}\n\n__doc__ for getter wrapper:<<{__getter_doc__}\n>>'



class ImplicitRecursionError(Exception):
	"""Used in Getter-decorated classes to alert to infinite recursions"""
	def __init__(self, attr, stack):
		self.str = self.__class__.format_str(attr, stack)
	
	def __repr__(self):
		return f'{self.__class__.__name__}(unresolved recursion: {self.str})'
	
	def __str__(self):
		return self.str
	
	@staticmethod
	def format_str(attr, stack):
		return f"{'->'.join(stack)}->{attr}"

def __getattr_plain__(self, attr):
	"""__getattr__ implementation that does not check for infinite recursion"""
	# make sure we can call this attribute
	try:
		caller = self.__getters__[attr]
	except KeyError:
		raise AttributeError(f"{self.__class__} has no attribute '{attr}'")
	
	# get the value and set it so that future accesses on this attribute
	# return immediately
	self.setattr(self, attr, self.__values__.setdefault(attr, (self)))
	
	# attribute is now set on this instance, return it
	return getattr(self, attr)

def __getattr_rec__(self, attr):
	"""__getattr__ implementation that can detect an infinite recursion"""
	# To detect recursion we need a call stack
	stack, _set = self.__getter_stack__
	if attr in _set:
# 		print('__getattr_rec__: attr, stack:',attr, stack)
		error = ImplicitRecursionError(attr, stack)
		# Cleanup: there should be no trace of the problematic call left over.
		# This way, one could theoretically catch a ImplicitRecursionError dynamically
		# without having to clean the attribute stack oneself.
		self.__getter_stack__ = ([], set(()))
		raise error
	
	# make sure we can call this attribute
	try:
		caller = self.__getters__[attr]
	except KeyError:
		raise AttributeError(f"{self.__class__} has no attribute '{attr}'")
	
	# we're clear to get the attribute, now we can modify the stack
	
	# track the attribute access before attempting to access it
	# this way it is on the stack as it is called on
	_set.add(attr)
	stack.append(attr)
	
	# make the call and set the attribute
	v = self.__values__.setdefault(attr, caller(self))
	setattr(self, attr, v)
	
	# call is complete, pop from the stack
	_set.remove(attr)
	stack.pop()
	
	return v

def Getter(name_to_callable_mapping, detect_recursion=False):
	# see top of this document for documentation
	def make_getter(class_object):
		# the mapping that tells how to get each particular attribute
		class_object.__getters__ = name_to_callable_mapping
		
		# the method that directs the class to __getters__
		# choose whether or not to detect recursion
		class_object.__getattr__ = \
			__getattr_rec__ if detect_recursion else __getattr_plain__
		
		# whatever the initially defined __init__ method was has to be replaced
		# catch the original object (slot wrapper or function) in a variable
		# and define a new function that calls it when called to initialize
		# an instance of the class
		i = class_object.__init__
		
		# intialize depends on recursion detection: if no check in place,
		# we do not set a '__getter_stack__' attribute
		if detect_recursion:
			def __init__(self):
				"""Initialize self and set attributes to track dynamic attribute setting"""
				# initial __init__ comes first
				i(self)
				# __values__ holds computed results
				self.__values__ = {}
				# used to detect recursion
				self.__getter_stack__ = ([], set(()))
		else:
			def __init__(self):
				"""Initialize self and set attributes to track dynamic attribute setting"""
				# initial __init__ comes first
				i(self)
				# __values__ holds computed results
				self.__values__ = {}
		
		class_object.__init__ = __init__
		
		return class_object
	
	# add documentation to this wrapper, just in case one looks here for it
	make_getter.__doc__ = __getter_doc__
	
	return make_getter

# put the getter documentation in the Getter decorator for easy reference
Getter.__doc__ = __Getter_doc__

__all__ = ('ImplicitRecursionError', 'Getter')

if __name__ == '__main__':
	from collections import deque
	
	@Getter({
		#non-loop
		'a': lambda self: self.b+1,
		'b': lambda self: 3,
		
		#recursive loop
		'c': lambda self: self.d,
		'd': lambda self: self.c,
		
		#recursive loop
		'r1': lambda self: self.r2,
		'r2': lambda self: self.r3,
		'r3': lambda self: self.r4,
		'r4': lambda self: self.r1,
		
		#non-loop
		'v1': lambda self: self.v2+1,
		'v2': lambda self: self.v3+1,
		'v3': lambda self: self.v4+1,
		'v4': lambda self: 1,
		
		#loop occurs at z2
		'z0': lambda self: self.z1+1,
		'z1': lambda self: self.z2+1,
		'z2': lambda self: self.z3+1,
		'z3': lambda self: self.z4+1,
		'z4': lambda self: self.z5+1,
		'z5': lambda self: self.z2+1,
		
		# fibonacci stack -- if calling f19 before any of the others
		# runs in any reasonable amount of time,
		# we know that values are being properly stored and accessed dynamically
		'f0': lambda self: 0,
		'f1': lambda self: 1,
		'f2': lambda self: self.f1 + self.f0,
		'f3': lambda self: self.f2 + self.f1,
		'f4': lambda self: self.f3 + self.f2,
		'f5': lambda self: self.f4 + self.f3,
		'f6': lambda self: self.f5 + self.f4,
		'f7': lambda self: self.f6 + self.f5,
		'f8': lambda self: self.f7 + self.f6,
		'f9': lambda self: self.f8 + self.f7,
		'f10': lambda self: self.f9 + self.f8,
		'f11': lambda self: self.f10 + self.f9,
		'f12': lambda self: self.f11 + self.f10,
		'f13': lambda self: self.f12 + self.f11,
		'f14': lambda self: self.f13 + self.f12,
		'f15': lambda self: self.f14 + self.f13,
		'f16': lambda self: self.f15 + self.f14,
		'f17': lambda self: self.f16 + self.f15,
		'f18': lambda self: self.f17 + self.f16,
		'f19': lambda self: self.f18 + self.f17

	}, True)
	class A: pass
	
	def _catch_recursion(Aobj, attr, stack, pre):
		try:
			print(getattr(Aobj, attr))
			# that shouldn't have worked
			raise RuntimeError(f'recursion not detected: attr {attr}, stack {stack}')
		except ImplicitRecursionError as e:
			# make sure the error got the order of attributes correct
			assert str(e) == e.format_str(stack[0], (*pre, *stack)), \
				f'stack={stack}, attr: {attr}, message: {str(e)}, ' \
				f'expected: {e.format_str(stack[0], (*pre, *stack))}'
# 			print(e)
		# cleanup: there should be no trace of the problematic call left over
		assert not any(map(Aobj.__values__.__contains__, stack))
	
	def catch_recursions(Aobj, stack, rec=0):
		pre, stack = deque(stack[:rec]), deque(stack[rec:])
		for _ in range(len(pre)):
			# attribute stack should consist of
			# every element of `pre` from the current one
			# and all of `stack`
			_catch_recursion(Aobj, pre[0], stack, pre)
			pre.popleft()
		for _ in range(len(stack)):
			# `pre` is not part of the call stack here
			# only `stack` is involved
			_catch_recursion(Aobj, stack[0], stack, ())
			stack.append(stack.popleft())
	
	def verify_stack(Aobj, stack, values):
		stack = deque(stack)
		values = deque(values)
		assert getattr(Aobj, stack[0]) == values[0]
		for n,v in zip(reversed(stack), reversed(values)):
			r = Aobj.__values__[n]
			assert r == v, f'__values__: {Aobj.__values__}, attr: {n}, expected {v}, got {r}'
		for _ in range(len(stack)):
			v = values.popleft()
			n = stack.popleft()
			r = getattr(Aobj, n)
			assert r == v, f'__dict__: {Aobj.__dict__}, attr: {n}, expected {v}, got {r}'
	
	a = A()
	verify_stack(a, ('a', 'b'), (4, 3))
	catch_recursions(a, ('c', 'd'))
	catch_recursions(a, ('r1', 'r2', 'r3', 'r4'))
	catch_recursions(a, ('z0', 'z1', 'z2', 'z3', 'z4', 'z5'), 2)
	verify_stack(a, ('v1', 'v2', 'v3', 'v4'), (4,3,2,1))
	
	
	
	# memoized fibonacci-value function
	fibcache={0:0, 1:1}
	def fib(i):
		if i==0: return 0
		if i==1: return 1
		result = fib(i-1) + fib(i-2)
		fibcache[i] = result
		return result
	
	frange = range(19, -1, -1)
	fib(19)
	values = [fibcache[i] for i in frange]
	fibcache.clear()
	fibcache[0], fibcache[1] = 0, 1
	del fib, fibcache # these are not being used in following call
	verify_stack(a, (f'f{i}' for i in frange), values)