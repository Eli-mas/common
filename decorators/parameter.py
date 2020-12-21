"""Decorators for operating on parameters passed to a callable at runtime,
with the option of automatically replacing parameters based on specified rules.

Note, at the moment the decorators here cannot be stacked,
which I intend to fix.
"""

from inspect import getfullargspec
from collections import deque
import functools
from ..collections import consume

def assert_objects_are_of_specified_types(
	_types=None, objects=None, names=None, source=None,
):
	"""
	Given set of types and objects, verify that each object is an instance
		of the corresponding type provided.
	
	If names and source are both not None,
		'names' should be a mapping (dict) of those names to the required types,
		and 'source' should be an object with dict-like access that contains
		the names as keys.
		Note: if only one of these is provided, an error is raised.
	
	Else if these are not provided,
		if objects is None, '_types' is assumed to be a container
		each of whose elements is a pair <an object, the required type>.
	
	Otherwise, '_types' should be a list of types, and 'objects' a list of objects
	
	Note to self: this chain of logic might be a bit much for one function;
	it might be better to split into different functions, each one of which
	handles a different one of the above cases; control flow/rules might be
	confusing.
	"""
	
	if (source is None) ^ (names is None):
		# accept neither or both, but not one in isolation
		raise ValueError(
			"pass arguments to both 'source' and 'names' or neither; "
			f"you passed source={source}, names={names}"
		)
	
	if names is None:
		# if no names, at least indicate the index of each item
		names = ("the object at index %i"%i for i in range(len(_types)))
		if objects is None:
			objects, _types = zip(*_types) # _types was a container of (object,type) pairs
	
	else:
		names,_types = zip(*names.items())
		# `names` was a dict of <name:type> entries
		# now `names` is a tuple of name (str) objects
		# and type objects have been stored in _types
		objects = tuple(source[n] for n in names)
		names = tuple("'%s'"%n for n in names)
	
	# by the time we get here:
	#     `names` is an iterable of strings indicating parameter names
	#     `objects` is an iterable of objects to which each name is assigned
	#     `_types` is an iterable telling the expected type of each object
	for n,o,t in zip(names, objects, _types):
		assert isinstance(o,t), \
		f"the value passed for {n} is not of required type {t}: " \
		f"instead received this object of type {type(o)}: {o}" \

def validate_parameter_types(**map_names_types):
	"""
	A decorator meant to monitor input types to a function at runtime.
	'map_names_types' is a mapping <parameter name: required type>
	
	notes:
		would be nice to incorporate MultiIterator into this
		and make 'assert_objects_are_of_specified_types' compatible
		for efficiency
	"""
	for n,t in map_names_types.items():
		assert isinstance(t, type), \
			f'passed object of type {type(t)}, not {type} for name {n}'
	
	def wrapper_around_validate_and_call(func):
		"""
		Perform one-time verification that function defaults match expected
		types, then define the wrapper function that calls the intended
		function and verifies the types of dynamically passed arguments
		"""
		defaults, pos_arg = separate_required_and_default_parameters(func)
		
		# validate that default values are in line with expected types
		assert_objects_are_of_specified_types(names=map_names_types, source=defaults)
		
		def validate_and_call(*args,**kwargs):
			"""
			Call the target function whose default values have already been
			verified, and assert that passed values match expected types
			
			Varargs go unchecked, and varkwargs are ignored
			"""
			print(f"calling 'validate_and_call' on {func.__qualname__} with args={args} and kw={kwargs}")
			
			# map names both in default and positional parameters
			# first, map names passed as keywords to respective objects
			name_obj_map = {k:kwargs[k] for k in kwargs.keys() & map_names_types.keys()}
			name_obj_map.update({pos_arg[i]:args[i]
								 for i in range(min(len(args), len(pos_arg)))
								})
			# why take the minimum of the lengths in the above comprehension?
			#	if len(pos_arg)<len(args), then more args were passed then required args,
			#		so varargs was used, and variable arguments should be ignored
			#	if len(args)<len(pos_arg), required arguments are either missing
			#		(a ValueError will be raised), or the args were pass as keywords
			#		rather than args, in which case they will be caught by kwargs
			
			# assert that passed values correspond to expected types
			assert_objects_are_of_specified_types(names=map_names_types, source=name_obj_map)
			return func(*args,**kwargs)
		
		return validate_and_call
	
	return wrapper_around_validate_and_call

def validate_parameters(**map_from_names_to_callables):
	"""More general version of parameter validation, where instead of verifying
	input types, we verify parameters on the basis of arbitrary conditions."""
	NotImplemented
	# the structure will be the same as in 'validate_parameter_types'

def separate_required_and_default_parameters(function):
		"""
		return 'source', 'pos_arg'
		
		source = a mapping
			<parameter names: default parameter values>
			based on inspection of function spec
		
		pos_arg = a mapping
			<index of required parameter: name of parameter at that index>
		"""
		# get the argument specification for the function
		spec = getfullargspec(function)
		
		#if keywords have defaults, we want to know what they are
		source={}
		if spec.kwonlydefaults:
			# kwonlydefaults is a dict mapping parameter names to default values
			source.update(spec.kwonlydefaults)
		if spec.defaults:
			# if n defaults are listed, they correspond
			# to the last n names in the defaults list
			source.update(dict(zip(spec.args[-len(spec.defaults):],spec.defaults)))
		
		pos_arg = {i:arg_name for i,arg_name in enumerate(spec.args)}
		
		return source, pos_arg

def operate_on_params_dynamically(**operations):
	"""
	Operate on parameters passed to a function at runtime via the specified
	operations. Note, this does not modify which parameters are passed to
	the function; so the parameters will be unmodified unless they are
	mutated by the called actions.
	
	operations: mapping <parameter name: callable to perform on parameter>
	"""
	def decorator_func(function):
		defaults, pos_arg = separate_required_and_default_parameters(function)
		
		"""
		print(f"host function '{function.__qualname__}': defaults={defaults}\n\n")
		"""
		
		@functools.wraps(function)
		def wrapper_func(*args,**kwargs):
			"""
			print(f"calling function '{function.__qualname__}': args={args},kwargs={kwargs}")
			"""
			# capture default keywords first
			source = {k:defaults[k] for k in operations.keys() & defaults.keys()}
			# now add in dynamic keywords in case of override
			source.update({k:kwargs[k] for k in kwargs.keys() & operations.keys()})
			# why both here, not as in validate_parameter_types?
			# because here, it is possible to operate on any parameters dynamically,
			# whether they are specified dynamically or as defaults
			
			# update with positionally specified args
			source.update({pos_arg[i]:args[i] for i in range(min(len(args),len(pos_arg)))})
			
			op_iterator	= (op(source[name]) for name,op in operations.items())
			deque(op_iterator, maxlen=0) #https://docs.python.org/3/library/itertools.html#itertools-recipes
			
			return function(*args,**kwargs)
		
		return wrapper_func
	return decorator_func

def modify_parameters_dynamically(**operations):
	"""
	Decrator on a function that allows for passing modified parameters
	to a function at runtime in place of the original parameters passed.
	The parameters can be positional or keyword arguments.
	
	Usage:
	
	>>> @modify_parameters_dynamically(a=lambda p: p*2, b=len)
	>>> def f(a,b):
	>>> 	return a,b
	>>> 
	>>> f(1,range(2)) # (2, 2)
	>>> f([1], [1]) # ([1,1], 1)
	
	Can be useful in security contexts:
	>>> @modify_parameters_dynamically(user_input_str = assert_valid_input, ...)
	>>> def security_critical_function(user_input_str, ...):
	>>>     ...
	
	Also can be convenient for replacing default kwargs in a way that
	must be handled at runtime; for an example see the makeax function
	in this module.
	"""
	
	def decorator(function):
		defaults, pos_arg = separate_required_and_default_parameters(function)
# 		print(f"host function '{function.__qualname__}': defaults={defaults}, pos_arg={pos_arg}")
# 		print(f'function spec: {getfullargspec(function)}\n\n')
		
		@functools.wraps(function)
		def wrapper(*args,**kwargs):
# 			print(f"calling function '{function.__qualname__}': args={args},kwargs={kwargs}")
			source = kwargs
			
			# don't repeat parmaeter passes in *args and **kwargs
			arglim = min(len(args),len(pos_arg))
			argint = set(kwargs.keys()).intersection(pos_arg[i] for i in range(arglim))
			if argint:
				raise TypeError(
					f"{function.__qualname__}() got multiple values for argument "
					f"'{min(v for k,v in pos_arg.items() if v in argint)}'"
				)
			
			opkeys = operations.keys()
			opkeys_defaults = opkeys & defaults.keys()
			overriden_default_opkeys = kwargs.keys() & opkeys_defaults
			non_overriden_default_opkeys = opkeys_defaults - overriden_default_opkeys
# 			print('opkeys:',opkeys)
# 			print('opkeys_defaults:',opkeys_defaults)
# 			print('overriden_default_opkeys:',overriden_default_opkeys)
# 			print('non_overriden_default_opkeys:',non_overriden_default_opkeys)
			
			# capture default keywords first
			source.update({k:operations[k](defaults[k]) for k in non_overriden_default_opkeys})
			# now add in dynamic keywords in case of override
			source.update({k:operations[k](kwargs[k]) for k in overriden_default_opkeys})
			# update with positionally specified args
			consume(source.pop(pos_arg[i],None) for i in range(arglim))
# 			arg_inds = [i if i<len(pos_arg else )]
			# this could be made more efficient: pos_arg[i] is being called twice
			# Python 3.8 --> assignment expressions
			subargs = (
				operations[pos_arg[i]](a) if (pos_arg[i] in operations)
				else a
				for i,a in enumerate((args[p] for p in range(arglim)))
			)
			varargs = args[arglim:]
			
			return function(*subargs,*varargs, **source)
		return wrapper
	return decorator

def verify_and_modify_parameters(verifications, modifications):
	"""Combination of 'validate_parameters' and 'modify_parameters_dynamically'"""
	NotImplemented
	# can either be a combination of internal logic of above functions,
	# or simply a decorator stack, once I get that to work

__all__ = ('assert_objects_are_of_specified_types','validate_parameter_types',
'separate_required_and_default_parameters','operate_on_params_dynamically',
'modify_parameters_dynamically')

if __name__ == '__main__':
	from matplotlib import pyplot as plt
	
	#@modify_parameters_dynamically(x=lambda x: range(5))
	@makeax('ax')
	### stacking these doesn't work yet
	def plot_points(x=None,y=None,ax=None):
		if x is None:
			x = range(5)
		if y is None:
			lines = ax.lines
			if lines: y = lines[-1].get_ydata()+1
			else: y = range(5)
		ax.plot(x,y,marker='o',ls='none')
	
	for _ in range(4):
		plot_points()
	plt.show()