"""At the moment"""

from inspect import getfullargspec
# from ..common.cls import MultiIterator
from collections import deque
import functools
from ..collections import consume

def assert_objects_are_of_specified_types(
	_types=None, objects=None, names=None, source=None,
	#raise_for_key=False
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
	handles a different one of the above cases
	"""
	#null = object() # used if 'names' and 'source' come into play
	
	if (source is None) ^ (names is None): # accept neither or both, but not one in isolation
		raise ValueError(
			"pass arguments to both 'source' and 'names' or neither; "
			f"you passed source={source}, names={names}"
		)
	
	if names is not None:
		names,_types = zip(*names.items()) # names was a dict of <name:type> entries
		"""if raise_for_key: # raise KeyError if name is missing from source
			objects = tuple(source[n] for n in names)
		else: # fill with a default value, null, when name not found
			objects = tuple(source.get(n,null) for n in names)"""
		objects = tuple(source[n] for n in names)
		names = tuple("'%s'"%n for n in names)
	
	else:
		# if no names, at least indicate the index of each item
		names = ("the object at index %i"%i for i in range(len(_types)))
		if objects is None:
			objects,_types = zip(*_types) # _types was a container of (object,type) pairs
	
	for n,o,t in zip(names, objects, _types):
		#if o is not null:
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
	for t in map_names_types:
		assert isinstance(t,type), f'passed object of type {type(t)}, not {type}'
	
	def validate_and_call_wrapper(func):
		"""
		perform one-time verification that function defaults match expected types,
		then define the wrapper function that calls the intended function
		and verifying the types of dynamically passed arguments
		"""
		defaults, pos_arg = separate_required_and_default_parameters(func)
		
		# validate that default values are in line with expected types
		assert_objects_are_of_specified_types(names=map_names_types,source=source)
		#for parameter_name,required_parameter_type in map_names_types.items():
		#	assert isinstance(source[parameter_name],required_parameter_type), \
		#		"validate_parameter_types: the default value for parameter " \
		#		f"{parameter_name}={source[parameter_name]} " \
		#		f"in {func.__qualname__} is not of required type {required_parameter_type}" \
		#		"; check the function's definition"
		
		def validate_and_call(*args,**kwargs):
			"""
			call the target function whose default values have already been verified,
			and assert that passed values match expected types
			
			configured to handle the case of varargs (which go unchecked)
			and varkwargs (keywords not defined in the original function are ignored)
			"""
			print(f"calling 'validate_and_call' on {func.__qualname__} with args={args} and kw={kwargs}")
			
			# map names both in default and non-default args
			name_obj_map = {k:kwargs[k] for k in kwargs.keys() & map_names_types.keys()} # mapping of names to respective objects
			name_obj_map.update({pos_arg[i]:args[i] for i in range(min(len(args),len(pos_arg)))})
			# why take the minimum of the lengths?
			#	if len(pos_arg)<len(args), then more args were passed then required args,
			#		so varargs was used, and variable arguments should be ignored
			#	if len(args)<len(pos_arg), required arguments are either missing
			#		(a ValueError will be raised), or the args were pass as keywords
			#		rather than args, in which case they will be caught by kwargs
			
			# assert that passed values correspond to expected types
			assert_objects_are_of_specified_types(names=map_names_types,source=name_obj_map)
			#for parameter_name,required_parameter_type in map_names_types.items():
			#	try:
			#		print(f'asserting isinstance({parameter_name}={kwargs[parameter_name]}, '
			#			  f'{required_parameter_type})')
			#		
			#		assert kwargs[parameter_name]==required_parameter_type, \
			#		"validate_parameter_types: the value passed for parameter " \
			#		f"{parameter_name}={spec.kwonlydefaults[parameter_name]}" \
			#		f"in {func.__qualname__} is not of required type {required_parameter_type}"
			#	except KeyError:
			#		# KeyError arises if we rely on default value
			#		# and do not pass a keyword explicitly
			#		pass
			return func(*args,**kwargs)
		
		return validate_and_call
	
	return validate_and_call_wrapper

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
				raise TypeError(f"{function.__qualname__}() got multiple values for argument "
								f"'{min(v for k,v in pos_arg.items() if v in argint)}'")
			
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
			# this could be made more efficient: pos_arg[i] is being called twice
			# Python 3.8 --> assignment expressions
			consume(source.pop(pos_arg[i],None) for i in range(arglim))
# 			arg_inds = [i if i<len(pos_arg else )]
			subargs = (
				operations[pos_arg[i]](a) if (pos_arg[i] in operations)
				else a
				for i,a in enumerate((args[p] for p in range(arglim)))
			)
			varargs = args[arglim:]
			
			return function(*subargs,*varargs, **source)
		return wrapper
	return decorator

def makeax(parameter):
	"""
	Given a function that specifies a keyword corresponding to a matplotlib
	Axes instance, automatically submit an axis for the keyword to the function
	at runtime if the keyword receives None (the intended default). The axis
	is given by plt.gca().
	"""
	from matplotlib import pyplot as plt
	op = lambda ax: plt.gca() if ax is None else ax
	
	if isinstance(parameter,str):
		# 'parameter' is the name of the argument that is targeted
		return modify_parameters_dynamically(
			**{parameter:lambda ax: plt.gca() if ax is None else ax}
		)
	else:
		# 'parameter' is the function, and it is assumed that 'ax' is the argument name
		return modify_parameters_dynamically(
			ax=lambda ax: plt.gca() if ax is None else ax
		)(parameter)

__all__ = ('assert_objects_are_of_specified_types','validate_parameter_types',
'separate_required_and_default_parameters','operate_on_params_dynamically',
'modify_parameters_dynamically', 'makeax')

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