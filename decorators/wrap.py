"""
To do:
	* add two decorators that act in tandem:
		- one scans the documentation of a function seeking out a pattern,
		  which it extracts in order to assign some part of the documentation
		  to a dictionary
		- another function can read from this dictionary and inject the doc
		  snippets into the documentation of other functions
		- it might be worth having functions to be scanned stored somewhere,
		  and then another function comes along to call the routine that
		  actually scans these. Useful if a function requests a documentation
		  snippet that is not yet defined.
"""

# from functools import partial
from itertools import chain
from ..common.funcs import getattrs
from ..collections import consume

__add_doc_template = '\n\n<<< documentation for {} >>>\n{}'
__add_doc_const_template = '\n\n<<< CONSTANTS >>>\n{}'

def tryname(obj):
	try: return obj
	except: return obj

def add_doc(*referent_functions, __add_doc_template = __add_doc_template):
	"""A decorator that, when called on a function, adds to the function's
	documentation the documentation from each function supplied in
	`refernt_functions`.
	
	To do:
		handle indentation issue
	"""
	def _add_doc(function):
		add = __add_doc_template*len(referent_functions)
		function.__doc__ = f"{function.__doc__}{add}".format(
			*chain.from_iterable(
				getattrs(f.__func__ if isinstance(f, classmethod) else f,
						'__qualname__','__doc__')
				for f in referent_functions
			)
		)
		
		return function
	return _add_doc

def add_doc_constants(mapping, *names):
	"""A decorator that, when called on a function, adds to the function's
	documentation a titled, newline-separated list of constants supplied to the
	decorator by way of a mapping <name: value> and a list of names
	to take from the mapping."""
	def _add_const(function):
		function.__doc__ = (
			f"{function.__doc__}{__add_doc_const_template}".format(
				"\n".join(
					f"\t{name}: {mapping[name]}"
					for name in names
				)
			)
		)
		return function
	return _add_const

def assign(names_values_mapping):
	"""Given a mapping (dict-like object) of names to values,
	return a function that assigns these values to an object
	when the object is passed to that function."""
	def _assign(object):
		consume(setattr(object, name, value)
				for name, value in names_values_mapping.items())
	return _assign

def redirect_output_closure():
	import sys
	STDOUT, STDERR = sys.stdout, sys.stderr
	names = ['stdout', 'stderr']
	def redirect_output(stdout = STDOUT, stderr = STDERR, mode = 'a')
		if 'r' in mode:
			raise ValueError("cannot specify read-only mode")
		def decorator(func):
			def wrapper(*a, **kw):
				files = [stdout, stderr]
				originals = [STDOUT, STDERR]
				close = [False]*2
				for i,f,n,o in enumerate(zip(files, names, originals)):
					if isinstance(n, str):
						try:
							files[i] = open(n, mode)
							close[i] = True
						except:
							files[i] = o
							close[i] = False
					setattr(sys, n, files[i])
				result = func(*a, **kw)
				for f,c in zip(files, close):
					if c: f.close()
				sys.stdout = STDOUT
				sys.stderr = STDERR
				return result
			return wrapper
		return decorator
	return redirect_output

redirect_output = redirect_output_closure()

__all__ = ('add_doc', 'assign', 'add_doc_constants')

if __name__ == '__main__':
	def a():
		"""the documentation for a"""
	
	def b():
		"""b documentation"""
	
	def b():
		"""documentation c"""
	
	_a, _b = a.__doc__, b.__doc__
	
	def doc_add(*ref):
		@add_doc(*ref)
		def f():
			"""documentation for f"""
		
		add = __add_doc_template*len(ref)
		d = 'documentation for f'
		for _ in ref:
			d += __add_doc_template.format(_.__qualname__, _.__doc__)
		assert f.__doc__ == d, f'\n* * documentation:\n{f.__doc__}\n* * expected"\n"{d}'
	
	from itertools import permutations
	from ..collections import consume
	consume(doc_add(*ref) for r in range(3) for ref in permutations((a, b), r))