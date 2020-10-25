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

__add_doc_template = '\n\n<<< documentation for {}>>>\n{}'

def tryname(obj):
	try: return obj
	except: return obj

def add_doc(*referent_functions, __add_doc_template = __add_doc_template):
	"""A decorator which, when called on a function, adds to the function's
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

def assign(names_values_mapping):
	"""Given a mapping (dict-like object) of names to values,
	return a function that assigns these values to an object
	when the object is passed to that function."""
	def _assign(object):
		consume(setattr(object, name, value)
				for name, value in names_values_mapping.items())
	return _assign

__all__ = ('add_doc','assign')

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