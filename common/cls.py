from collections import OrderedDict
from itertools import chain, cycle
import re

import numpy as np

import builtins

from ..arrays import make_object_array
from ..collections import consume

class EmptyDictType:
	_instance = None
	def __new__(cls):
		if cls._instance is None:
			cls._instance = super().__new__(cls)
		
		return cls._instance
	
	def keys(self): return ()
	def values(self): return ()

EmptyDict = EmptyDictType()

class Struct: # extension of https://stackoverflow.com/a/45517161
	"""
	struct-like object that allows accessing values
	as attributes rather than by dictionary access
	"""
	def __init__(self, *aitems, **kitems):
		self.names = []
		self.add(*aitems, **kitems)

	def add(self, *aitems, **kitems):
		for a in aitems: self._add(**a)
		self._add(**kitems)
		return self

	def _add(self, **items):
		self.__dict__.update(**items)
		self.names.extend(items.keys())
	
	def addfrom(self, obj, **kwargs):
		self._add({k:(getattr(obj,v) if isintance(v,str) else v) for k,v in kwargs.items})
		return self
	
	def keys(self):
		return self.names
	
	def values(self, names=None):
		if names is None: names = self.names
		return (getattr(self, n) for n in names)
	
	def __iter__(self):
		return ((n, getattr(self, n)) for n in self.names)
	
	def get(self, key, value=None):
		return self.__dict__.get(key, value)
	
	def __getitem__(self, key):
		return self.__dict__[key]
	
	def __setitem__(self, key, value):
		self.__dict__[key] = value

class ArgumentManager:
	"""Used to store, access, and update arguments (positional/keyword)
	to be used elsewhere. Associates a set of names (strings) with
	positional and keyword arguments for each name."""
	def __init__(self):
		"""Initialize self.map, which maps names to arguments."""
		self.map = {}
	
	def add(self, name, *a, **kw):
		"""For a name, add args/kwargs to be associated with this name."""
		args, kwargs = self.map.setdefault(name, ([], {}))
		args.extend(a)
		kwargs.update(kw)
	
	def keys(self):
		"""Return names as a dict_keys instance."""
		return self.map.keys()
	
	def values(self):
		"""Return zipped args/kwargs for each name as a dict_values instance."""
		return self.map.values()
	
	def items(self):
		"""Return each name with associated args/kwargs as a dict_values instance."""
		return self.map.items()
	
	def __len__(self):
		"""Return the number of names currently stored by this instance."""
		return len(self.map)
	
	def __iter__(self):
		"""Iterator over the names stored in this instance."""
		return iter(self.map)
	
	def __getitem__(self, key):
		"""Return the (args,kwargs) associated with a given `key`."""
		return self.map[key]
	
	def __contains__(self, key):
		"""Return whether this instance holds data for a given `key`."""
		return key in self.map
	
	def args(self, key=None):
		"""Return arguments corresponding to certain names.
		
		If key is None, return iterator
			((name, associated args) for name in this instance)
		
		If key is a `str`, return just the args associated with that key.
		
		Otherwise, key is assumed to be an iterable, and the iterator
		returns only the args for each name in key.
		"""
		if key is None:
			return ((name,item[0]) for name,item in self.items())
		
		if isinstance(key, str):
			return self.map[key][0]
		
		return (self.map[key][0] for name in key)
	
	def kwargs(self, key=None):
		"""Return keyword arguments corresponding to certain names.
		
		If key is None, return iterator
			((name, associated kwargs) for name in this instance)
		
		If key is a `str`, return just the kwargs associated with that key.
		
		Otherwise, key is assumed to be an iterable, and the iterator
		returns only the kwargs for each name in key.
		"""
		if key is None:
			return ((name,item[1]) for name,item in self.items())
		
		if isinstance(key, str):
			return self.map[key][1]
		
		return (self.map[key][1] for name in key)

class IterCall:
	"""Provides a convenient way of calling a function repeatedly
	with changing keyword arguments, using the IterDict interface.
	
	Specifically, an instance of this class is initialized with some function.
	Calling __call__ on this class will simply call the function
	with whatver args/kwargs are passed. Calling the call method
	will instead create an IterDict of the **kw passed and return a
	tuple of the results from calling the function iterating through
	kw via an IterDict. See the `call` method for details.
	"""
	def __init__(self, function):
		self.function = function
	
	# the problem with __iter__: how to pass args/kwargs?
	# could assign them to instance attributes,
	# but then this would also require setting these attributes
	# elsewhere, in __init__ or other methods, seems like clutter
# 	def __iter__(self):
# 		"""return an iterator that yields the same items from self.call"""
# 		return (self.function(*a, **k) for k in IterDict(iter_kw, **kw))
	
	def __call__(self, *a, **kw):
		self.function(*a, **kw)
	
	def call(self, *a, iter_kw=None, **kw):
		"""return tuple(self.function(*a, **k) for k in IterDict(iter_kw, **kw))"""
		return tuple(self.function(*a, **k) for k in IterDict(iter_kw, **kw))

class _CrossDictIterator:
	def __init__(self, kwargs=None, **kw):
		self._dictionary = kw
		if kwargs is not None:
			self._dictionary.update(kwargs)
	
	def __getitem__(self, key):
		return self._dictionary[key]
	
	def __setitem__(self, key, value):
		self._dictionary[key] = value
	
	def keys(self):
		return self._dictionary.keys()
	
	def values(self):
		return self._dictionary.values()
	
	def items(self):
		return self._dictionary.items()
	
	def update(self, kw):
		self._dictionary.update(kw)

class IterDict(_CrossDictIterator):
	"""
	Given a dict whose values are each iterables:
		>>> d = dict(a=(a0,a1,a2,...), b=(b0,b1,b2, ...), c=(c0,c1,c2,...), ...)
	
	Iterate over the dictionary by zipping the value iterables into new dictionaries:
		>>> list(d)
		>>> # [dict(a=a0, b=b0, c=c0, ...), dict(a=a1, b=b1, c=c1, ...),
		>>> #  dict(a=a2, b=b2, c=c2, ...), ...]
	
	Iterables passed may be of different lengths; keywords are
	dropped from the iteration as soon as they are omitted.
	
	Useful, e.g., for concisely specifying keywords to multiple function calls,
	as done in the IterCall class.
	
	Note: the current implementation is functional, but could be more efficient.
	A more efficient implementation might drop keywords as they are exhausted.
	This could be done in tandem with zipping values across kw-iterables
	at the start and then reading off these values as iterated over.
	"""
	def __iter__(self):
		# current position for each key-iterable
		self.__counters = {k: 0 for k in self._dictionary.keys()}
		# how far we can iterate for each key
		self.__lengths = {k: len(v) for k, v in self._dictionary.items()}
		# collective position across all keys
		self.__position = 0
		
		# how far we can iterate collectively across all keys
		# ValueError --> empty dictionary
		try: self.__max_position = max(self.__lengths.values())
		except ValueError: self.__max_position=0
		return self
	
	def __next__(self):
		# when we hit max_position, we have exhausted the longest iterable
		if self.__position < self.__max_position:
			# grab results for keywords that have an available value
			result = {k: self._dictionary[k][self.__position]
					  for k in self._dictionary.keys()
					  if self.__position < self.__lengths[k]}
			
			# move up an index
			self.__position += 1

			return result
		else:
			raise StopIteration

class CycleDict(_CrossDictIterator):
	def __init__(self, kwargs=None, **kw):
		self._dictionary = {k:cycle(v) for k,v in kw.items()}
		if kwargs is not None:
			self._dictionary.update({
				k:cycle(v)
				for k,v in kwargs.items()
		})
	
	def update(self, kw):
		self._dictionary.update({
				k:cycle(v)
				for k,v in kw.items()
		})
	
	def __iter__(self):
		return self
	
	def __next__(self):
		return {k:next(c) for k,c in self._dictionary.items()}

class MultiIterator:
	"""
	Use this to efficiently iterate over all the entries contained in
	each container in a collection of containers
	
	Similar to itertools.chain (actually wraps around this class)
	but unlike chain, this class stored target iterables
	so that they can continually be re-iterated
	
	Deeper-nested iteration can be accomplished by setting the
	targets of an instance to other MultiIterator or chain instances.
	
	Iteration down to some specified nesting can be accomplished
	via the 'depth' parameter; if depth>1, then each of the targets
	specified is transformed into a MultiIterator of its elements with depth=depth-1
	before being added to targets.
		** Note: in current implementation, targets must be deep enough to support
		specified depth level, otherwise this will try to iterate over scalars
		and raise a TypeError; even worse, if you have strings as the scalars,
		then they will be iterated over (which is not desired) without raising an error
	
	
	Note: flattening is provided elsewhere, might be incorporated here.
	See the module <common.collections.collection> for a flattening iterator.
	"""
	def __init__(self, *targets, depth=1, flatten=False):
		"""
		more to be done:
			what happens if the targets do not make a robust array--
			i.e. if some targets are of greater depth than others,
			and the depth parameter is too deep for some of them?
		"""
		self.targets=[]
		#self.add_targets(*targets)
		if flatten:
			"""
			targets = [
				t if isscalar(t)
				else MultiIterator(*t,flatten=flatten)
				for t in targets
			]
			self.add_targets(targets)
			"""
			raise NotImplementedError('flatten: this functionality is not implemented yet')
		else:
			if depth>1:
				#print('depth>1: targets:',targets)
				self.add_targets(*(MultiIterator(*targets,depth=depth-1)))
			else:
				#print('adding targets:',targets)
				self.add_targets(*targets)
	
	def __iter__(self):
		"""
		equivalent to an iterator that does this:
		>>> result = []
		>>> for t in targets:
		>>> 	for obj in t:
		>>>			yield obj
		"""
		self.iterator = chain.from_iterable(self.targets)#(obj for t in self.targets for obj in t)
		return self
	
	def __next__(self):
		return next(self.iterator)
	
	def add_target(self,t):
		"""
		add a target to this iterator;
		a target is a container that will be iterated over in the multi-iteration
		"""
		self.targets.append(t)
	
	def add_targets(self,*t):
		"""
		varargs function that allows adding multiple targets at once
		"""
		self.targets.extend(t)
	
	def extend(self, t):
		self.targets.extend(t)
	
	def __len__(self):
		return sum(map(len, self.targets))
	
	def __contains__(self,value):
		"""
		returns any(value in t for t in self.targets)
		resort to the __contains__ methods of targets
			so that they can search for the value according to their own logic
			rather than iterating over all values in them from the outside
		"""
		return any(value in t for t in self.targets)
	
	def unique(self,frozen=False):
		"""
		return a set (or frozenset if `frozen`=True)
		of the elements in this iterator
		"""
		if frozen: return frozenset(self)
		return set(self)
	
	def swap(i1,i2):
		"""
		swap the targets at indices i1,i2,
		i.e. change the order of iteration
		
		IndexError raised if either is invalid
		"""
		self.targets[i1],self.targets[i2] = self.targets[i2],self.targets[i1]
	
	def reorder(indices1,indices2):
		"""
		perform multiple swaps simultaneously
		
		...
		"""
		...
		NotImplemented

class NamedMultiIterator:
	"""
	use this to efficiently iterate over all the entries contained in
	each container in a collection of containers
	
	the difference with MultiIterator is that this class
	allows for iterating selectively over targets based on names
	associated with them; the names do not have to be unique,
	i.e. a name can map to multiple targets
	"""
	def __init__(self,*targets,names=None):
		self.targets={}
		self.add_targets(*targets,names)
	
	def __iter__(self):
		"""
		equivalent to iterating over this:
		>>> result = []
		>>> for t in targets.values():
		>>> 	for obj in t:
		>>>			result.append(obj)
		"""
		self.iterator = (obj for t in self.targets.values() for obj in t)
		return self
	
	def __next__(self):
		return next(self.iterator)
	
	def __call__(self,*key):
		if len(key)==1:
			return self.targets[key]
		else:
			return MultiIterator(*(self.targets[k] for k in key))
	
	def add_target(self,t,name=None):
		"""
		add a target to this iterator;
		a target is a container that will be iterated over in the multi-iteration
		
		a name may be specified; the name does not have to be unique
		
		if no name is provided, the name associated with the target is the None reference
		"""
		self.setdefault(name,[]).append(target)
	
	def add_targets(self, *t, name=None, names=None):
		"""
		varargs function that allows adding multiple targets at once
		
		either of name or name may be specified
		
		if name is given, then all targets provided have that name associated with them
		
		if names is given, each the target and name containers are zipped
			and thus associated
		
		if neither is provided, the name associated with the targets is the None reference
		"""
		#self.targets.extend(t)
		if (name is None) and (names is None):
			self.targets.setdefault(None,[]).extend(names)
		elif names is None:
			self.targets.setdefault(None,[]).extend(names)
		elif name is None:
			for n,t in zip(names,t):
				self.targets.setdefault(n).append(t)
		else:
			raise ValueError("specify one of 'name' or 'names', not both")
	
	def __contains__(self,value):
		"""
		returns any(value in t for t in self.targets)
		resort to the __contains__ methods of targets
			so that they can search for the value according to their own logic
			rather than iterating over all values in them from the outside
		"""
		return any(value in t for t in self.targets.values())
	
	def unique(self,frozen=False):
		"""
		return a set (or frozenset if `frozen`=True)
		of the elements in this iterator
		"""
		if frozen: return frozenset(self)
		return set(self)

class ProxyFunc:
	"""Used with the proxy class from this module."""
	def __init__(self, function, Proxy_obj):
		self.function = function
		self.Proxy_obj = Proxy_obj

	def __call__(self, *a, **k):
		targets = self.Proxy_obj.targets
		res = tuple(getattr(target, self.function)(*a, **k) for target in targets)
		# try:
		# 	<above>
		# except TypeError as e:
		# 	print('ProxyFunc Error:',repr(e))
		# 	res = tuple(getattr(target, self.function) for target in targets)
		try:
			return np.array(res)
		except ValueError as e:
			#print('FuncProxy: <{self.function.__qualname__}>: could not return array due to ValueError:',str(e))
			#print(res)
			return make_object_array(res)

class Proxy:
	'''
	This idea of this class is to provide for concise access to attributes
	on a set of target objects when numerous attribute accesses will occur
	across these objects. Requesting
		>>> Proxy(targets).attribute
	returns [obj.attribute for attribute in targets].
	
	Requesting
		>>> Proxy(targets).callable
	returns a ProxyFunc, such that
		>>> Proxy(targets).callable(*a, **kw)
	returns [obj.callable(*a, **kw) for attribute in targets].
	
	Results are returned as numpy arrays. If the objects returned
	are arbitrary Python objects, then the array will have dtype <object>.
	
	Proxy nesting is supported; i.e., if a Proxy has targets
	which are themselves Proxy instances, then getting an attribute
	on the top-level Proxy will delegate to the target Proxy instances,
	which will in turn delegate to their respective targets. This
	can occur to any depth. The results will be nested, not flattened.
	'''
	
	# def __init__(self, targets, primary=True):
	# 	self.targets=targets
	# 	self.count=len(targets)
	# 	self.primary=primary
	#
	# def __getattr__(self, attr):
	# 	if self.primary: return ProxyFunc(attr, self)
	# 	#return Proxy(, primary=False)
	
	def __init__(self, targets):
# 		print('called Proxy.__init__')
		self.targets = targets
	
	def __getattr__(self, attr):
		"""Retrieve the passed attribute on each object in self.targets.
		If the attribute yields a callable, return a ProxyFunc.
		Note: if the first result is callable, it is assumed that
		all results are callable."""
# 		print(f'Proxy: looking for attribute "{attr}" on targets {self.targets}')
		it = (getattr(t,attr) for t in self.targets)
		# if self.targets_are_homogenous:
		first = next(it)
		if callable(first):
			# print('first:',first)
			return ProxyFunc(attr, self)
		else:
			results = [None] * len(self.targets)
			results[0], results[1:] = first, it
			return results
# 			try:
# 				return np.array(results)
# 			except ValueError as e:
# 				#print(f'Proxy: <{attr}>: could not return array due to ValueError:',e.get_message())
# 				return make_object_array(results)
	
	def getattrs(self,*attrs, asarray=True):
		if asarray:
			return np.array([self.__get_named_attribute(a) for a in attrs])
		return [self.__get_named_attribute(a) for a in attrs]
	
	def __get_named_attribute(self,attr):
		v = getattr(self,attr)
		if isinstance(v, ProxyFunc): return v()
		return v
	
	"""def tryattr(self, t, attr):
		try: return getattr(t, attr)
		except AttributeError: return None"""
	
	# else:
	#	...
	
	def __getitem__(self, item):
		return self.targets.__getitem__(item)
		# questionable: should the behavior instead be
		# return [t.__getitem__(item) for t in self.targets] ?
	
	def __iter__(self):
		"""return self.targets.__iter__()"""
		return self.targets.__iter__()
	
	def __next__(self):
		return self.targets.__next__()
	
	def apply(self, callable):
		return tuple(callable(item) for item in self.targets)
	
	def map(self, callable):
		return map(callable, self.targets)
	
	def iterate(self,attr):
		return (getattr(p,attr) for p in self.targets)
		# to do: handle the case of a ProxyFunc being returned
	
	def __len__(self): return len(self.targets)

class ArgumentDeferrer:
	class DeferredMethod:
		def __init__(self, argdeferrer, method_name):
			self.argdeferrer = argdeferrer
			self.method_name = method_name
		
		def __call__(self, *a, **kw):
# 			print(
# 			'** adding call to ArgumentDeferrer from DeferredMethod: '
# 			f"{self.method_name}({', '.join(a)}, {', '.join('%s=%s'%(k,v) for k,v in kw.items())})"
# 			)
			self.argdeferrer.add_call(self.method_name, a, kw)
			return self
	
	def __init__(self):
		self.calls = OrderedDict()
	
	def __getattr__(self, method_name):
		return self.DeferredMethod(self, method_name)
	
	def add_call(self, method_name, a, kw):
# 		print(
# 		'** adding call to ArgumentDeferrer: '
# 		f"{method_name}({', '.join(a)}, {', '.join('%s=%s'%(k,v) for k,v in kw.items())})"
# 		)
		self.calls[method_name] = (list(a), kw)
	
	def set_obj(self, obj):
		self.obj = obj
	
	def get_obj(self):
		return self.obj
	
	def execute(self, obj=None):
# 		print(f"{self.__class__.__name__}: execute: call sequence:")
# 		for method_name, (a, kw) in self.calls.items():
# 			print(f'\t{method_name}:',a,kw)
		if obj is None: obj = self.obj
		consume(getattr(obj, method_name)(*a, **kw)
				for method_name, (a, kw) in self.calls.items())
	
	def clear(self):
		self.calls.clear()
	
	def _update(self, method_name, a, kw):
		args, kwargs = self.calls[method_name]
		args.extend(a)
		kwargs.update(kw)
	
	def update(self, method_name, *a, **kw):
		self._update(method_name, a, kw)

# class String(str):
# 	def split_iter(self, splitter):
# 		if isinstance(splitter, str):
# 			splitter = re.escape(splitter)
# 			return (m.group() for m in re.finditer(f'((!({splitter})).)*', self))

__all__ = ('Struct','ArgumentManager','IterDict','Proxy','IterCall',
'MultiIterator','NamedMultiIterator', 'EmptyDict', 'CycleDict',
'ArgumentDeferrer')