"""Algorithms/routines related to iterating over collections."""

from collections import deque
from heapq import heapify, heappush, heappop
from itertools import takewhile, islice, chain
from bisect import bisect_left, bisect_right

__NULL = object()

def consume(iterable, n=None):
	"""
	as in https://docs.python.org/3/library/itertools.html#itertools-recipes
	"""
	if n is None:
		deque(iterable,0)
	else:
		next(islice(iterator, n, n), None)

def isscalar(val):
	"""Determine if a value is scalar. Strings are taken to be scalars, and
	anything else non-iterable/iterable is taken to be scalar/non-scalar."""
	if isinstance(val,str):
		return True
	try:
		iter(val)
		return False
	except TypeError:
		return True

def heapiter(heap):
	return (heappop(heap) for _ in range(len(heap)))

def flatten_loop(iterable):
	stack = deque()
	container = iter(iterable)
	try:
		while True:
			try:
				val = next(container)
				#print('val:',val)
				if isscalar(val):
					#print('\tval is scalar:',val)
					yield val
				else:
					#print('\tval is iterable:',val)
					stack.append(container)
					container = iter(val)
			except StopIteration:
				container = stack.pop()
	except IndexError:
		pass

def flatten_modify_loop(iterable, action):
	stack = deque()
	container = iter(iterable)
	try:
		while True:
			try:
				val = next(container)
				#print('val:',val)
				if isscalar(val):
					#print('\tval is scalar:',val)
					yield action(val)
				else:
					#print('\tval is iterable:',val)
					stack.append(container)
					container = iter(val)
			except StopIteration:
				container = stack.pop()
	except IndexError:
		pass

def empty_deque(d,n=None,reverse=False):
	"""
	Return an iterator over the values in a deque,
	which removes those values from the deque as they are yielded.
	Argument `n` allows for removing a variable number of elements,
	and `reverse` allows for iterating backwards over the deque.
	"""
	if n is None:
		n = len(d)
	elif n>len(d):
		raise ValueError('specify `n` such that 1 <= n <= len(d)')
	
	func = d.pop if reverse else d.popleft
	return (func() for _ in range(n))

def empty_dict(d,inorder=False, with_keys=False):
	"""
	Return an iterator over the values in a dict, which removes
	the corresponding keys from the dict as values are yielded.
	If `inorder`, yield over keys in sorted order.
	If `with_keys`, yield tuples (key,d[key]) as pops occur.
	"""
	if inorder:
		if with_keys:
			return ((k,d.pop(k)) for k in sorted(d))
		return (d.pop(k) for k in sorted(d))
	if with_keys:
		return ((k,d.pop(k)) for k in d)
	return (d.pop(k) for k in d)

def eslice(iterable,start=None,end=None,step=None):
	"""
	Combination of enumerate and itertools.islice that mimics enumerate behavior,
	but allows for variable starting position: i.e. yield (index of value in iterable, value)
	starting at the specified `start` index. `end` and `step` may be specified and are
	passed to corresponding arguments in islice.
	"""
	return enumerate(islice(iterable,start,end,step),start)


def groupby_whole(values,key,double=False):
    """
    Accomplish what `groupby` accomplishes, but without results being sorted,
    and returning all results at once rather than as a generator. Output is returned as a dict.
    
    If double is True, then instead of using lists to hold results while
    the output dict is being built, use deques, and cast these into lists
    in the most memory-efficient way possible (using `empty_deque`) after
    all the dict is complete. The benefit is that intermediary lists
    do not have to be resized while building results.
    """
    r = {}
    if double:
    	consume(r.setdefault(key(i),deque()).append(i) for i in values)
    	consume(r.__setitem__(k,list(empty_deque(r[k]))) for k in r.keys())
    else:
    	consume(r.setdefault(key(i),[]).append(i) for i in values)
    return r





def groupby(values, key):
	"""
	Accomplish what `itertools.groupby` accomplishes, but in one function call.
	The key,value pairs yielded by this function are now the key outputs
	and containers (not generators) of values; i.e., this function
	is a generator only over key-bound value sets.
	
	Key groups are yielded in sorted order by key,
	and values within each group are also sorted.
	
	Complexity is O(n*lg(n) + nK), where K is the amortized complexity
	of calling the key function on any particular value.
	
	This is a generator function.
	"""
	from heapq import heapify,heappop
# 	print('groupby: values:',values)
	values = [(key(v),v) for v in values]
	heapify(values)
	
	d=deque()
# 	print('\tgroupby: values:',values)
	current_key,v = heappop(values)
	d.append(v)
	
	while values:
		key,v = heappop(values)
		if key != current_key:
			yield current_key,list(empty_deque(d))
			current_key = key
		d.append(v)
	
	yield current_key,list(empty_deque(d))

def counter(values,key=None,include=()):
	"""
	Return a dict consisting of the counts of each unique
	item found in the iterable. If `include` is provided,
	it should be an iterable of expected items;
	if these items are not found, they will be set with count=0
	in the output dict.
	
	To do: consider parallelizing for large container iterables
	"""
	d={}
	def update(k):
		try: d[k] += 1
		except KeyError: d[k]=1
	if key is None:
		consume(map(update,values))
	else:
		consume(map(update,map(key,values)))
	for v in include:
		d.setdefault(v,0)
	return d

def collect_dicts(dicts, aggregator=None, container=False):
	out = {}
	dicts = sorted(dicts, key=len, reverse=True)
	
	if container:
		for i in range(len(dicts)-1,-1,-1):
			keys = deque(dicts[i])
			for k in empty_deque(keys):
				out[k] = tuple(chain.from_iterable(_.pop(k,()) for _ in reversed(dicts)))
			del dicts[-1]
	elif aggregator:
		null = object()
		for i in range(len(dicts)-1,-1,-1):
			keys = deque(dicts[i])
			for k in empty_deque(keys):
				out[k] = aggregator(filter(
					lambda v: v is not null,
					(_.pop(k,null) for _ in reversed(dicts))
				))
			del dicts[-1]
	else:
		null = object()
		for i in range(len(dicts)-1,-1,-1):
			keys = deque(dicts[i])
			for k in empty_deque(keys):
				out[k] = tuple(filter(
					lambda v: v is not null,
					(_.pop(k,null) for _ in reversed(dicts))
				))
			del dicts[-1]
	return out

def ifind(target, iterable, default=-1):
	"""see https://stackoverflow.com/a/9542768"""
	return next((i for i,_ in enumerate(iterable) if _==target), default)

def ifind_is(target, iterable, default=-1):
	"""similar to ifind, but search for identical objects"""
	return next((i for i,_ in enumerate(iterable) if _ is target), default)

def setdefaults(d, keys, value=__NULL, values=__NULL):
	"""Set defaults on a dict 'd' over multiple keys.
	If all keys are set to the same default value, specify 'value'.
	If each key receives a particular value associated with it,
	specify 'values'. If both are specified, 'value' takes precedence."""
	if value is not __NULL:
		consume(d.setdefault(k,value) for k in keys)
	else:
		consume(d.setdefault(k,v) for k,v in zip(keys,values))

def partition(values, target, indices=False, ensure_all_keys = False):
	"""
	Partition an iterable with respect to a target value.
	
	If indices if False, returning a dict with these key/value pairs:
		-1: list of values with v < target
		0: list of values with v == target
		1: list of values with v > target
	
	If indices is True, return a dict with the same keys,
	but with lists of indices of values rather than values.
	"""
	from operator import lt, eq, gt
	funcs = (lt, eq, gt)
	if indices:
		result = dict(groupby(
			range(len(values)),
			key = lambda v: next(f for f in funcs if f(values[i], target))
		))
	else:
		result = dict(groupby(
			values,
			key = lambda v: next(f for f in funcs if f(v, target))
		))
	if not ensure_all_keys:
		return result
	setdefaults(result, funcs, value = ())
	return dict(zip((-1,0,1), (result[k] for k in funcs)))

def partition_sorted(values, target, indices=True):
	low, bisect_left(values, target)
	high = bisect_right(values, target, low)
	if indices:
		return low, high
	return values[:low], values[low:high], values[high:]

def argsort(values):
	return sorted(range(len(values)), key = lambda i: values[i])

def sort_by(*iterables):
	inds = argsort(iterables[0])
	return tuple([coll[i] for i in inds] for coll in iterables)



__all__ = ('consume', 'ifind', 'ifind_is', 'groupby', 'groupby_whole')

if __name__ == '__main__':
	from itertools import takewhile, count
	
	sys.exit()
	
	def flatten_native(iterable):
		"""
		this attempts to use native python iterators,
		but it must do so in an unwieldy way--
		it builds a result list instead of
		yielding values as a generator, and uses a lambda,
		so it is slower
		"""
		stack = deque()
		container = iter(iterable)
		container_container=[container]
		ret = deque()
		
		def inner(stack,container_container):
			try:
				val = next(container_container[0])
				#print('val:',val)
				if isscalar(val):
					#print('\tval is scalar:',val)
					ret.append(val)
				else:
					#print('\tval is iterable:',val)
					stack.append(container_container[0])
					container_container[0] = iter(val)
			except StopIteration:
				container_container[0] = stack.pop()
		
		try:
			consume(takewhile(lambda i: True,(inner(stack,container_container) for _ in count())))
		except IndexError:
			return ret
	
	a = [[[1, 2], 3, [4, 5]], [6, 7, 8], [9]]
	def timer():
		from timeit import timeit
		a = [a]*100000
		statement_native = 'list(flatten_native(a))'
		timing_native = timeit(statement_native,globals=locals(),number=10)
		print(timing_native)
		
		del statement_native, timing_native
		
		statement_loop = 'list(flatten_loop(a))'
		timing_loop = timeit(statement_loop,globals=locals(),number=10)
		print(timing_loop)	
	
	print(list(flatten_modify_loop(a,lambda i: 2*i)))