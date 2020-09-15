from collections import deque
from heapq import heapify, heappush, heappop
from itertools import islice
from.algo import empty_deque



# def combinations(elements, r, group_level):
# 	"""
# 	combinations, but you can group at arbitrary depth
# 	"""
# 	elements = tuple(elements)
# 	
# 	...

		

class _AbstractOneSidedContainer:
	"""
	abstract class that bases Stack and Queue implementations
	"""
	def __init__(self,*a):
		self.d = deque(*a)
	
	def push(self,value):
		"""Add a new value."""
		self.d.append(value)
	
	def extend(self, values):
		self.d.extend(values)
	
	def is_empty(self):
		"""Boolean test of emptiness. Equivalent to <not self>."""
		return not self.d
	
	def clear(self):
		"""Delete all elements."""
		self.d.clear()
	
	def __len__(self):
		return len(self.d)
	
	def empty(self, out=list, n=None):
		"""Remove and return some number of elements `n` from the container.
		If n is None (default), return all elements. Requires implementation of
		the 'pop' method."""
		if n is None:
			n = len(self)
		elif n<0 or n>len(self.d):
			raise ValueError('specify `n` such that 1 <= n <= len(self)')
		
		return out(self.pop() for _ in range(n))
	
	def __bool__(self):
		"""True if the container has any elements, False otherwise."""
		return self.d.__bool__()

class Queue(_AbstractOneSidedContainer):
	"""
	First-in, first-out container
	"""
	def pop(self):
		"""Remove and return the earliest element added."""
		return self.d.popleft()
	
	def peek(self):
		"""Retrieve, but do not remove, the earliest element added."""
		return self.d[0]
	
	def __str__(self):
		return f"Queue<[{str(self.d)[7:-2]})>"
	
	def __repr__(self):
		return self.__str__()
	
	def forwards(self):
		"""Iterator over the queue in standard direction (FIFO)."""
		return iter(self.d)
	
	def backwards(self):
		"""Iterator over the queue in reverse direction (LIFO)."""
		return reversed(self.d)

class Stack(_AbstractOneSidedContainer):
	"""
	First-in, last-out container
	"""
	def pop(self):
		"""Remove and return the last element added."""
		return self.d.pop()
	
	def peek(self, i=0):
		"""Retrieve, but do not remove, the last element added.
		If i is specified, return the value at i positions down
		from the stack top."""
		return self.d[-1-i]
	
	def __str__(self):
		return f"Stack<({str(self.d)[7:-2]}]>"
	
	def __repr__(self):
		return self.__str__()
	
	def top_down(self):
		"""Iterator over the stack in standard direction (LIFO)."""
		return reversed(self.d)
	
	def bottom_up(self):
		"""Iterator over the stack in reverse direction (FIFO)."""
		return iter(self.d)
	
	def walk(self):
		"""Iterator over the stack in standard direction (LIFO),
		equivalent to 'top_down'."""
		return reversed(self.d)

class Deque(deque):
	def filtrate(boolean_callable):
		"""
		Generator function: cycle through elements, keeping ones that satisfy
		the boolean_callable passed, removing and yielding the others.
		"""
		for _ in range(len(self)):
			value = self.popleft()
			if boolean_callable(value):
				self.append(value)
			else:
				yield value
	
	def exfiltrate(boolean_callable):
		"""
		Generator function: cycle through elements, keeping ones that do not
		satisfy the boolean_callable passed, removing and yielding the others.
		"""
		for _ in range(len(self)):
			value = self.popleft()
			if not boolean_callable(value):
				self.append(value)
			else:
				yield value

class disjoint_chunked:
	"""Iterate over tuples of a given size constructed from non-overlapping
	values yielded from an iterable. If the iterable's size is not divisble
	by the chunk size, the last tuple yielded will be smaller than the
	expected."""
	def __init__(self, iterable, chunk_size):
		self.size = chunk_size
		self.iterable = iter(iterable)
	
	def _chunked_iter(self):
		"""the generator function that yields elements."""
		items = list(islice(self.iterable,self.size))
		while items:
			print('items:',items)
			yield tuple(items)
			items[:] = islice(self.iterable,self.size)
	
	def __iter__(self):
		return self._chunked_iter()

class windowed_chunked:
	"""
	Iterate over an iterable in chunks of size n (sliding window).
	If the iterable has fewer elements than the specified size,
	a ValueError is raised.
	"""
	def __init__(self,iterable,chunk_size,skip=1):
		self.size = chunk_size
		self.iterable = iter(iterable)
		self.skip = skip
	
	def _chunked_iter(self):
		"""the generator function that yields elements."""
		items = deque(islice(self.iterable,self.size), maxlen = self.size)
		"""if len(items) < self.size:
			self.items = tuple(items)
			raise ValueError('chunk size must be no larger than length of iterable')"""
		for item in self.iterable:
			yield tuple(items)
			items.append(item)
		yield tuple(empty_deque(items))
	
	def _chunked_iter_skip(self):
		"""to be done"""
	
	def __iter__(self):
		if self.skip > 1: return self._chunked_iter_skip()
		return self._chunked_iter()
	
	'''def get_items_from_raise(self):
		"""
		In the event the iterable provided is too small and winds up being consumed
		without being used, use this method to get back the items in question.
		"""
		return self.items'''

class takewhile:
	"""
	Similar to itertools.takewhile, but this keeps track of the first
	element that does not yield the specified condition.
	To get this element on a takewhile instance t, get t.last.
	To get this element and the remaining elements in the iterable,
	call t.remaining(), which yields a generator on these values.
	This works on both containers and generators.
	"""
	def __init__(self,condition,iterable):
		"""
		condition: the callable, boolean-yielding condition
		iterable: the iterable to be iterated over
		"""
		self.condition = condition
		self.iterable = iter(iterable)
	
	def __iter__(self):
# 		self.result = deque()
		return self
	
	def __next__(self):
		v = next(self.iterable)
		
		if self.condition(v):
			return v
		else:
			self.last = v
			raise StopIteration
	
	def remaining(self,out=None):
		"""
		Return a generator yielding the last element
		that did not satisfy the condition, and then
		all remaining elements in the iterable.
		"""
		# first, return the missed element
		yield self.last
		
		# self.iterable is a generator and by now only includes
		# remaining elements, so simply yield them
		for v in self.iterable:
			yield(v)

__all__ = ('Stack', 'Queue', 'takewhile', 'disjoint_chunked', 'windowed_chunked')
