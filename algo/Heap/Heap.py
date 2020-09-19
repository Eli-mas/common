"""NOTE: currently has a bug, do not use until fixed: building an array by
heapification works as expected, but building a heap by pushing values and then
emptying the heap results is an unsorted result.

numba-/numpy-driven implementation of a heap data structure:
a left-complete, binary tree where every parent is <= its children.

'Heap' is actually a factory method that produces an instance of a numba-compiled
class created by calling jitclass on the _Heap class, which implements the heap
behavior. Depending on the dtype of the array passed into the heap, a new compiled
type may have to be created; it is then cached so it never has to be compiled a
second time. For example, if we call this for the first time:
>>> Heap(np.array([0,1,2,3]))

numba will compile the _Heap class with the _array attribute set to the
numba.int_[:] type. This compilation may take a few moments. If we then call
>>> Heap(np.array([0,1,2,3,4,5,6]))

the factory will realize that this data type (array of integers of default
precision) has already been specified and a jitclass-compiled class has
already been created, so this will be used with recompilation. Note:
the factory relies on numba's `typeof` method to handle typing logic.
It is for this reason that at the moment, only numpy-array inputs
are supported--if a Python list (or other container) is passed, it will be
converted to a numpy array before being passed to the compiled class. I intend
to modify this behavior at some point will be so that lists become supported.

Currently numeric and character types are supported: <int16, int32,
int64, uint16, uint32, uint64, float32, float64>, and <chr x n, unichr x n>.
I will attempt to provide a mechanism for handling custom compiled types,
though this will require a variant on the _Heap class, and will require
supporting list types.

_Heap implements the min-heap data structure. The methods provided are:
	__init__(self, data, downsize=False, empty=False):
		when a collection of values is supplied,
	    it is heapified in place in linear time.
	push(self):
		add a value to the heap, maintaining the invariant
	pop(self):
		remove the minimal value from the heap, maintaining the invariant
	empty:
		pop all values off the heap in sorted order, returning an array

Additional functionality provided by the module:
	heapsort(values): sort values via a compiled _Heap class

For reasons involving numba's automatic type-detection mechanisms,
a *truly* empty heap cannot be created. The way to create an empty heap
is to pass in an array with a single value, and pass the 'empty' keyword
argument to be true. That value MUST have the same type as the values
with which you later intend to fill the heap.

The underlying array containing the heap data is resized upwards as required,
by doubling its size when it runs out of room. It can also be configured
to down-size the array when the heap gets much smaller than the array,
by passing downsize=True to the Heap factory, or by setting h.downsize=True
for a compiled _Heap h.

The functionality is intended to mirror that of Python's heapq module where
possible. Certain functionality, such as retrieving n largest/smallest items,
multiway merge of sorted arrays, and poppush not yet implemented.

Dependencies are minimal: numpy & numba.
Currently tested on numpy 1.17.3, numba 0.51.2.

To do:
	* unit tests -- beyond the basic testing I have already done
	* aforementioned missing features
	* support using lists instead of arrays,
		which will allow for comparing arrays as the elements on the heap,
		as well as other njit-compiled objects
	* Support a 'maxsize' feature which, if passed, causes numba to compile
		a different class (_HeapMaxsizeBound) which has hard-coded checks
		in place to prevent heap from growing beyond a certain size
	...
"""

from ...common import print_update
import numpy as np
from numba import njit, int_ as nb_int, float_ as nb_float, bool_ as nb_bool
from numba.experimental import jitclass#,
from numba import typeof
from numba.types import Array as nb_Array

def heapmaker_maker():
	heap_types = {}
	def heapmaker(data=None, data_type=nb_int[::1], **kw):
		if data is None:
			data = np.array([0])
			# note: this is a bug waiting to happen!
			# if a character type is passed, this is invalid--have to determine
			# a workaround
		else:
			data = np.asarray(data)
			data_type = typeof(data)
		
		print_update('heapmaker: data_type:',data_type)
		
		try:
			t = heap_types[data_type]
		except KeyError:
			spec = {
				'size': nb_int,
				'_array': data_type,
				'_downsize': nb_bool
			}
			t = jitclass(spec)(_Heap)
			heap_types[data_type] = t
		
		print_update('')
		
		return t(data, **kw)
	
	return heapmaker

Heap = heapmaker_maker()
spec = {
	'size': nb_int,
	'_array': nb_int[::1],
	'_downsize': nb_bool
}
# @jitclass(spec)
class _Heap:
	def __init__(self, data, downsize = True, empty=False):
		"""Initialize the heap, possibly with data.
		Parameters:
			data: <compatible with numba.int_[:]>: (optional):
				container of values; if provided, it will be heapified,
				after being copied to a numpy array, in O(n) time. Otherwise,
				heap is initially empty (actually it has one element, but
				the heap does not know it is there).
			
			downsize: bool (optional):
				whether or not to periodically downsize
				the array as it is reduced in size by way of the 'pop' method
		"""
		size = len(data)
		self._array = data
		self._downsize = downsize
		if empty:
			self.size=0
		else:
			self.size = size
			if size != 1:
				self._heapify()
	
	def push(self, value):
		"""Push a value onto the heap, maintaining the heap invariant.
		Time is O(lg n). Note: if you build a heap from scratch by pushing
		values repeatedly, the time is O(n*lg n), so you are better off
		initializing the heap with the data if you can."""
# 		print('push:',value,'array:',self._array)
		i = self.size
		if len(self._array)==self.size:
# 			print('upsizing')
			#self.resize_upwards()
			new = np.empty(len(self._array)<<1, dtype=self._array.dtype) # a<<1 = a*2
			new[:self.size] = self._array
			self._array = new
# 		print('\tsetting array index',i,'to',value)
		self._array[i] = value
# 		print('\tarray after initial assign:',self._array)
		parent = (i-1)>>1
# 		print(f'\tparent: index {parent} value {self._array[parent]}')
		while (parent > 0) and self._array[i]<self._array[parent]:
			temp = self._array[i]
			self._array[i] = self._array[parent]
			self._array[parent] = temp
			i = parent
			parent = (i-1)>>1
# 			print(f'\tparent: index {parent} value {self._array[parent]}')
		self.size += 1
	
	def pop(self):
		"""Remove the minimal value from the heap and restore the invariant.
		Time is O(lg n). Popping all the values off returns them in sorted
		order."""
		if self.size == 0:
			raise ValueError('cannot pop element off empty heap')
		popped = self._array[0]
		
		if self.size==1:
			self.size = 0
			return popped
		
		self._array[0] = self._array[self.size-1]
		self.size -= 1
		self._maintain_invariant(0)
		if self._downsize and (self.size < (len(self._array)>>1)):
# 			print('downsizing')
			"""new = np.empty(self.size, dtype=self._array.dtype)#
			new[:] = self._array[:self.size]
			self._array = new"""
			self._array = np.copy(self._array[:self.size])
		return popped
	
	def _heapify(self):
		"""Build a heap in place in O(n) time."""
		for i in range((self.size-1) >> 1, -1, -1): # a>>1 = a//2
			self._maintain_invariant(i)
	
	def _isleaf(self, i):
		"""Return whether or not the index denotes a leaf.
		Note: currently no bounds checking implemented, so you will not be
		warned if you call an index beyond the heap's size.
		"""
		return (i<<1) + 1 >= self.size
	
	def __maintain_invariant(self, i):
		"""Ensure that the invariant is maintained for a parent at position `i`
		and its children, i.e. ensure the parent value is not greater than
		the values of its children; if so, correct the invariant (swap the
		parent with the lesser of its children), and return the index that
		received the former parent value. If no swap occurs, return -1 to
		signify that the invariant was already maintained."""
# 		print('\t__maintain_invariant: index =',i)
# 		print('\tarray:',self._array)
		if not self._isleaf(i):
			parent_value = self._array[i]
			left_index = (i<<1) + 1
			right_index = (i<<1) + 2
			left_value = self._array[left_index]
			if right_index == self.size:
				if parent_value>left_value:
					j = left_index
# 					print('\t\theap inconsistency:',j)
					temp = self._array[j]
					self._array[j] = self._array[i]
					self._array[i] = temp
# 					print('\t\tarray after swap:',self._array)
					return j
			else:
				right_value = self._array[right_index]
# 				print(f'\t\tparent (index={i}): {parent_value}, left(index={(i<<1) + 1}) = {left_value}, right(index={(i<<1) + 2}) = {right_value}')
# 				print('values of parent, left, right:',parent_value, left_value, right_value)
				if parent_value>left_value or parent_value>right_value:
					if left_value < right_value: j = left_index
					else: j = right_index
# 					print('\t\theap inconsistency:',j)
					temp = self._array[j]
					self._array[j] = self._array[i]
					self._array[i] = temp
# 					print('\t\tarray after swap:',self._array)
					return j
		return -1
	
	def _maintain_invariant(self, i):
		"""Ensure the heap invariant for the root at position `i`.
		Uses an iterative solution rather than a recursive one because
		(1) it is more efficient anyway and (2) numba (0.5.1) does not yet
		support re-entrant calls on methods of compiled classes."""
# 		print('top _maintain_invariant call:',i)
		
		while i!=-1: # return signal to indicate that invariant is ensured
			i = self.__maintain_invariant(i)
	
	def empty(self):
		"""Pop all values off the heap in sorted order, returning an array"""
		return np.array([self.pop() for _ in range(self.size)])
	
	def __str__(self):
		return str(self._array[:self.size])

def heapsort(values):
	"""Sort values via the compiled _Heap class"""
	return Heap(np.asarray(values)).empty()

if __name__ == '__main__':
	from random import shuffle
	"""
	# very basic tests for the time being
	hv = np.array([0,5,2,4,3,1,7,6,8,9])
	dtypes = (np.int16, np.int32, np.int64, np.float32, np.float64, np.uint16, np.uint32, np.uint64)
	for dtype in dtypes:
		h = Heap(hv.astype(dtype))
# 		for i in : h.push(i)
		tuple(h.pop() for _ in range(h.size))
	for dtype in dtypes:
		h = Heap(hv.astype(dtype)[:1],empty=True)
	for dtype in dtypes:
		h = Heap(hv.astype(dtype))
# 		for i in : h.push(i)
# 		print('built heap:', h._array)
		popped, target = tuple(h.pop() for _ in range(h.size)), tuple(range(len(hv)))
		assert popped == target, f'expected {target}, got {popped}'
	
	for _ in range(10):
		shuffle(hv)
		target = np.sort(hv)
		h = Heap(hv.copy())
		popped = h.empty()
		assert np.array_equal(popped, target), f'expected {target}, got {popped}'
	
	for _ in range(10):
		shuffle(hv)
		repeat = 1
		target = np.sort(hv.tolist()*(repeat+1))
		h = Heap(hv.copy())
		for _ in range(repeat):
			for v in hv:
				h.push(v)
		popped = h.empty()
		assert np.array_equal(popped, target), f'expected {target}, got {popped}'
	
	for _ in range(3):
		for values in (list('abcdefghij'),['ab','c','def','g','hij','abcd']):
			for _ in range(10):
				shuffle(values)
				hv = np.array(values)
				target = np.sort(hv)
				h = Heap(hv)
				popped = [h.pop() for _ in range(h.size)]
				assert np.array_equal(popped, target), f'expected {target}, got {popped}'
		
		for dtype in ('<U10','<S10'):
			hv = np.array(values).astype(dtype)
			target = np.sort(hv)
			h = Heap(hv)
			popped = [h.pop() for _ in range(h.size)]
			assert np.array_equal(popped, target), f'expected {target}, got {popped}'
	"""
	print('sorting large arrays')
	for _ in range(10):
		values = np.random.randint(0, 1000, 10**6)
		h = Heap(values.copy())
		e = h.empty()
		assert (np.ediff1d(e)>=0).all(), \
			'large heap built initially: array is not sorted'
		assert np.array_equal(e, np.sort(values)), \
			f'large heap built initially: expected:\n{np.sorted(values)}\ngot:\n{e} (size={len(e)})'
		
		h = Heap(values[:1],empty=True)
		for v in values: h.push(v)
		e = h.empty()
		assert (np.ediff1d(e)>=0).all(), \ # this is the breaking issue
		'large heap built by pushing: array is not sorted'
		assert np.array_equal(e, np.sort(values)), \
			f'large heap built by pushing: expected:\n{np.sorted(values)}\ngot:\n{e} (size={len(e)})'
	
# 	_