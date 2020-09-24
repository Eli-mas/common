"""numba-/numpy-driven implementation of a heap data structure:
a left-complete, binary tree where every parent is <= its children.
For large arrays this is more efficient than Python's native heapq module--
see Heap_timing.py. For smaller arrays, heapq runs faster, but this could
change if the time taken to convert arrays to Python lists is factored in.

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
			# if a character type is passed, this is invalid--
			# have to determine a workaround
		else:
			data = np.asarray(data)
			data_type = typeof(data)
		
# 		print_update('heapmaker: data_type:',data_type)
		
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
		
# 		print_update('')
		
		return t(data, **kw)
	
	return heapmaker

Heap = heapmaker_maker()
# spec = {
# 	'size': nb_int,
# 	'_array': nb_int[::1],
# 	'_downsize': nb_bool
# }
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
			
			empty: bool (optional):
				specify true if you want the heap to be empty;
				due to current design, you still must provide at least
				a length-1 array to the constructor even if set to True.
				Providing a larger array than the current data stored
				can be useful if you know the array will receive more data,
				so you can prevent some amount of dynamic resizing.
		"""
		size = len(data)
		self._array = data
		self._downsize = downsize
		if empty:
			self.size=0
		else:
			self.size = size
			if size != 1: # make the array into a heap in place; O(n) time
				self._heapify()
	
	def push(self, value, verbose = False):
		"""Push a value onto the heap, maintaining the heap invariant.
		Time is O(lg n), amortized in the case where the array has to
		be enlarged. Note: if you build a heap from scratch by pushing
		values repeatedly, the time is O(n*lg n), so you are better off
		initializing the heap with the data if you can."""
		
		# heap has n elements, so index (len(heap)-1)+1 = n is the next index
		i = self.size
		if len(self._array)==self.size:
			# can't push unless the array has room; first,
			# enlarge the array in a way that has constant amortized-time
			new = np.empty(len(self._array)<<1, dtype=self._array.dtype) # a<<1 = a*2
			new[:self.size] = self._array
			self._array = new
		# now push
		self._array[i] = value
		parent = (i-1)>>1
		
		# percolate value up the array: while child is less than parent, swap
		# parent and child, and make the value at the position of the previous
		# parent the location of the new child
		while (i > 0) and self._array[i]<self._array[parent]:
			temp = self._array[i]
			self._array[i] = self._array[parent]
			self._array[parent] = temp
			i = parent
			parent = (i-1)>>1
		self.size += 1
	
	def pop(self):
		"""Remove the minimal value from the heap and restore the invariant.
		Time is O(lg n). Popping all the values off returns them in sorted
		order."""
		if self.size == 0:
			raise ValueError('cannot pop element off empty heap')
		
		# top item is minimal item
		popped = self._array[0]
		
		# no reason to do all the work following this block if size is 1
		if self.size==1:
			self.size = 0
			return popped
		
		# replace top item with last item; this generally breaks the invariant
		self._array[0] = self._array[self.size-1]
		self.size -= 1
		# the heap invariant has to be restored: O(lg n) time
		self._maintain_invariant(0)
		
		# we can downsize if we want to conserve on memory
		if self._downsize and (self.size < (len(self._array)>>1)):
# 			print('downsizing')
			"""new = np.empty(self.size, dtype=self._array.dtype)#
			new[:] = self._array[:self.size]
			self._array = new"""
			self._array = np.copy(self._array[:self.size])
		
		return popped
	
	def _heapify(self):
		"""Build a heap in place in O(n) time."""
		# Simply ensure that heap invariant is maintained at all indices
		# starting from the bottommost parents of the heap.
		# Although _maintain_invariant takes O(n*lg(n)) times in worst case,
		# not all nodes experience this worst case; summing over all nodes
		# results in linear time
		for i in range((self.size-1) >> 1, -1, -1): # a>>1 = a//2
			self._maintain_invariant(i)
	
	def _isleaf(self, i):
		"""Return whether or not the index denotes a leaf.
		Note: currently no bounds checking implemented, so you will not be
		warned if you call an index beyond the heap's size.
		"""
		# (i<<1) + 1 gives the index that specifies the left child of the value
		# at index i if the child exists. If this is beyond the heap's size,
		# it means that child does not exist, i.e. this is a leaf.
		return (i<<1) + 1 >= self.size
	
	def __maintain_invariant(self, i):
		"""Ensure that the invariant is maintained for a parent at position `i`
		and its children, i.e. ensure the parent value is not greater than
		the values of its children; if so, correct the invariant (swap the
		parent with the lesser of its children), and return the index that
		received the former parent value. If no swap occurs, return -1 to
		signify that the invariant was already maintained."""
		# since a leaf has no children, by default it maintains the invariant,
		# so we do not check leaves
		if not self._isleaf(i):
			parent_value = self._array[i]
			left_index = (i<<1) + 1
			right_index = (i<<1) + 2
			left_value = self._array[left_index]
			if right_index == self.size:
				# a child cannot exist at this value of right_index,
				# so only check the left child
				if parent_value>left_value:
					# swap parent with left child
					j = left_index
					temp = self._array[j]
					self._array[j] = self._array[i]
					self._array[i] = temp
					return j
			else:
				right_value = self._array[right_index]
				if parent_value>left_value or parent_value>right_value:
					# choose the smaller child to swap with the parent
					if left_value < right_value: j = left_index
					else: j = right_index
					# do the swap
					temp = self._array[j]
					self._array[j] = self._array[i]
					self._array[i] = temp
					return j
		
		# no swap performed if we get here
		return -1
	
	def _maintain_invariant(self, i):
		"""Ensure the heap invariant for the root at position `i`.
		Uses an iterative solution rather than a recursive one because
		(1) it is more efficient anyway and (2) numba (0.5.1) does not yet
		support re-entrant calls on methods of compiled classes."""
		
		# iteratively send a value down the heap until it winds
		# up in a position were it maintains the invariant.
		# __maintain_invariant returns the new index needing to be checked,
		# if there is one; otherwise it returns -1.
		while i!=-1:
			i = self.__maintain_invariant(i)
	
	def empty(self):
		"""Pop all values off the heap in sorted order, returning an array"""
		return np.array([self.pop() for _ in range(self.size)])
	
	def __str__(self):
		# I don't think numba actually uses this when this gets compiled?
		return str(self._array[:self.size])
	
	def peek(self):
		"""Identify but do not pop the minimum value on the heap."""
		return self._array[0]
	
	### test methods ###
	
	def _maintains_invariant_at_index(self, i):
		"""Verify that a parent is <= its children"""
		parent_value = self._array[i]
		left_index = (i<<1) + 1
		right_index = (i<<1) + 2
		if (left_index < self.size) and (parent_value > self._array[left_index]):
			print('\t!!! parent @ index',i,'> left child @ index', left_index)
			return False
		if (right_index < self.size) and (parent_value > self._array[right_index]):
			print('\t!!! parent @ index',i,'> right child @ index', right_index)
			return False
		return True
	
	def _maintains_invariant(self):
		print('\t_maintains_invariant: starting at index', (self.size-1) >> 1, 'and working backwards')
		for i in range((self.size-1) >> 1, -1, -1):
			if not self._maintains_invariant_at_index(i):
				print('\tinvariant broken at index', str(i)+',', 'value', str(self._array[i])+', ',
# 					  'children {self.get_children(i)}, array {self._array}'
					  )
				return False
		return True
	
	# not used right now
	"""def get_children(self, i):
		left_index = (i<<1) + 1
		right_index = (i<<1) + 2
		return , \
			   self._array[right_index] if right_index < self.size else '/'"""

def heapsort(values):
	"""Sort array of values via the compiled _Heap class"""
	return Heap(np.asarray(values)).empty()

if __name__ == '__main__':
	from random import shuffle
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
	"""
	
	h = _Heap(np.array([0]),empty=True)
	print('push test')
# 	values = np.arange(20)
# 	np.random.shuffle(values)
	values = [7,0,6,1,5,2,4,3] # reverse-sorted tuples
	for i,v in enumerate(values):
		print('pushing',v)
		h.push(v)
		if not h._maintains_invariant():
			print('\nheap invariant broken, starting with a fresh heap and re-pushing values with verbose output')
			h = _Heap(np.array([0]),empty=True)
			for _ in range(i+1):
				h.push(values[_], verbose=True)
			break
	
# 	import sys; sys.exit()
	
	print('sorting large arrays')
	for _ in range(10):
		print_update(f'trial {_+1}/10: build in place')
		values = np.random.randint(0, 1000, 10**6)
		h = Heap(values.copy())
		e = h.empty()
		assert (np.ediff1d(e)>=0).all(), \
			'large heap built initially: array is not sorted'
		assert np.array_equal(e, np.sort(values)), \
			f'large heap built initially: expected:\n{np.sorted(values)}\ngot:\n{e} (size={len(e)})'
		
		print_update(f'trial {_+1}/10: build by push')
		h = Heap(values[:1],empty=True)
		for v in values: h.push(v)
		e = h.empty()
		assert (np.ediff1d(e)>=0).all(), \
			'large heap built by pushing: array is not sorted'
		assert np.array_equal(e, np.sort(values)), \
			f'large heap built by pushing: expected:\n{np.sorted(values)}\ngot:\n{e} (size={len(e)})'
	
# 	_