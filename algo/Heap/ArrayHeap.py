"""ArrayHeap extends the basic Heap implementation to act on arrays.
The primary purpose of this module is to support the heap-based merging of sorted
arrays. Currently the _ArrayHeap class has been tested on ints and floats.

To do:
    - make a factory meathod as in Heap.py that will compile _ArrayHeap
	  for different input types
	...
"""

from numba.experimental import jitclass
from numba import typed, typeof
import numba as nb
from numba import types
import numpy as np


class _ArrayHeap:
	# These next few methods mark the difference between the basic heap
	# and the heap of arrays: assigning order of heap elements is based
	# on comparisons between the first values of those elements.
	# Note we only need to compare the first values, not any subsequent values.
	# In place of 'self._array[i] < self._array[j]', we use
	# self._index_lt(i, j), or self._element_lt(self._array[i], self._array[j])
	def _element_lt(self, element1, element2):
		return element1[0] < element2[0]
	
	def _index_lt(self, index1, index2):
		return self._array[index1][0] < self._array[index2][0]
	
	def _element_gt(self, element1, element2):
		return element1[0] > element2[0]
	
	def _index_gt(self, index1, index2):
		return self._array[index1][0] > self._array[index2][0]
	
	"""def _array_lt(self, arr1, arr2):
		for i in range(self.k):
			if arr1[i]<arr2[i]:
				return True
		return False
	
	def _array_gt(self, arr1, arr2):
		for i in range(self.k):
			if arr1[i] > arr2[i]:
				return True
		return False"""
	
	def __init__(self, data, downsize = True, empty=False):#k=1, 
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
# 		self._array.extend(data)
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
		i = self.size
		# resize array if needed
		if len(self._array)==self.size:
			self._array.extend(self._array)
		
		# add the value to the end to start
		self._array[i] = value
		
		# now percolate it up to the extent required: repeatedly swap with
		# ascending predecessors until the invariant is maintained
		parent_index = (i-1)>>1
		while (i > 0) and self._index_lt(i, parent_index):
			temp = self._array[i]
			self._array[i] = self._array[parent_index]
			self._array[parent_index] = temp
			i = parent_index
			parent_index = (i-1)>>1
		self.size += 1
	
	def pop(self):
		"""Remove the minimal value from the heap and restore the invariant.
		Time is O(lg n). Popping all the values off returns them in sorted
		order."""
		if self.size == 0:
			raise ValueError('cannot pop element off empty heap')
		
		# smallest element is always top element
		popped = self._array[0]
		
		if self.size==1:
			self.size = 0
			return popped
		
		self._array[0] = self._array[self.size-1]
		self.size -= 1
		self._maintain_invariant(0)
		"""if self._downsize and (self.size < (len(self._array)>>1)):
			new = np.zeros(self.size, dtype=self._array.dtype)#
			new[:] = self._array[:self.size]
			self._array = new"""
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
		# by default invariant is maintained at leaves, so if we find a leaf
		# there is no need to check whether it maintains the invariant
		if not self._isleaf(i):
			parent = self._array[i]
			left_index = (i<<1) + 1 # (i*2) + 1
			right_index = (i<<1) + 2 # (i*2) + 2
			left = self._array[left_index]
			
			# valid heap indices range is [0, size-1]
			# don't consider right index if it exceeds range of heap
			if right_index == self.size:
				if self._element_gt(parent, left):
					j = left_index
					temp = self._array[j]
					self._array[j] = self._array[i]
					self._array[i] = temp
					return j
			else:
				# consider left and right children
				right = self._array[right_index]
				if self._element_gt(parent, left) or self._element_gt(parent, right):
					# if parent > either of children, swap with smaller child
					if self._element_lt(left, right):
						j = left_index
					else:
						j = right_index
					temp = self._array[j]
					self._array[j] = self._array[i]
					self._array[i] = temp
					return j
		return -1
	
	def _maintain_invariant(self, i):
		"""Ensure the heap invariant for the root at position `i`.
		Uses an iterative solution rather than a recursive one because
		(1) it is more efficient anyway and (2) numba (0.5.1) does not yet
		support re-entrant calls on methods of compiled classes."""
		
		while i!=-1: # return signal to indicate that invariant is ensured
			i = self.__maintain_invariant(i)
	
	def empty(self):
		return typed.List([self.pop() for _ in range(self.size)])
	
	def merge(self):
		"""Given this instance with sorted arrays on the heap,
		merge them via the heap-merge approach. Worst-case time complexity
		for k arrays with sizes n1 > n2 > ... > nk, total elements N,
		= n1(lg(k)) + n2(lg(k-1)) + n3(lg(k-2)) + ... nk(lg(1)) + (n+n2+...+nk)
		= N + sum(n_i * lg(k-(i-1))).from(i=1).to(i=k)
		
		If each array is ~ n in size and there are k arrays, we have
		O(n lg(k)) time.
		"""
		# basic checks: make sure there is something to return
		total_size = 0
		for item in self._array:
			total_size += len(item)
		if total_size==0:
			# no arrays on heap, or no values in arrays
			return np.empty(0, dtype = self._array[0].dtype)
		
		# pre-initialize to avoid cost of dynamic resizing
		out = np.empty(total_size, dtype = self._array[0].dtype)
		
		for i in range(total_size):
			# get the array with the smallest first value
			array = self.pop()
			# add smallest value of array to out array
			# room for optimization: we can compute average repeat
			# amount above when figuring out total_size. If it is high,
			# we add all repeat values here at once rather than repeatedly
			# pushing and popping from heap.
			out[i] = array[0]
			if len(array)>1:
				self.push(array[1:])
		
		return out
	
	def _maintains_invariant_at_index(self, i):
		"""Test method: verify that a parent is <= its children"""
		parent_value = self._array[i]
		left_index = (i<<1) + 1
		right_index = (i<<1) + 2
		if (left_index < self.size) and self._index_gt(i, left_index):
			print('\t!!! parent @ index',i,'> left child @ index', left_index)
			return False
		if (right_index < self.size) and self._index_gt(i, right_index):
			print('\t!!! parent @ index',i,'> right child @ index', right_index)
			return False
		return True
	
	def _maintains_invariant(self):
		"""Test method: verify that invariant is maintained throughout heap."""
		for i in range((self.size-1) >> 1, -1, -1):
			if not self._maintains_invariant_at_index(i):
				print('\tinvariant broken at index', str(i)+',', 'value', self._array[i],
# 					  'children {self.get_children(i)}, array {self._array}'
					  )
				return False
		return True

def merge(arrays):
	return ArrayHeap(typed.List(arrays).merge())

ArrayHeap = jitclass(
	{'size':nb.int_,'_downsize':nb.int_,
	'_array':nb.types.ListType(nb.int_[::1])}
)(_ArrayHeap)
if __name__ == '__main__':
	import numpy as np
	from common import print_update
	
	source_lists = [
		np.sort(np.random.randint(0, 1000, np.random.randint(100,1000)))
		for _ in range(100)
	]
	
	print_update('large array: built initially')
	h = ArrayHeap(typed.List(source_lists[:]))
	popped = [h.pop()[0] for _ in range(h.size)]
	assert (np.ediff1d([a for a in popped]) >= 0).all(), \
		'heap built initially: the arrays\' initial elements are not in order'
	print_update('merger: assertion complete')
	
	print_update('large array: built by pushing')
	h = ArrayHeap(typed.List(source_lists[:1]),empty=True)
	for a in source_lists: h.push(a)
	popped = [h.pop()[0] for _ in range(h.size)]
	assert (np.ediff1d([a for a in popped]) >= 0).all(), \
		'heap built by pushing: the arrays\' initial elements are not in order'
	print_update('merger: assertion complete')
	
	h = ArrayHeap(typed.List(source_lists[:]))
	print_update(f'merger: len(h._array) = {len(h._array)}, # total elements = {sum(map(len, h._array))}')
	out = h.merge()
	assert np.array_equal(out,np.sort(np.concatenate(source_lists)))
	print_update('merger: assertion complete')
	
	print_update('test on a simple array')
	a = list(np.sort(np.random.randint(0,10,[20,2]),axis=1))
	h = ArrayHeap(typed.List(a[:]))
	out = h.merge()
	assert np.array_equal(out,np.sort(np.concatenate(a))), \
		f'expected:\n{np.sort(np.concatenate(a))}\ngot:\n{out}'
	print_update('merger: assertion complete')
	
	print_update('1-element ordering test: simple array')
	h = ArrayHeap(typed.List(a[:]))
	popped = h.empty()
	assert (np.ediff1d([e[0] for e in popped]) >= 0).all(), \
		f'array is not sorted by first column:\n{np.array(popped)}'
	print_update('merger: assertion complete')
	
	print_update('1-element ordering test: source_list')
	h = ArrayHeap(typed.List(a[:]))
	popped = h.empty()
	try:
		assert (np.ediff1d([e[0] for e in popped]) >= 0).all()
	except AssertionError:
		join = '\n'.join(map(source_lists,str))
		raise AssertionError(f'array is not sorted by first column:\n{join}')
	print_update('merger: assertion complete')
	
	print()
	
	