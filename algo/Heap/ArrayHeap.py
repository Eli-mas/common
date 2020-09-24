"""ArrayHeap extends the basic Heap implementation to act on arrays.
The primary purpose of this module is to support the heap-based merging of sorted
arrays. Currently the _ArrayMegerHeap class has been tested on ints and floats.

To do:
    - make a factory meathod as in Heap.py that will compile _ArrayMegerHeap
	  for differnt input types
	...
"""

from numba.experimental import jitclass
from numba import typed, typeof
import numba as nb
from numba import types


class _ArrayMegerHeap:
	def _element_lt(self, element1, element2):
		return element1[0] < element2[0]
	
	def _index_lt(self, index1, index2):
		return self._array[index1][0] < self._array[index2][0]
	
	def _element_gt(self, element1, element2):
		return element1[0] > element2[0]
	
	def _index_gt(self, index1, index2):
		return self._array[index1][0] > self._array[index2][0]
	
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
		if len(self._array)==self.size:
			self._array.extend(self._array)
		self._array[i] = value
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
# 		print('\t__maintain_invariant: index =',i)
# 		print('\tarray:',self._array)
		if not self._isleaf(i):
			parent = self._array[i]
			left_index = (i<<1) + 1
			right_index = (i<<1) + 2
			left = self._array[left_index]
			if right_index == self.size:
				if self._element_gt(parent, left):
					j = left_index
# 					print('\t\theap inconsistency:',j)
					temp = self._array[j]
					self._array[j] = self._array[i]
					self._array[i] = temp
# 					print('\t\tarray after swap:',self._array)
					return j
			else:
				right = self._array[right_index]
# 				print(f'\t\tparent (index={i}): {parent}, left(index={(i<<1) + 1}) = {left}, right(index={(i<<1) + 2}) = {right}')
# 				print('values of parent, left, right:',parent, left, right)
				if self._element_gt(parent, left) or self._element_gt(parent, right):
					if self._element_lt(left, right):
						j = left_index
					else:
						j = right_index
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
	
# 	def empty(self):
# 		"""Pop all values off the heap in sorted order, returning an array"""
# 		return np.array([self.pop() for _ in range(self.size)])
# 	
# 	def __str__(self):
# 		return str(self._array[:self.size])
	
	def merge(self):
# 		print('arrays:')
# 		for a in self._array:
# 			print(f'\t{a}')
		total_size = 0
		for item in self._array:
			total_size += len(item)
		if total_size==0:
			return np.empty(0, dtype = self._array[0].dtype)
		
		out = np.empty(total_size, dtype = self._array[0].dtype)
		
		for i in range(total_size):
			array = self.pop()
			if not self._maintains_invariant():
				raise ValueError('after pop, before push: invariant not maintained')
			
			out[i] = array[0]
			if len(array)>1:
				self.push(array[1:])
		
		return out
	
	def _maintains_invariant_at_index(self, i):
		"""Verify that a parent is <= its children"""
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
# 		print('\t_maintains_invariant: starting at index', (self.size-1) >> 1, 'and working backwards')
		for i in range((self.size-1) >> 1, -1, -1):
			if not self._maintains_invariant_at_index(i):
				print('\tinvariant broken at index', str(i)+',', 'value', self._array[i],
# 					  'children {self.get_children(i)}, array {self._array}'
					  )
				return False
		return True

ArrayHeap = jitclass(
	{'size':nb.int_,'_downsize':nb.int_,
	'_array':nb.types.ListType(nb.int_[::1])}
)(_ArrayMegerHeap)
if __name__ == '__main__':
	import numpy as np
	
	source_lists = [
		np.sort(np.random.randint(0, 1000, np.random.randint(100,1000)))
		for _ in range(100)
	]
	
	h = ArrayHeap(typed.List(source_lists[:]))
	popped = [h.pop()[0] for _ in range(h.size)]
	assert (np.ediff1d([a for a in popped]) >= 0).all(), \
		'heap built initially: the arrays\' initial elements are not in order'
	
	h = ArrayHeap(typed.List(source_lists[:1]),empty=True)
	for a in source_lists: h.push(a)
	popped = [h.pop()[0] for _ in range(h.size)]
	assert (np.ediff1d([a for a in popped]) >= 0).all(), \
		'heap built by pushing: the arrays\' initial elements are not in order'
	
	h = ArrayHeap(typed.List(source_lists[:]))
	print(f'merger: len(h._array) = {len(h._array)}, # total elements = {sum(map(len, h._array))}')
	out = h.merge()
	assert np.array_equal(out,np.sort(np.concatenate(source_lists)))
	print('merger: assertion complete')