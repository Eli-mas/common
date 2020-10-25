"""ArrayHeap extends the basic Heap implementation to act on arrays.
The primary purpose of this module is to support the heap-based merging of sorted
arrays. Currently the _ArrayHeap class has been tested on ints and floats.

To do:
    - make a factory meathod as in Heap.py that will compile _ArrayHeap
	  for different input types
	...
"""

from numba.experimental import jitclass
from numba import typed, typeof, int_
from numba.types import ListType
import numpy as np
from common.decorators import assign

from .__ArrayHeap_functions__ import __functions__

@assign(__functions__)
class _ArrayHeap:
	# These next few methods mark the difference between the basic heap
	# and the heap of arrays: assigning order of heap elements is based
	# on comparisons between the first values of those elements.
	# Note we only need to compare the first values, not any subsequent values.
	# In place of 'self._array[i] < self._array[j]', we use
	# self._index_lt(i, j), or self._element_lt(self._array[i], self._array[j])
	def _element_lt(self, element1, element2):
		"""Return whether a given array compares less than another array,
		with the comparison defined by comparison of first elements."""
		return element1[0] < element2[0]
	
	def _index_lt(self, index1, index2):
		"""Return whether the array at a given index compares less than the
		array at another index, with the comparison defined by comparison of
		first elements."""
		return self._array[index1][0] < self._array[index2][0]
	
	def _element_gt(self, element1, element2):
		"""Return whether a given array compares greater than another array,
		with the comparison defined by comparison of first elements."""
		return element1[0] > element2[0]
	
	def _index_gt(self, index1, index2):
		"""Return whether the array at a given index compares greater than the
		array at another index, with the comparison defined by comparison of
		first elements."""
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

def merge(arrays):
	return ArrayHeap(typed.List(arrays).merge())

ArrayHeap = jitclass(
	{'size':int_,'_downsize':int_,
	'_array':ListType(int_[::1])}
)(_ArrayHeap)


@assign(__functions__)
class _ArrayMedianHeap:
	"""Heap on sorted arrays that are compared by median values over windows.
	
	The median of a sorted array over a given window is calculated as
	>>> array[(high+low) // 2]
	
	if it has an odd number of elements, this is the proper median;
	for an even-sized array, it is the left median.
	
	The low and high values can be adjusted to adjust the window
	over which the median is sought. The reasons for doing this involves
	an algorithm that makes use of this data structure.
	
	Because numba allows for reading/writing data beyond the valid bounds of
	Python arrays, a trick is used in this class to store the low/high bounds:
	they are stored immediately beyond the range of data that Python recognizes
	an array as having. The constructor ensures this logic is followed.
	The undesirable part is that this requires copying all arrays that are
	received; I will consider later how to address this.
	"""
	def element_median_index(self, element)
		"""Return the index of the median of an element array
		over its current window."""
		n = len(element)
		return (element[n] + element[n+1]) // 2
	
	def index_median_index(self, index)
		"""Return the index of the median the array at a given index
		over the array's current window."""
		element = self._array(index)
		n = len(element)
		return (element[n] + element[n+1]) // 2
	
	def element_median(self, element):
		"""Return the median of an element array over its current window."""
		n = len(element)
		# the low/high bounds are stored at memory addresses
		# beyond what Python recognizes to be valid, but the
		# class constructor operates in such a way so as to ensure
		# that the desired values will indeed be stored there,
		# and numba allows this to be done because it operates
		# at the C level.
		return element[(element[n] + element[n+1]) // 2]
	
	def index_median(self, index):
		"""Return the median of an the array at a given index
		over its current window."""
		element = self._array(index)
		n = len(element)
		return element[(element[n] + element[n+1]) // 2]
	
	def element_medians(self):
		"""Return the medians of all arrays over their current windows."""
		return np.array([self.element_median(e) for e in self._array], dtpye=int)
	
	def element_median_indices(self):
		"""Return the indices of the medians of all arrays
		over their current windows."""
		return np.array([self.element_median_index(e) for e in self._array], dtpye=int)
	
	def element_low_high(self, element):
		"""Return the low and high bounds (both inclusive) defining the
		current window for an element array."""
		n = len(element)
		return np.array((element[element[n]], element[element[n+1]]), dtype=int_)
	
	def index_low_high(self, index):
		"""Return the low and high bounds (both inclusive) defining the
		current window for an element array at the given index."""
		element = self._array[index]
		n = len(element)
		return np.array((element[element[n]], element[element[n+1]]), dtype=int_)
	
	def elements_lows_highs(self):
		"""Return the low and high bounds (both inclusive) defining the
		current window for all arrays on the heap."""
		return np.array([self.element_low_high(e) for e in self._array], dtype=int_)
	
	def _element_lt(self, element1, element2):
		"""Return whether a given array compares less than another array,
		with the comparison defined by comparison of medians."""
		return element_median(element1) < element_median(element2)
	
	def _index_lt(self, index1, index2):
		"""Return whether the array at a given index compares less than the
		array at another index, with the comparison defined by comparison of
		medians."""
		return element_median(self._array[index1]) < element_median(self._array[index2])
	
	def _element_gt(self, element1, element2):
		"""Return whether a given array compares greater than another array,
		with the comparison defined by comparison of medians."""
		return element_median(element1) > element_median(element2)
	
	def _index_gt(self, index1, index2):
		"""Return whether the array at a given index compares greater than the
		array at another index, with the comparison defined by comparison of
		medians."""
		return element_median(self._array[index1]) > element_median(self._array[index2])

ArrayMedianHeap = jitclass(
	{'size':int_,'_downsize':int_,
	'_array':ListType(int_[::1])}
)(_ArrayMedianHeap)


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
	
	