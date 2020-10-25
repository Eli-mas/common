import numpy as np
from numba import njit

@njit
def partition_on_value(data, sep, start=0, end=-1):
	"""Partition an array into values <= sep, values > sep.
	
	Partition a dataset such that all values <= `sep` are to the left
	of the separating index, and all values greater are to the right.
	If either of `start` and `end` are supplied, the partitioning only occurs
	on the region data[start:end+1]. By default, start = 0, end = len(data)-1.
	
	Returned by this routine is the last index where data has a value <= `sep`.
	If there is no such value (all values in the array > `sep`), start-1 is returned.
	"""
	if start<0:
		raise IndexError('specify start >= 0')
	if end >= len(data):
		raise IndexError('specify end < len(data)')
	
	if end == -1:
		end = len(data)-1
	elif start>end:
		raise IndexError('specify start <= end')
	
	first_high = start
	highest = end+1 # stop if we reach this index
	while data[first_high] <= sep:
		# first index where data[i] > sep
		first_high+=1
		if first_high == highest:
			return end - 1
	
	last_low = end
	lowest = start-1 # stop if we reach this index
	while data[last_low] > sep:
		# last index where data[i] <= sep
		last_low -= 1
		if last_low == lowest:
			return lowest
	
	while first_high - last_low != 1:
		data[first_high], data[last_low] = data[last_low], data[first_high]
		while data[first_high] <= sep:
			# first index where data[i] > sep
			first_high+=1
		while data[last_low] > sep:
			# last index where data[i] <= sep
			last_low -= 1
	
	return last_low

# @njit
def collect_forwards(array, start=0, end=-1):
	"""Partition an array into values == array[start], values != array[start].
	
	Partition an array such that all repeats of the value at array [index]
	are moved to be contiguous at the start, with other values pushed to the
	right. The partitioning occurs over array[start:end+1]. If `end` is not
	specified, it is set to len(array)-1.
	
	The logic is exactly the same as in 'partition_on_value'.
	The difference is the use of == and != operators in place of <= and >.
	"""
	v = array[start]
	
	if end==-1:
		end = len(array) - 1
	
	first_unequal = start
	highest = end+1
	while data[first_unequal] == v:
		# first index where data[i] != v
		first_unequal+=1
		if first_unequal == highest:
			return end - 1
	
	last_equal = end
	lowest = start-1 # stop if we reach this index
	while data[last_equal] != v:
		# last index where data[i] == v
		last_equal -= 1
		if last_equal == lowest:
			return lowest
	
	while first_unequal - last_equal != 1:
		data[first_unequal], data[last_equal] = data[last_equal], data[first_unequal]
		while data[first_unequal] == v:
			# first index where data[i] != v
			first_unequal+=1
		while data[last_equal] != v:
			# last index where data[i] == v
			last_equal -= 1
	
	return last_equal

# @njit
def collect_backwards(array, start, end):
	"""Partition an array into values != array[end], values == array[end].
	
	This implementation of calls 'collect_forwards' on array[::-1], assuming
	that `array` is a numpy array, and that array[::-1] is a view on it,
	so that in-place modifications work as expected.
	"""
	l = len(array)
	view = array[::-1].copy()
	print('collect_backwards: view.flags.writeable:',view.flags.writeable)
	i = collect_forwards(view, l-end-1, l-start-1)
	return l-i-1

def find_median(data, expect_repeat_median = False):
	"""Deterministic serial algorithm for finding the median of a dataset.
	
	Similar to randomized quick select in that it works by partitioning the data
	iteratively and narrowing it down until the partition occurs on the median
	index, but instead of choosing a random value from the dataset, it chooses
	the partition as the halfway point between the min and max of the current
	window of the dataset being examined at each iteration.
	
	The runtime is determined by the reduction in the size of the window being
	examined at each iteration. Reducing the dataset exponentially/linearly
	yields quadratic/linear run time.
	
	Worst case, O(n^2), occurs data are exponentially distributed: e.g.
	>>> [1, 3, 9, 27, 81, 243 ...] # 3^n
	>>> [1, 3/2, 9/4, 27/8, ...]   # (3/2)^n
	
	In this case, the size of the data subset examined at each iteration
	will be reduced by constant amount. Ideally the dataset is approximately halved
	at each iteration, leading to O(n) runtime; this occurs when the data are
	uniformly distributed.
	
	In a naive implementation of this algorithm, a set of data with the same
	value n times (e.g. [1,1,1,1,1, ...]) would take Θ(n^2) time; the same
	holds for any array where the median of the array repeats O(n) times.
	This implementation contains logic to ensure that this case is handled in
	O(n) time rather than O(n^2). This logic, however, makes constants worse,
	and is not desirable if the number of repeats of any value is on average
	expected to be rather low; so it is disabled by default. To enable it,
	set `expect_repeat_median` to True.
	
	TODO: the logic of checking for repeat values encounters a problem in the
	'collect_backwards' function when 'collect_forwards' is njit-compiled.
	Specifically, numba reports the view array[::-1] as being readonly,
	although when collect_backwards is not run with njit compilation,
	it reports view.flags.writeable = True. The result is that the functions
	work but cannot be compiled--not acceptable for peformance.
	Possible solutions:
		* Prevent the array from becoming read-only
		  (seems internal to numba, so might not be an option)
		* Rewrite 'collect_backwards' to avoid taking a view, meaning it has to
		  implement the full logic explicitly--undesirable if avoidable.
		* Extend 'partition_on_value' to include an option for handling
		  the logic of 'collect_backwards' and 'collect_forwards' itself:
		  it can call itself with the `sep` parameter being set to data[i]
		  and the bounds being chosen appropriately.
		  OR call 'partition_on_value' separately within 'find_median' to
		  handle this behavior.
		  This seems like the most desirable option--it removes the need for
		  separate 'collect_backwards' and 'collect_forwards' routines.
	"""
	target_i = len(data)//2
	
	print('find_median:', data)
	print('target_i:', target_i)
	low, high = 0, len(data)-1
	
	if not expect_repeat_median:
		while True:
			ds = data[low:high+1]
			p = .5*(ds.min() + ds.max())
			print('partitioning on:', p, 'low, high:', low, high)
			i = partition_on_value(data, p, low, high)
			print(f'\ti={i}', data)
			if i==target_i:
				break
			if i < target_i:
				# It is possible that the middle of the min and max is >
				# the first element in the window and < all the others.
				# In this case, setting low=i would be setting low=low,
				# leading to an infinite loop; to avoid, set low=i+1.
				if i==low:
					low = i+1
				else:
					low = i
			else:
				# require analogous logic here?
				high = i
		
		if len(data)%2 == 1:
			return data[i]
		else:
			return .5 * (data[i] + np.max(data[:i]))
	
	else:
		# same logic as above, with the extra work of addressing repeat values
		while True:
			ds = data[low:high+1]
			p = .5*(ds.min() + ds.max())
			print('partitioning on:', p, 'low, high:', low, high)
			i = partition_on_value(data, p, low, high)
			print(f'\ti={i}', data)
			if i==target_i:
				break
			
			# this requires swapping values such that all repeat values
			# are placed next to one another; this also may have to be handled
			# separately in the cases below
			
			if i < target_i:
				# collect all repeats of the value at data[i] to the beginning
				# of the current window, and adjust the lower bound accordingly
				i = collect_forwards(data, i, high)
				# if the value straddles the median index, return it
				if i >= target_i:
					return data[i]
				if i==low:
					low = i+1
				else:
					low = i
			else:
				# same idea, but collect values at the end rather than beginning
				i = collect_backwards(data, low, i)
				if i <= target_i:
					return data[i]
				high = i
		
		if len(data)%2 == 1:
			return data[i]
		else:
			return .5 * (data[i] + np.max(data[:i]))

__all__ = ('partition_on_value','find_median')

if __name__ == '__main__':
# 	data = np.array([-1, 0, 4, 10, 20, 19, 7, 3, 2, 1, 5, 13, 12])
	data = np.ones(20)
	dc = data.copy()
# 	print(f'data: {data}\npartition: i={partition_on_value(data, 6)}, {data}')
	
# 	median = find_median(dc)
# 	print(f'{dc}\n\tmedian: {median}')
# 	assert median == np.median(dc), f'find_median: got {median}, expected {np.median(dc)}'
	
	median = find_median(dc[1:], expect_repeat_median = True)
	print(f'{dc[1:]}\n\tmedian: {median}')
	assert median == np.median(dc[1:]), \
		f'find_median: got {median}, expected {np.median(dc[1:])}'
	
	