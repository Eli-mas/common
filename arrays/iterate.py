import numpy as np

def rollaxes(array, source, dest):
	"""transpose an array by rolling the axes such that axis `source` arrives at axis `dest`"""
	if dest < 0: dest = array.ndim + dest
	if source < 0: source = array.ndim + source
	print(f'rollaxes: shape={array.shape}, t={np.roll(np.arange(array.ndim), dest - source)}')
	return np.transpose(array, np.roll(np.arange(array.ndim), dest - source))

def access_1d_strands(array, axis):
	"""Get a 2-dimensional reshape of an array transpose such that
	iterating over axis 0 of the transpose-reshape yields every
	1-d array embedded in the original array when traversing along
	the dimension specified by `axis`."""
	
	# the amount to roll is determined by dest - source
	# here: dest, source = -1, axis
# 	shape = np.roll(array.shape, -1 - axis)
	new = np.moveaxis(array, axis, -1)
	shape = new.shape
# 	return rollaxes(array, axis, -1).reshape(np.prod(shape[:-1]), shape[-1])
	return new.reshape(np.prod(shape[:-1]), shape[-1])

# def access_slice_of_original_in_1d_strand_array(slice, axis, original_shape):
# 	"""Given a slice corresponding to an array with shape `original_shape`,
# 	calculate the slice that would correspond to the new location of the
# 	elements from this slice in the array as transformed by 'access_1d_strands'."""
# 	
# 	if isinstance(slice, int):
# 		# axis is 0
# 		slice_shape = [1, *(original_shape[i] for i in range(1, len(original_shape)))]
# 		new_slice_shape = [*(s for i,s in enumerate(slice_shape) if i != axis), slice_shape[axis]]
# 	
# 	shape = 
	

def recreate_from_1d_strands(strand_array, axis, original_shape):
	"""
	Say that we made this call on an array `array`:
		>>> strand_array = access_1d_strands(array, axis)
	
	We could then retrieve the original array as follows:
		>>> restored = recreate_from_1d_strands(strand_array, axis, array.shape)
	
	Thus, this is always true:
		>>> np.array_equal(
		>>> 	array,
		>>> 	recreate_from_1d_strands(
		>>> 		access_1d_strands(array, axis), axis, array.shape
		>>> 	)
		>>> ) # True
	"""
	if axis < 0: axis = len(original_shape) + axis
	shape = [*(s for i,s in enumerate(original_shape) if i != axis), original_shape[axis]]
# 	print('recreate_from_1d_strands: shape:', shape)
	return np.moveaxis(strand_array.reshape(shape), -1, axis)

def test__recreate_from_1d_strands(array):
	assert all(
		np.array_equal(
			a,
			recreate_from_1d_strands(
				access_1d_strands(a,axis), axis, a.shape
			)
		)
		for axis in range(a.ndim)
	)

def access_nd_strands(array, axes):
	"""Get a reshape/transpose of an array such that iterating
	over axis 0 of the transpose-reshape yields every n-dimensional
	array embedded in the original array when traversing along
	the dimensions specified by `axes`, where n = len(axes).
	The axes may be specified out of order.
	
	Note: if n = array.ndim, the result of this function is equal to
	np.expand_dims(np.transpose(array, axes), 0), which, congruent
	with this function's definitions, yields np.transpose(array, axes)
	in a single iteration over its 0th axis.
	"""
	
	axes = np.atleast_1d(axes)
	new = np.moveaxis(array, axes, range(-len(axes), 0))
	shape = new.shape
# 	print('shapes (orignal, new):',array.shape, new.shape)
# 	print('asserting')
# 	assert np.array_equal(new.shape[-len(axes):], [array.shape[i] for i in axes])
# 	print('access_nd_strands: shape slice product:',np.prod(shape[:-len(axes)]))
	return new.reshape(np.prod(shape[:-len(axes)], dtype=int), *shape[-len(axes):])

def recreate_from_nd_strands(strand_array, axes, original_shape):
	"""
	Say that we made this call on an array `array`:
		>>> strand_array = access_nd_strands(array, axes)
	
	We could then retrieve the original array as follows:
		>>> restored = recreate_from_nd_strands(strand_array, axes, array.shape)
	
	Thus, this is always true:
		>>> np.array_equal(
		>>> 	array,
		>>> 	recreate_from_nd_strands(
		>>> 		access_nd_strands(array, axes), axes, array.shape
		>>> 	)
		>>> ) # True
	"""
	axes = np.atleast_1d(axes)
	axes[axes<0] += len(original_shape)
	shape = [
		*(s for i,s in enumerate(original_shape) if i not in axes),
		*(original_shape[i] for i in axes)
	]
# 	print('recreate_from_1d_strands: shape:', shape)
	return np.moveaxis(strand_array.reshape(shape), range(-len(axes), 0), axes)

def test__recreate_from_nd_strands(a):
	from itertools import permutations
	assert all(
		np.array_equal(
			a,
			recreate_from_nd_strands(
				access_nd_strands(a,axes), axes, a.shape
			)
		)
		for d in range(1,a.ndim+1) for axes in permutations(range(a.ndim), d)
	)

__all__ = ('rollaxes', 'access_1d_strands', 'access_nd_strands',
'test__recreate_from_nd_strands', 'test__recreate_from_1d_strands')


if __name__ == '__main__':
	from itertools import combinations, permutations, product
# 	a = np.arange(2*3*4).reshape(2,3,4)
# 	print('** ** ** original array ** ** **')
# 	print(a)
# 	for i in range(a.ndim):
# 		print('-- -- -- access_1d_strands: axis %i-->0 -- -- --'%i)
# 		print(access_1d_strands(a,i))
# 	print('** ** ** finished ** ** **')
# 	
# 	print('** ** ** original array ** ** **')
# 	print(a)
# 	for axes in permutations(range(a.ndim), 2):
# 		print('-- -- -- access_nd_strands: axes %s-->0 -- -- --'%(axes,))
# 		print(access_nd_strands(a,axes))
# 	print('** ** ** finished ** ** **')
	
	## the following test block meant to be enabled
	## with assertions de-commented in access_nd_strands
	r = range(2,9)
	a = np.arange(np.prod(r)).reshape(r)
	print('** ** ** original array ** ** **')
# 	print(a)
	for d in range(1,a.ndim+1):
		for axes in permutations(range(a.ndim), d):
			print_update('-- -- -- access_nd_strands: axes %s-->0 -- -- --'%(axes,))
			access_nd_strands(a,axes)
	print('** ** ** finished ** ** **')
	