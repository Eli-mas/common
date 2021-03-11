"""Time- & memory-efficient computation of rolling sum over n-dimensional arrays.

Given a d-dimensional array totaling N scalar elements,
the rolling sum can be computed in (loosely) o(d*N) time and o(N) memory
by iteratively computing rolling window sums over different dimensions.
This module implements this routine. This implementation replaces
the former, slower implementation from roll_wrap_funcs.

Note: there are further optimizations that can be introduced:
	* Check if window elements are 1
	* Re-use final output array for intermediary steps
"""

import numpy as np
# from functools import partial
from .. import consume, tprint, print_update

def _rolling_1d(A, w):
	"""simple roling sum over 1-d array. Demonstrates how a rolling sum
	can be calcualted in linear time over 1-d array. Test function to
	get implementation logic working, for reuse in n-dimensional function."""
	out = np.empty(A.size - w + 1, A.dtype)
	sum = A[:w].sum()
	out[0] = sum
	
	for oi,Ai in enumerate(range(w, A.size)):
		sum -= A[oi]
		sum += A[Ai]
		out [oi+1] = sum
	
	return out


def rolling_sum_along_dim(ar, axis, size, out=None):
	"""Get a rolling sum for array `ar` along dimension `axis` with window size `size`.
	
	Requires size > 0. size=1 has no effect (function returns immediately).
	Optionally specify `out` to avoid creating a new array.
	
	If size > ar.shape[axis], a ValueError is raised.
	
	Implementation performs the operation in linear time with respect to
	the total number of elements in ar. Takes advantage of the fact that
	a rolling sum can be updated incrementally to avoid recomputing
	intermediary sums more than twice. Specifically, we can start with
	an initial sum of the first `size` elements along the `axis` axis,
	assign this result to the output array, and iteratively subtract
	the first element in the rolling window and add the next element,
	assigning each result to the output. Each element in the input array
	is thus added to the rolling sum exactly once and subtracted at most once.
	"""
	if size==1:
		return ar
	
	if size==0:
		raise ValueError(f"`size` must be strictly > 0 (passed {size})")
	
	ar = np.asarray(ar)
	
	assert np.isscalar(size), f"pass a scalar for `size` (passed {size})"
	assert np.isscalar(axis), f"pass a scalar for `axis` (passed {axis})"
	
	if size > ar.shape[axis]:
		raise ValueError(f"window size {size} exceeds {ar.shape[axis]}"
						 f' = array size along dimension {axis}')
	
	shape = list(ar.shape)
	shape[axis] = ar.shape[axis] - size + 1 # output shape
	
	if out is None:
		out = np.empty(shape, dtype=ar.dtype)
	else:
		if not np.array_equal(out.shape, shape):
			raise ValueError(f"out.shape should be {tuple(shape)}, but is {out.shape}")
	
	# working along axis 0 is simpler, use moveaxis to create a view
	ar_view = np.moveaxis(ar, axis, 0)
	sum = ar_view[:size].sum(axis=0)
	out_view = np.moveaxis(out, axis, 0)
	out_view[0] = sum
	for oi,Ai in enumerate(range(size, ar_view.shape[0])):
		sum -= ar_view[oi]
		sum += ar_view[Ai]
		out_view[oi+1] = sum
	
	return out

def rolling_sum(ar, window):
	"""Calculate a multidimensional rolling-window sum on an array.
	
	`ar` is the input ndarray, and `window` is a container of rolling window
	sizes along each dimension of the array; thus `window` must be 1-d
	and have size == ar.ndim.
	
	If window[i]>ar.shape[i] for any i, a ValueError is raised.
	
	If `ar` has n dimensions and N total elements, the running time is
	(loosely) O(n*N). More precisely, if the dimensions of ar are given in
	descending order by (d1, d2, ..., dn), and `window` has sizes
	(w1, w2, ..., wn) along these dimensions, none of which are equal to 1,
	the running time is
	
	O(P(d1,d2,...,dn) +
	  P(d1-w1,d2,...,dn) +
	  P(d1-w1,d2-w2,...,dn)+
	  ... +
	  P(d1-w1, d2-w2, ..., d(n-1)-w(n-1), dn)
	)
	
	Where P(...) indicates a product of all arguments to P.
	In practice, this will tend towards O(N*n), unless the window sizes
	reduce the output dimensions to O(1) terms.
	
	Auxiliary space is loosely O(N), more precisely
	Theta((d1-w1) * (d2-w2) * ... * (dn-wn)). An auxiliary
	array is allocated once to host all intermediate calculations, and a copy
	is made from a subset of it at the end to delete a reference to it.
	
	If some window sizes are 1, then (d1, d2, ..., dn) and (w1, w2, ..., wn)
	are replaced with respective tuples that exclude all dimensions where the
	window is 1. If the resultant number of non-1 window sizes, the running
	time is loosely O(k*N), and has a full expression analogous to the full
	analysis given above.
	"""
	ar = np.asarray(ar)
	window = np.asarray(window)
	if window.shape != (ar.ndim,):
		raise ValueError('`window` must have 1-d shape with size {ar.ndim}')
	
	dim_dif = ar.shape - np.array(window)
	if np.any(dim_dif < 0):
		where = (dim_dif < 0).nonzero[0]
		c, d = ('', where[0]) if where.size==1 else ('s', tuple(where))
		raise ValueError(f'window is too large along dimension{c} {d}')
	
	i = np.argsort(dim_dif)[::-1]
	dim_dif+=1
	dim_dif = np.asarray(dim_dif)[i]
	window = window[i]
	
	shape = np.array(ar.shape)
	shape[i[0]] += (1-window[0])
	
	# this is why we reverse-sorted dim_dif (and window) before
	out = np.empty(shape, dtype = ar.dtype)
	
	for axis, size, d in zip(i, window, dim_dif):
		# TO DO: check whether the size is 1
		# a new array at each stage
		
		# TO DO: out parameter is not working
		out = np.moveaxis(np.moveaxis(out, axis, 0)[-d:], 0, axis)
		ar = rolling_sum_along_dim(
			ar, axis, size, #out=out
		)
	
	# ar holds a reference to out, which is auxiliary
	return ar.copy()

if __name__ == '__main__':
	from itertools import permutations, product
	
	if False:
		def _1d_test(A, w, dtype=None):
			if dtype is not None:
				A = np.asarray(A, dtype=dtype)
# 			print(A, _rolling_1d(A, 4), sep=' --> ')
			eq = np.array_equal if A.dtype.name.startswith('int') else np.allclose
			assert eq(_rolling_1d(A, w), [A[i:i+w].sum() for i in range(A.size-w+1)]), \
				f'{A}: {_rolling_1d(A, w)}, {np.array([A[i:i+w].sum() for i in range(A.size-w+1)])}'
		
		print_update('testing 1d logic...')
		
		consume(
			_1d_test(A, w=w, dtype=dtype) for A in (
				np.ones(10),
				np.arange(10),
				np.array([1,4,2,7,8,9,4,5,6,3,0]),
				*(np.array(a) for a in permutations(range(7)))
			) for w in range(1, A.size+1)
			  for dtype in (int, float)
		)
		
		print_update('')
	
	if False:
		a = np.tile(np.arange(5)[:, np.newaxis], [1,5])
		for ar in (a,a.T):
			print('test array:', ar, sep='\n')
			for axis in range(ar.ndim):
				for size in range(1, ar.shape[axis]+1):
					tprint(f'axis={axis}, w={size}:', rolling_sum_along_dim(ar, axis, size))
	
		ar = a = np.arange(3*4*5).reshape(3,4,5)
		print('test array:', ar, sep='\n')
		for axis in range(ar.ndim):
			for size in range(1, ar.shape[axis]+1):
				tprint(f'axis={axis}, w={size}:', rolling_sum_along_dim(ar, axis, size))
	
	# use the other array 
	from ..arrays.roll_wrap_funcs import rolling_sum as rs1
	rs2 = rolling_sum
	
	def compare_rs(*dims):
		a = np.arange(np.prod(dims)).reshape(dims)
		ranges = [range(1, (d+1)//2) for d in dims]
		for window in product(*ranges):
			window = np.asarray(window)
			r1 = rs1(a, window, range(a.ndim), squeeze=False)
			r2 = rs2(a, 2*window+1)
			assert np.array_equal(r1,r2), \
				f"r1 != r2: window {2*window+1}, a:\n{a}\nr1:\n{r1}\nr2:\n{r2}\n{r1-r2}"
		print_update('')
	
	
	compare_rs(5,7,9)
	compare_rs(3,3,3)
	compare_rs(1,1,1)
	compare_rs(2,2,2)
	compare_rs(1,2,3,4)
	compare_rs(3,4,5,6)
	compare_rs(3,2,5,7,3)
	compare_rs(1,)
	compare_rs(3,)
	compare_rs(10,)
	compare_rs(10,20,30)
	compare_rs(3,4,5,7,3,4,6)
	print('rolling_sums: tests finished')
