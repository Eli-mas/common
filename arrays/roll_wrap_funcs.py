import numpy as np
from functools import partial
from skimage.util import view_as_windows

from numpy import nan

def rolling_sum(ar, half, axis, weights=None, weighter=None, pr=False, squeeze=True):
	"""
	Take a rolling sum along a given axis/axes
	:param ar: input array
	:param half: the length-from-center of the rolling window; the window has size 2*half+1
	:param axis: axis along which to take the sum; can be an int or tuple of ints
	:param weights: weights for each window sampled
	:param weighter: ... ?
	:param pr: print updates to calculation progress
	:param squeeze:
	:return:
	"""
	a = np.copy(ar)
	# the window should have value 1 along any axes
	# where we are not taking a rolling aggregate
	w = np.ones(a.ndim, dtype=int)
	axis = np.atleast_1d(axis)
	window = 2*np.atleast_1d(half)+1
	w[axis] = window
	
	newaxis = axis+a.ndim
	
	if len(window) != len(axis):
		raise ValueError('`window` and `axis` must have equal lengths')
	
	"""
	to understand why this works:
	
	view_as_windows returns a rolling-window view into a
	such that if
		>>> v = view_as_windows(a, window)
		>>> s = np.atleast_1d(a.shape)
	
	where d == len(window) == a.ndim, then
		>>> np.array(v.shape) == np.append(1 + s - window, window)
		>>> v[i_0, i_1, ..., i_d] == a[
		... 	i_0 : i_0 + window[0],
		... 	i_1 : i_1 + window[1], 
		... 	...,
		... 	i_d : i_d + window[d]
		... ] \
	
	In words, indexing into v at some index with size a.ndim yields the
	same result as taking a slice of a at that index with the slice having
	the dimensions of `window`.
	
	So if we want to perform an aggregation over each location of the rolling
	window in a, what we really want to do is perform the same aggreagation over
	the equivalent location in v; this means aggregating over the last d axes
	of v.
	"""
	view = view_as_windows(a, w)
	if pr: print(
		'input array',a,'axis',axis,'newaxis',newaxis,
		'window',window,'w',w,'view <shape:%s>'%(view.shape,),view,
		sep='\n'
	)
	
	if weights is not None:
		weights = np.atleast_1d(weights)
		wshape = weights.shape
		if not np.array_equal(wshape, window):
			raise ValueError('weights must have equal number of dimensions to window dimensions')
		
		if pr: print('view shape',view.shape,'weights (initial)',weights,'wshape',wshape, sep='\n')
		
		wshape_new = np.ones(2*a.ndim, dtype=int)
		wshape_new[newaxis] = wshape
		weights = weights.reshape(wshape_new)
		if pr: print('wshape_new',wshape_new,'weights (reshaped)',weights, sep='\n')
		weights = np.broadcast_to(weights, view.shape)
		
		view = view * weights
		"""
		!	!	!
		Note, the above line CANNOT be written as the following:
			>>> view *= weights
		
		reason: view_as_windows provides views into the original array,
		and doing an in-place operation when multiple addresses in new array
		map to same locations in original array causes problems
		"""
		if pr: print('weights (broadcasted)',weights,'weights shape',weights.shape,'view (weighted)',view,sep='\n')
		
	elif weighter is not None:
		if weighter.shape!=a.shape:
			raise ValueError('if specified, `weighter` must have equal shape to input array')
		
		view = view * view_as_windows(weighter, w)
	# return weights,view
	res = np.nansum(view, axis=tuple(newaxis))
	if squeeze: res = res.squeeze()
	return res

def rolling_mean(ar, *a, **k):
	sum_res = rolling_sum(ar,*a,**k)

	nancheck = np.ones_like(ar, dtype = float)
	nancheck[np.isnan(ar)] = nan

	nancheck_sum_res = rolling_sum(nancheck, *a, **k)
	
	return sum_res/nancheck_sum_res

def rolling_gmean(ar, *a, **k):
	return 10 ** rolling_mean(np.log10(ar), *a, **k)

def padder_for_roll_wrap(a, half, axis):
	"""
	Pad an array around the specified axes by the given pad sizes
	:param a: input array
	:param half: see argument `half` in 'rolling_sum'
	:param axis: axis/axes along which to pad; int or tuple of ints
	:return: padded array
	"""
	pad_axis=np.zeros([len(a.shape),2],int)
	pad_axis[axis]+=half
	p=np.pad(a,pad_axis,'wrap')
	return p

def _rolling_wrap_template(a,half,axis=0,weights=None,weighter=None,func=None):
	"""
	the error handling within this function assumes that a rolling window will not be requested
	which exceeds the dimensionality of `a` along the requested axis
	"""
	
	if a.shape[axis]<=(2*half+1):
		raise ValueError(
		'`_rolling_wrap_template`: '
		'the size along the requested axis is insufficient'
		)
	if weighter is not None: weighter=padder_for_roll_wrap(weighter,half,axis)
	return func(padder_for_roll_wrap(a,half,axis),half,axis,weights=weights,weighter=weighter)

rolling_sum_wrap=partial(_rolling_wrap_template,func=rolling_sum)
rolling_mean_wrap=partial(_rolling_wrap_template,func=rolling_mean)
rolling_gmean_wrap=partial(_rolling_wrap_template,func=rolling_gmean)


__all__ = ('rolling_sum','rolling_mean','rolling_gmean','rolling_sum_wrap',
'rolling_mean_wrap','rolling_gmean_wrap')
