import numpy as np
from .collection import consume

def unique_in_order(a):
	u,i = np.unique(a,return_index=True)
	return u[np.argsort(i)]

def assign(array,indices,values):
	def f(i,v): array[i]=v
	consume(f(i,v) for i,v in zip(indices,values))

def get_regions(ar,threshold):
	"""
	find subarrays in a 1d sorted array that are related, i.e. where
	the distances between consecutive elements of the subarray do not surpass threshold
	"""
	return np.split(ar,np.where(np.ediff1d(ar)>threshold)[0]+1)

def split_sorted_1d_coordinates_contiguous(array):
	"""
	given a list of sorted 1-d coordinates,
	split them into contiguous chunks (subarrays)
	"""
	return np.array([chunk[[0,-1]] for chunk in get_regions(array,1)])

def get_contiguous_1d_regions(array):
	"""
	treating an array as a boolean array where 0 indicates discontinuity,
	get contiguous regions of the array
	"""
	return split_sorted_1d_coordinates_contiguous(np.where(array)[0])
