from common.collections import consume
import numpy as np

def minmax(ar, handle_nan = True, flatten = False, **kw):
	if flatten: ar = np.hstack(ar)
	if handle_nan: return np.array((np.nanmin(ar,**kw),np.nanmax(ar,**kw)))
	try: return np.array((ar.min(**kw),ar.max(**kw)))
	except AttributeError: return np.array((np.min(ar,**kw),np.max(ar,**kw)))

def round_to_nearest(value, rounder, mode='round'):
	"""NOTE: floating-point errors can occur here, the resolution is not yet implmenented"""
	if mode=='round': func=np.round
	elif mode=='floor': func=np.floor
	elif mode=='ceil': func=np.ceil
	result=rounder*func(np.atleast_1d(value)/rounder)
	return result.squeeze()#np.round(result, int(np.floor(np.abs(np.log10(rounder%1)))))

def make_object_array(iterable):
	"""
	Sometimes calling
	>>> np.array(iterable,dtype=object)
	
	raises a ValueError (probably a bug that should be submitted to Numpy).
	
	This is a workaround that produces the intended result.
	"""
# 	if hasattr(iterable,'__len__') and len(iterable)<20:
# 		print('called make_object_array on:',iterable)
	ret = np.array([None]*len(iterable)) # dtype is object
# 	ret[:] = iterable[:]
	"""
	For some reason, when initializing an AllSeriesProxy,
	the above results in a list of errors like this:
		AttributeError("'Galaxy' object has no attribute '__array_interface__'")
	
	So the following line is used instead, which accomplishes the same thing,
	but without error
	"""
	consume(ret.__setitem__(i,iterable[i]) for i in range(len(iterable)))
	return ret


__all__ = ('minmax', 'round_to_nearest', 'make_object_array', )
