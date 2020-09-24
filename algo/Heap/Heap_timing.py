"""Simple module to time my `Heap` implementation against inbuilt `heapq` module."""
import numpy as np
from time import process_time
from mpl_wrap.plotting_functions import make_common_axlims

from common.algo.Heap.Heap import Heap

# run all these through once to be sure
# compilation does not get factored into timings
a = np.arange(100)
h = Heap(a)
h.pop()
h.push(0)
h.empty()

from heapq import heappush, heappop, heapify
import matplotlib.pyplot as plt

# build heap in place and empty completely: Heap
def time_compiled_build(values, downsize=True):
	values = values.copy()
	t = process_time()
	h = Heap(values, downsize=downsize)
	h.empty()
	tf = process_time()
	return tf-t

# build heap in place and empty completely: heapq
def time_build(values):
	values = values.tolist()
	t = process_time()
	heapify(values)
	[heappop(values) for _ in range(len(values))]
	tf = process_time()
	return tf-t

# generate set of times for arrays of specified sizes
def time_random_of_exp_size(sizes, **kw):
	print('time_random_of_exp_size')
	arrays = (np.random.randint(0,int(1000*s),int(s)) for s in sizes)
	return np.array([(time_build(v), time_compiled_build(v, **kw)) for v in arrays])

# plot the times and the speedup factor
# factor < 1 indicates heapq is faster, factor > 1 indicates Heap is faster
def plot_times(maxexp=7.3, trials=10, bins=10, ax=None, plotkw={}, **kw):
	sizes = 10**np.linspace(1,maxexp,bins)
	times = np.average([time_random_of_exp_size(sizes, **kw) for _ in range(trials)], axis=0)
	print('times (heapq, Heap):')
	print(times)
	if ax is None:
		ax = plt.subplot()
		no_ax = True
	else:
		no_ax = False
	for ar,l,m in zip(times.T, ('heapq (native module)','Heap (custom numba implementation)'),('o','^')):
		ax.plot(np.log10(sizes),np.log10(ar),label=l, marker=m, ms=12)
	ax.plot([],[],c='g', marker=(6,2,0), ms=12, label = 'speedup')
	ax.set_xlabel(r'${log}_{10}(\rm{input\ array\ size})$',size=16)
	ax.set_ylabel(r'${log}_{10}({time})$',size=16)
	ax.legend(fontsize=16)
	
	ax2 = ax.twinx()
	ax2.plot(np.log10(sizes), np.log10(times[:,0]/times[:,1]), c='g', marker=(6,2,0), ms=12)
	ax2.set_ylabel(r'${log}_{10}({speedup\ factor})$', size = 16)
	
	print('speedup:',times[:,0]/times[:,1])
	
	if no_ax: plt.show()
	return ax2

if __name__ == '__main__':
	fig, (ax1,ax2) = plt.subplots(1,2)
	ax1_twinx = plot_times(7, 5, 10, downsize=True, ax=ax1)
	ax2_twinx = plot_times(7, 5, 10, downsize=False, ax=ax2)
	ax1.set_title('downsize=True', size=20)
	ax2.set_title('downsize=False', size=20)
	make_common_axlims(ax1,ax2)
	make_common_axlims(ax1_twinx,ax2_twinx)
	plt.show()