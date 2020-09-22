"""Simple module to time my `Heap` implementation against Python's native `heapq` module."""
import numpy as np
from time import process_time

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

def time_compiled_build(values):
	values = values.copy()
	t = process_time()
	h = Heap(values)
	h.empty()
	tf = process_time()
	return tf-t

def time_build(values):
	values = values.tolist()
	t = process_time()
	heapify(values)
	[heappop(values) for _ in range(len(values))]
	tf = process_time()
	return tf-t

def time_random_of_exp_size(sizes):
	print('time_random_of_exp_size')
	arrays = (np.random.randint(0,int(1000*s),int(s)) for s in sizes)
	return np.array([(time_build(v), time_compiled_build(v)) for v in arrays])

def plot_times(maxexp=7.3, trials=10):
	sizes = 10**np.linspace(1,maxexp,15)
	times = np.average([time_random_of_exp_size(sizes) for _ in range(trials)], axis=0)
	print('times (heapq, Heap):')
	print(times)
	ax = plt.subplot()
	for ar,l,m in zip(times.T, ('heapq (native module)','Heap (custom numba implementation)'),('o','^')):
		ax.plot(np.log10(sizes),np.log10(ar),label=l, marker=m, ms=12)
	ax.plot([],[],c='g', marker=(6,2,0), ms=12, label = 'speedup')
	ax.set_xlabel(r'${log}_{10}(\rm{input\ array\ size})$',size=16)
	ax.set_ylabel(r'${log}_{10}({time})$',size=16)
	ax.legend(fontsize=16)
	
	ax2 = ax.twinx()
	ax2.plot(np.log10(sizes), np.log10(times[:,0]/times[:,1]), c='g', marker=(6,2,0), ms=12)
	ax2.set_ylabel(r'${log}_{10}({speedup factor})$', size = 16)
	
	print('speedup:',times[:,0]/times[:,1])
	
	plt.show()

if __name__ == '__main__':
	plot_times(7, 3)