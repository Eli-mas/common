"""
Iterate over segments of iterables, skipping specific indices.
"""

from .algo import consume

def segmented_iter(values, breaks, by_index, *, start = 0, n=None):
	"""
	Iterate over segments of iterables, skipping specific indices.
	
	values:     any iterable
	
	breaks:     iterable of indices in range(start, n) to be skipped;
	            can be a container or generator
	
	by_index:   if True, treat 'values' as a container and grab values
			    by standard indexing notation, i.e. values[i];
			    otherwise, treat it as a generic iterator,
			    and retrieve elements via next(values)
	
	start:      starting index of iteration; default is 0
	
	n:          the upper limit of iteration; default is None,
	            i.e. iterate through end of values
	"""
	
	if by_index:
		for b in breaks:
			for i in range(start, b):
				yield values[i]
			else:
				start = b+1
		for i in range(start,len(values)):
			yield values[i]
	else:
		values = iter(values)
		for b in breaks:
# 			print('start, b:',start,b)
			for _ in range(b-start):
				v = next(values)
# 				print('\tyielding:',v)
				yield v
			start = b+1
			next(values)
# 			try:
# 			except StopIteration:
# 				break
# 		else:
		for v in values:
			yield(v)

def iter_ranges(values, range_tuples):
	"""Iterate over specific ranges of values. This function
	operates on iter(values) regardless of the type of values.
	
	values:          any iterable
	
	range_tuples:    an iterable of tuples, where each tuple specifies
	                 (start inclusive, end exclusive). This iterable should be
	                 sorted and should not specify overlapping ranges (i.e.,
	                 the end of one tuple should not be greater than the start
	                 of the following tuple). For each tuple, ensure that
	                 start <= end. Checks are not performed on these conditions,
	                 and not adhering to them will yield indeterminate behavior.
	                 
	"""
	values = iter(values)
	current_start = 0
	for (start,end) in range_tuples:
		consume(next(values) for _ in range(start-current_start))
		for _ in range(end-start):
			yield(next(values))
		current_start = end

if __name__ == '__main__':
	from ..common import print_update
	
	def test__segmented_iter(values, breaks=None):
		test__segmented_iter.runs.append((values, breaks))
		vt = list(values)
		indices = range(len(vt))
		if breaks is None: breaks = indices
		vstr = str(values)
		if len(vstr) > 30:
			vstr = f'{vstr[:15]} ... {vstr[-15:]}'
		for r in range(len(breaks)):
			for inds in combinations(breaks,r):
				print_update(f'asserting segmented_iter on values {vstr}, break inds {inds}')
				t = list(segmented_iter(vt,inds,True))
				f = list(segmented_iter(iter(vt),inds,False))
				assert t == f == [vt[i] for i in indices if i not in inds], \
					f'values: {values}\nbreak inds: {inds}\n{t}\n{f}\n{[i for i in values if i not in inds]}'
		print_update('')
	test__segmented_iter.runs=[]
	
	import sys
	from itertools import combinations
	# edge-case test: empty iterable
	test__segmented_iter([])
	
	# edge-case test: len-1 iterable, skip value
	test__segmented_iter([0])
	# edge-case test: len-1 iterable, include value
	test__segmented_iter([0], [])
	
	# edge-case test: len-2 iterable, skip values
	test__segmented_iter([0,1])
	# edge-case test: len-2 iterable, skip single value
	test__segmented_iter([0,1], [0])
	test__segmented_iter([0,1], [1])
	# edge-case test: len-2 iterable, include values
	test__segmented_iter([0,1], [])
	
	# test with str type
	test__segmented_iter('abcd')
	
	# test with containers and more types
	test__segmented_iter(('abcd',('a',1.0,2,3+1j),[bool,len,str],(test__segmented_iter,segmented_iter)))

	# every possible arrangement for len-10 iterable
	test__segmented_iter(range(10))
	
	# following test covers a few cases:
	#     starting index == 0
	#     starting index != 0
	#     ending index == len(values)
	#     ending index != len(values)
	#     consecutive index runs (e.g. 33,34,35; 40,41)
	test__segmented_iter(range(50), (0,3,5,6,8,10,13,20,33,34,35,40,41,45,47,48,49,))
	
	print_update('test__segmented_iter: all tests finished:\n')
	for values, breaks in test__segmented_iter.runs:
		print(f'\t{values} <--> {breaks}')
	sys.exit()
