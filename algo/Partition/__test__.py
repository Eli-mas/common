import numpy as np
from common import consume
from .__main__ import partition_on_value as partition

def _test_partition(a, sep):
	i = partition(a, sep)
	assert (a[:i+1]<=sep).all(), \
		f'partition error: partition on:\n{a}\nsep={sep}, i={i}, expected {np.sum(a<=sep)}'
	assert (a[i+1:]>sep).all(), \
		f'partition error: partition on:\n{a}\nsep={sep}, i={i}, expected {np.sum(a<=sep)}'

def test_partition_random(low, high, n):
	a = np.random.randint(low, high, n)
	m,M = a.min(),a.max()
	consume(_test_partition(a.copy(),s) for s in (m, M, (m+M)//2))

trials = 1000
lows = np.random.randint(-100, 100, trials)
highs = lows + np.random.randint(1, trials//10, trials)
counts = np.random.randint(1, trials//10, trials)
consume(test_partition_random(l,h,c) for l,h,c in zip(lows, highs, counts))