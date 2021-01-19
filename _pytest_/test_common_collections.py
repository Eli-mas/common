from common.collections import *

from collections import deque

def test_consume():
	l=[]
	consume(l.append(i) for i in range(4))
	assert l==[0,1,2,3]
	
	l=[[],[],[],[]]
	consume(c.append(i) for c,i in zip(l,range(4)))
	assert l==[[0],[1],[2],[3]]
	
	l=[]
	consume(l.insert(0, word) for word in 'these are words'.split())
	assert l==['words','are','these']

def test_empty_deque():
	from common.collections.algo import empty_deque
	d = deque([1,2,3])
	assert(len(d))==3
	l = list(empty_deque(d))
	assert l==[1,2,3]
	assert len(d)==0

def test_argsort():
	from common.collections.algo import argsort
	l=[2,4,3,1]
	assert argsort(l)==[3,0,2,1]
	
	l=[1,0]
	assert argsort(l)==[1,0]

