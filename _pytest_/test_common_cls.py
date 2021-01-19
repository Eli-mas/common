from common import *

def test_MultiIterator():
	m = MultiIterator()
	assert list(m)==[]
	assert tuple(m)==()
	assert len(m)==0 # empty
	
	m = MultiIterator((),[])
	assert m.targets[0]==()
	assert m.targets[1]==[]
	assert len(m.targets)==2
	assert len(m)==0 # targets are empty
	assert list(m)==[]
	assert tuple(m)==()
	
	m = MultiIterator(range(2),range(2,4))
	assert m.targets[0]==range(2)
	assert m.targets[1]==range(2,4)
	assert len(m.targets)==2
	assert len(m)==4 # total number of elements across all targets
	assert list(m)==[0,1,2,3]
	assert tuple(m)==(0,1,2,3)
	
	m = MultiIterator([0,1],(2,3),[4,5])
	assert m.targets[0]==[0,1]
	assert m.targets[1]==(2,3)
	assert m.targets[2]==[4,5]
	assert len(m.targets)==3
	assert len(m)==6
	assert tuple(m) == (0,1,2,3,4,5)
	assert list(m) == [0,1,2,3,4,5]

def test_EmptyDictType():
	e = EmptyDict
	
	assert {**e}==dict()
	assert list(e.keys())==[]
	assert list(e.values())==[]
	assert list(e.items())==[]

def test_Struct():
	s = Struct(a=4, b=2, c=1)
	assert s.a==4
	assert s.b==2
	assert s.c==1
	assert s['a']==4
	assert s['b']==2
	assert s['c']==1
	assert set(s.keys()) == {'a','b','c'}
	assert set(s.values()) == {4,2,1}
	
	s.add({'d':0, 'e':3}, {'f':5,'g':7}, word='this', other='that')
	assert s.a==4
	assert s.b==2
	assert s.c==1
	assert s['a']==4
	assert s['b']==2
	assert s['c']==1
	assert s.d==0
	assert s.e==3
	assert s.f==5
	assert s.g==7
	assert s.word=='this'
	assert s.other=='that'
	assert s['d']==0
	assert s['e']==3
	assert s['f']==5
	assert s['g']==7
	assert s['word']=='this'
	assert s['other']=='that'
	assert set(s.keys()) == {'a','b','c','d','e','f','g','word','other'}
	assert set(s.values()) == {4,2,1,0,3,5,7,'this','that'}
	
	s['new'] = 'more'
	assert s.a==4
	assert s.b==2
	assert s.c==1
	assert s['a']==4
	assert s['b']==2
	assert s['c']==1
	assert s.d==0
	assert s.e==3
	assert s.f==5
	assert s.g==7
	assert s.word=='this'
	assert s.other=='that'
	assert s['d']==0
	assert s['e']==3
	assert s['f']==5
	assert s['g']==7
	assert s['word']=='this'
	assert s['other']=='that'
	assert set(s.keys()) == {'a','b','c','d','e','f','g','word','other','new'}
	assert set(s.values()) == {4,2,1,0,3,5,7,'this','that','more'}

def test_IterDict():
	d = IterDict(a=(1,2,3),b=(10,20,30),c=(100,200,300))
	l = tuple(d)
	assert l[0]==dict(a=1,b=10,c=100)
	assert l[1]==dict(a=2,b=20,c=200)
	assert l[2]==dict(a=3,b=30,c=300)
	
	d = IterDict(a=(1,2,3),b=(10,20,30),c=(100,200))
	l = tuple(d)
	assert l[0]==dict(a=1,b=10,c=100)
	assert l[1]==dict(a=2,b=20,c=200)
	assert l[2]==dict(a=3,b=30)
	
	d = IterDict(a=(1,2,3),b=(10,20),c=(100,200))
	l = tuple(d)
	assert l[0]==dict(a=1,b=10,c=100)
	assert l[1]==dict(a=2,b=20,c=200)
	assert l[2]==dict(a=3)
	
	d = IterDict(a=(1,2,3),b=(10,20),c=(100,))
	l = tuple(d)
	assert l[0]==dict(a=1,b=10,c=100)
	assert l[1]==dict(a=2,b=20)
	assert l[2]==dict(a=3)
	
	d = IterDict(a=(1,2,3))
	l = tuple(d)
	assert l[0]==dict(a=1)
	assert l[1]==dict(a=2)
	assert l[2]==dict(a=3)

	d = IterDict()
	l = tuple(d)
	assert l==()

def test_Proxy():
	targets = ([],[],[])
	p = Proxy(targets)
	p.append(4)
	assert(list(p)==[[4],[4],[4]])
	p.pop()
	assert(list(p)==[[],[],[]])
	
	import numpy as np
	targets = (np.arange(i) for i in range(4))
	p = Proxy(targets)
	assert len(p)==4
	assert tuple(p.size)==(0,1,2,3)
	assert tuple(p.ndim)==(1,1,1,1)
	
	targets = (np.arange(i).reshape(i,*([1]*(i-1))) for i in range(1,4))
	p = Proxy(targets)
	assert len(p)==3
	assert tuple(p.size)==(1,2,3)
	assert tuple(p.ndim)==(1,2,3)
	assert tuple(p.sum())==(0,1,3)
	assert tuple(p.prod())==(0,0,0)

def test_ArgumentDeferrer():
	l=[]
	a = ArgumentDeferrer()
	a.append(4)
	a.extend([3,2,1])
	a.execute(l)
	assert l==[4,3,2,1]
	
	a.execute(l)
	assert l==[4,3,2,1,4,3,2,1]
	
	a.clear()
	a.execute(l)
	assert l==[4,3,2,1,4,3,2,1]
	
	# ArgumentDeferrer currently does not support multiple calls
	# to the same method -- only one at a time!
	# This should be considered a bug!
# 	consume(a.pop() for _ in range(4))
# 	assert l==[4,3,2,1,4,3,2,1]
# 	a.execute(l)
# 	assert l==[4,3,2,1]

