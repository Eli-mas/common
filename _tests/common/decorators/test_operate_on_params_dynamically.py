"""To do: automate tests"""

from common.decorators import operate_on_params_dynamically

@operate_on_params_dynamically(
	a=lambda v: print(f'operating on \'a\': {v}'),
	b=lambda v: print(f'operating on \'b\': {v}'),
	c=lambda v: print(f'operating on \'c\': {v}'),
	d=lambda v: print(f'operating on \'d\': {v}'),
	e=lambda v: print(f'operating on \'e\': {v}'),
	f=lambda v: print(f'operating on \'f\': {v}')
)
def f(a,b,c=2,d=3,*varargs,e=4,f=5,**varkwargs):
	#print('- - called f - -')
	#print('\tno defaults')
	#for arg in ('a','b'):
	#	print(f'\t\t{arg}:',eval(arg))
	#print('\tpositional with defaults')
	#for arg in ('c','d'):
	#	print(f'\t\t{arg}:',eval(arg))
	#print('\tvarargs')
	#for arg in varargs:
	#	print(f'\t\t{arg}')
	#print('\tkeyword defaults')
	#for arg in ('e','f'):
	#	print(f'\t\t{arg}:',eval(arg))
	#print('\tvarkwargs')
	#for k,v in varkwargs.items():
	#	print(f'\t\t{k}:',v)
	print()

#def f2()

def test_suite_1():
	f(1,2,3,4,e=100,f=200)
	
	f(a=1,b=2,c=3,d=4,e=100,f=200)
	
	f(1,b=2,c=3,d=4,e=100,f=200)
	
	f(1,2,c=3,d=4,e=100,f=200)
	
	f(1,2,3,4,5,6,7,e=100,f=200,k=123456789)

def test_suite_2():
	f(100,200,e=100200)
	f(a=100,b=200,e=100200)
	f(100,b=200,e=100200)

def test_suite_3():
	f(100,200,d=400,c=300,f=1234)
	f(a=100,b=200,f=1234,d=400,c=300)
	f(100,b=200,c=300,f=1234,d=400)
	f(a=100,b=200,c=300,f=1234,d=400)

#test_suite_1()
#test_suite_2()
#test_suite_3()


@operate_on_params_dynamically(
	a=lambda v: print(f'operating on \'a\': {v}'),
	b=lambda v: print(f'operating on \'b\': {v}'),
	c=lambda v: print(f'operating on \'c\': {v}'),
	d=lambda v: print(f'operating on \'d\': {v}'),
	e=lambda v: print(f'operating on \'e\': {v}'),
	f=lambda v: print(f'operating on \'f\': {v}')
)
def f(a=1,b=2,c=3,*,d=4,e=5,f=6,**varkwargs): print()

def test_suite_4():
	f()
	f(1,2,c=3)
	f(a=1,b=2,c=3)
	f(d=4)
	f(f=6)
	f(e=5)

def test_suite_5():
	f(100)
	f(100,200,c=300)
	f(a=100,b=200,c=300)
	f(d=400)
	f(f=600)
	f(e=500)
	f(f=600,c=300,a=100,e=500,b=200,d=400)
	f(100,f=600,c=300,e=500,b=200,d=400)

#test_suite_4()
test_suite_5()