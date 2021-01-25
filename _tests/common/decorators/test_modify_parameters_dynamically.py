from ....common.decorators import modify_parameters_dynamically

def test_suite_1():
	@modify_parameters_dynamically(
		a=lambda v: -v,
		b=lambda v: v*2,
		c=lambda v: v*3,
		d=lambda v: v*4,
		e=lambda v: v*5,
		f=lambda v: v*6
	)
	def f(a,b,c=2,d=3,*varargs,e=4,f=5,**varkwargs):
	# 	for arg in :
	# 		print(f'\t{arg}',eval(arg))
		return list(map(eval,'abcdef')), varargs, varkwargs
	
	r6 = (-1,2,3,4,5,6)
	expected = ([0,2,6,12,20,30], (), {})
	got = f(0,1)
	assert expected == got, f'expected {expected} , got {got}'
	expected = ([v*m for v,m in zip((0,1,-1,-2,4,5),r6)], (-3,-4,-5), {})
	got = f(0,1,-1,-2,-3,-4,-5)
	assert expected == got, f'expected {expected}, got {got}'
	expected = ([v*m for v,m in zip((0,1,-1,-2,10,20),r6)], (-3,-4,-5), {'g':100})
	got = f(0,1,-1,-2,-3,-4,-5,e=10,f=20,g=100)
	assert expected == got, f'expected {expected}, got {got}'
	
	for repeat in 'abcd':
		try:
			got = f(0,1,-1,-2,-3,-4,-5,d=None,**{repeat:None})
		except TypeError as e:
			expected_error = f"got multiple values for argument '{repeat}'"
			expected_error2 = f"got multiple values for keyword argument '{repeat}'"
			assert str(e).endswith(expected_error) or str(e).endswith(expected_error2), \
				f'expected to fail with "{expected_error}", instead failed with: "{str(e)}"'
			
		else:
			raise AssertionError(
				"f(0,1,-1,-2,-3,-4,-5,a=0) should have failed with "
				f"multiple arguments to '{repeat}' but raised no exception"
			)

test_suite_1()

def test_suite_2():
	@modify_parameters_dynamically(c=lambda s: s.upper(), d=lambda s:s*2)
	def f(a,b,c='c',d='d',e='e'):
	    names='abcde'
	    #print(dict(zip(names,map(eval,names))))
	    return list(map(eval,names))
	
	assert f(1,2) == [1,2,'C','dd','e']
	assert f(1,2,'word',d=4) == [1,2,'WORD',8,'e']
	assert f(1,b=2,c='other',e=4) == [1,2,'OTHER','dd',4]
	assert f(1,2,'arg','arg','arg') == [1,2,'ARG','argarg','arg']

test_suite_2()