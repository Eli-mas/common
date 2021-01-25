import re, operator
import warnings
from functools import partial
from itertools import product, permutations
# from inspect import getfullargspec

from ...arrays.darray.darray import DeforestedArray, bin_op_to_np
from ...collections.string  import finditer_whole_words, whole_word_expr
from ...collections import windowed_chunked
from ... import consume
from ...decorators import *

import numpy as np

counter = CallCounter()
argcounter = ArgumentCounter()

# true_divide_pattern = re.compile('(?<!/)/(?!/)')

class TestFailure(Exception):
	def __init__(self, msg):
		self.msg = msg
	
	def get_message(self):
		return self.msg

@counter.count
def test_expr(expr, *, _v = False, _checktype=False, ndarray=np.ndarray):#, tdp=true_divide_pattern
# 	dtype = 'float' if sum(1 for _ in tdp.finditer(expr)) else 'int'
	verbose = _v
	if verbose: print("test_expr:", expr)
	
	unique_names = sorted(set(m.group() for m in finditer_whole_words(expr)))
	if verbose: print("\tunique_names:",unique_names)
	data_strings = tuple(f"(10**{i} * np.arange(1, 5))"
						 for i,n in enumerate(unique_names))
	for _____n, _____d in zip(unique_names, data_strings):
		exec(f"{_____n} = DeforestedArray(data={_____d})")
		if verbose: print(f'\t\t{_____n}: string = "{_____d}", result =', eval(_____n))
	
	deforested = eval(expr)
	assert isinstance(deforested, DeforestedArray), \
		"The result of deforested arithmetic is not a DeforestedArray," \
		f" but instead a {type(deforested)}"
	
	if verbose: print(f"\texpr = {expr}, deforested:", deforested)
	
	evaluated = deforested.evaluate(verbose)
	
	for _____n, _____d in zip(unique_names, data_strings):
		exec(f"{_____n} = {_____d}")
	
	raw = eval(expr)
	assert isinstance(raw, ndarray), \
		"The result of ndarray arithmetic is not an ndarray," \
		f" but instead a {type(raw)}"
	
	if _checktype:
		if raw.dtype.name.startswith('int'):
			eq_func = np.array_equal
		else:
			eq_func = partial(np.allclose, equal_nan=True)
	else:
		eq_func = partial(np.array_equal, equal_nan=True)
	try:
		assert eq_func(evaluated, raw)
	except AssertionError:
		raise TestFailure(f"'{expr}': {evaluated} != {raw}")
	return raw

@counter.count
@argcounter.count
def test_equal_expr(
	*a,
	ndarray = np.ndarray,
	**kw,
):
	results = [test_expr(expr, _checktype=True, **kw) for expr in a]
# 	print('test_equal_expr:', a, '-->', results[0])
	assert(all(isinstance(r, ndarray) for r in results))
	
	
	if results[0].dtype.name.startswith('int'):
		eq_func = np.array_equal
	else:
		eq_func = partial(np.allclose, equal_nan=True)
	
	assert(all(eq_func(r1, r2)
		   for r1, r2 in windowed_chunked(results, 2)
	))

# class DeforestedArray_TestCase(TestCase):
# 	
# 	def test_evaluate_single(): ...
# 	
# 	...

comparison_ops = {'<','<=','!=','==','>=','>'}

int_compatible_ops = tuple(o for o in bin_op_to_np if o!='/')
def int_compatible_op_product(n, *, comparison_ops = comparison_ops):
	p = product(*(int_compatible_ops for _ in range(n)))
	return filter(
		lambda ops: sum((o in comparison_ops) for o in ops)<2, p)

low_int_excluded_ops = {'/','//','**', '==', '<', '>', '<=', '>=', '!='}

low_int_only_ops = tuple(o for o in bin_op_to_np if o not in low_int_excluded_ops) # {'<<','>>','|','&','^'}
def low_int_only_op_product(n, ops = low_int_only_ops):
	return product(*(ops for _ in range(n)))

test_expr("a")

consume(test_expr(f"a {op} 1") for op in bin_op_to_np)
consume(test_expr(f"1 {op} a") for op in bin_op_to_np)
consume(test_expr(f"b {op} a") for op in bin_op_to_np)

warnings.filterwarnings('ignore',"divide by zero encountered in")

consume(
	test_expr(f"b{op1}2 {op2} a{op3}1")
	for op1,op2,op3 in int_compatible_op_product(3)
)
consume(
	test_expr(f"(b{op1}2) {op2} (a{op3}1)")
	for op1,op2,op3 in int_compatible_op_product(3)
)

consume(
	test_expr(f"a {op1} b {op2} c {op3} d {op4} e")
	for op1, op2, op3, op4 in low_int_only_op_product(4)
)

warnings.filterwarnings('default')

test_expr("a*2 + b")
test_expr("a*2 + b*2")
test_expr("2*a + 2*b")
test_expr("2 * (a + 2) * b")

test_expr("a*2 + 1%b")
test_expr("a*2 + 2%b")
test_expr("2*a + 2%b")

test_expr("1+o")
test_expr("1+(2*o)")
test_expr("1+(2*o)*4")
test_expr("1+(2*(3-o))")
test_expr("1+(2*(3-o))+1")
test_expr("1+(2*(3-o)/5)")
test_expr("1+(2*(3-o)/5)+7")
test_expr("((a+1)*(1+(b-2)))+(2*(3-a)/5)+7")
test_expr("((a+1)*(1+(b-2)))/(1+(2*(3-a)/5)+7)")
test_expr("((2/(a+1))*(1+(b-2)))/(1+(2*(3-a)/5)+7)")
test_expr("1/((2/(a+1))*(1+(b-2)))/(1+(2*(3-a)/5)+7)")
test_expr("(c<<(((2|(3>>b))<<(a+2)) + 1)*b) / (1/((2/(a+1))*(1+(b-2)))/(1+(2*(3-a)/5)+7))")
# Why would you ever have to do this? But, it works.
test_expr("((a+b-c/d*e)+(b-c/d)-(c/d*e)) % (c<<(((2|(3>>b))<<(a+2)) + 1)*b) / (1/((2/(a+1))*(1+(b-2)))/(1+(2*(3-a)/5)+7))")

test_expr("o+(o**2)+o*2")
test_expr("o+(o**2 - o)+o*2")
test_expr("o+(2+(o-1)**2 - o)+o*2")
test_expr("a + b + c + d + e")
test_expr("(a + b) + (c + d) + e")
test_expr("1 + (a + b) + (c + d) + e")
test_expr("1 + (a + b) * (c + d) + e")
test_expr("2 + (3*a + b) * (c + d) + e")

test_equal_expr("a*2 + b", "2*a + b", "a*2 + 1*b", "a*2 + b*1")
test_equal_expr("2*(a+b)", "2*a + 2*b", "a + b + a + b")
test_equal_expr("2+o", "o+2", "1 + o + 1", "1 + ((0+o)*1) + 1")

# these operators are all associate & commutative
consume(
	test_equal_expr(*(op.join(p) for p in permutations('abcd')))
	for op in '+*&|^'
)

print(
	f"# tests:\n\t'test_expr' = {counter[test_expr]}\n\t"
	f"'test_equal_expr' = {counter[test_equal_expr]}\n"
	f"# args:\n\t'test_equal_expr' = {argcounter[test_equal_expr]}"
)

# v = argcounter.pop(test_equal_expr)
# print(v)
# print(counter.pop(test_equal_expr))