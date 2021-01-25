"""Defines a wrapper around the numpy array (in theory, any array type)
that automatically deforests chained operations.

I.e., say we have this expression:
	>>> result = a + b * (c**2 + d**2)

Where a,b,c,d are each arrays. In this form,
the expression creates four intermediate arrays (z0-z3):
	>>> result = a + b * (c**2 + d**2)
	# results in the following under the hood
	... # z0 = c**2
	... # z1 = d**2
	... # z2 = z0 + z1
	... # z3 = z2 * b
	... # result = z3 + a

Notice, had we programmed this differently, we could
have used only two intermediary arrays (z0, z1):
	>>> z0 = c**2
	>>> z1 = d**2
	>>> np.add(z0, z1, out = z1)
	>>> np.multiply(b, z1, out = z1)
	>>> np.add(a, z1, out=z1)
	>>> result = z1

But this is much more verbose than the one-liner "result = a + b * (c**2 + d**2)"

This module defines the DeforestedArray type, which allows for
keeping concise syntax while preventing unnecessary intermediate
arrays from being created. This is accomplished by lazy evaluation
and an `evaluate` method:
	result = (a + b * (c**2 + d**2)).evaluate()

TO DO:
	* `UNI_OP` implemented but not tested
	* `FUNC` case not implemented
	* Instead of `op_map` and `bin_op_to_np`, we could just have one
	  dict directly mapping the input functions to the numpy ufuncs
"""
from functools import partial
from itertools import islice

import numpy as np
from numpy.core._exceptions import UFuncTypeError

from ...collections import consume

VERBOSE = False

op_map = {
	'add':'+',
	'and':'&',
	'eq':'==',
	'floordiv':'//',
	'ge':'>=',
	'gt':'>',
	'le':'<=',
	'lshift':'<<',
	'lt':'<',
	'mod':'%',
	'mul':'*',
	'ne':'!=',
	'or':'|',
	'pow':'**',
	'rshift':'>>',
	'sub':'-',
	'truediv':'/',
	'xor':'^'
	# missing: logical {and, or}
}

op_map_r = {
	op:op_map[op]
	
	for op in ('add','and','floordiv','lshift','mod',#,'matmul','divmod'
			   'mul','or','pow','rshift','sub','truediv','xor')
}

unitary_map={
	'inv':'~',
	'invert':'~',
	'neg':'-',
# 	'not':'not ', # not used for arrays
	'pos':'+'
}
func_map={
	'abs':'abs',
	'bool':'bool'
}

bin_op_to_np = {
	'+': np.add,
	'&': np.bitwise_and,
	'==': np.equal,
	'//': np.floor_divide,
	'>=': np.greater_equal,
	'>': np.greater,
	'<=': np.less_equal,
	'<<': np.left_shift,
	'<': np.less,
	'%': np.remainder,
	'*': np.multiply,
	'!=': np.not_equal,
	'|': np.bitwise_or,
	'**': np.power,
	'>>': np.right_shift,
	'-': np.subtract,
	'/': np.true_divide,
	'^': np.bitwise_xor
}

un_op_to_np = {
	'~' : np.invert,
	'-' : np.negative,
	'+' : np.positive
}

VERBOSE_INDENT = -1


class DeforestedArray:
	"""Track operations for an expression such as (a + a**2) % 3,
	where only a single array `a` is utilized. Creates a postfix
	representation which is converted into an operation sequence."""
	
	OP_TRACK = -1 # denotes an OperationTracker instance
	VAR = 0 # denotes a variable/constant expression
	
	UNI_OP = 1 # denotes a unary operator (understood to operate on self)
	BIN_OP = 2 # denotes a (left) binary operator
	BIN_OP_R = 3 # denotes a (right) binary operator
	FUNC = 4 # denotes a function, e.g. abs or bool
	
	FUNCTYPES = {UNI_OP, BIN_OP, BIN_OP_R, FUNC}
	
	np_func_map = {
		# BINARY
		'+': np.add,
		'-': np.subtract,
		'*': np.multiply,
		'/': np.true_divide,
		'**': np.power,
		'//': np.floor_divide,
		'==': np.equal,
		'!=': np.not_equal,
		'<': np.less,
		'<=': np.less_equal,
		'>=': np.greater_equal,
		'>': np.greater,
		'<<': np.left_shift,
		'>>': np.right_shift,
		'&': np.bitwise_and,
		'|': np.bitwise_or,
		'~': np.bitwise_not,
		'^': np.bitwise_xor,
		'%': np.mod,
		
		## UNITARY
		'-': np.negative,
		'+': np.positive,
		'~': np.invert,
		
		## FUNC
		
		# bool is interesting because it is a unitary function but must be
		# converted from a binary function because it is a method on ndarray,
		# not a numpy module-level function; partial transformation ensures
		# it can be be called as a unitary function here
		'bool': partial(np.ndarray.astype, dtype=bool),
		'abs': np.abs,
	}
	
	def __init__(self, ops=None, index=1, optypes=None, data=None, remember=None):
		data = np.asarray(data)
		self.data = data
		self.refcount = 0
		self.out = None
		if ops is None:
			self.ops = [data]
			self.optypes = [self.VAR]
			self.remember = set(())
		else:
			self.ops = ops
			self.optypes = optypes
			self.remember = remember
		self.index = index
		self.__shape__ = data.shape
	
	def _get_shape(self): return self.__shape__
	def _set_shape(self, shape): self.__shape__ = shape
	def _del_shape(self): del self.__shape__
	
	
	shape = property(_get_shape, _set_shape, _del_shape,
					 "shape of resultant evaluation")
	
	'''def __update_binary_right__(self, other, symbol):
		"""Add a new right-hand side binary operation to this
		instance's operation tracker.
		
		`other` is assumed to be a constant."""
		# this could be done more efficiently, but suffices for now
		if isinstance(other, DeforestedArray):
			optype = self.OP_TRACK
		else:
			optype = self.VAR
		if self.index == len(self.ops):
			ops = [other, *self.ops, symbol]
			optypes = [optype, *self.optypes]
		else:
	'''
	def __update_binary__(self, other, symbol, direction_constant):
		"""Add a new operation to this instance's operation tracker.
		`other` is assumed to be a constant, since this class only
		tracks operations on a single array. Adds operations in order
		consistent with postfix notation.
		
		direction_constant should be one of:
			* BIN_OP : when operands are evaulated in the order in which they occur
			* BIN_OP_R : when operatnds are evaluated opposite the order of their occurence
		"""
		
		# optimization: if we are adding a new operation to the end of the list,
		# we do not have to copy the list at all; the index stored on this
		# instance tells how far to traverse the list.
		if self.index == len(self.ops):
			ops = self.ops
			optypes = self.optypes
		# only if we are adding an operation in the middle,
		# we have to copy up to that point
		else:
			ops = self.ops[:self.index]
			optypes = self.optypes[:self.index]
		
		# track the other object
		ops.append(other)
		# track the type of the other object;
		# log if it is an operation tracker (OP_TRACK) or not (VAR)
		if isinstance(other, DeforestedArray):
			optypes.append(self.OP_TRACK)
			other.refcount += 1
		else:
			optypes.append(self.VAR)
		# track the (binary) operation and note that we have added a BIN_OP
		ops.append(symbol)
		optypes.append(direction_constant)
		# return the new instance. we have add two elements (`other` and a
		# symbol denoting an operation), so the index increases by 2.
		return self.__class__(ops, index = self.index+2, optypes = optypes)
	
	def __update_unary__(self, symbol):
		# optimization: see above for explanation
		if self.index == len(self.ops):
			ops = self.ops
			optypes = self.optypes
		else:
			ops = self.ops[:self.index]
			optypes = self.optypes[:self.index]
		# track the (unary) operation and note that we have added a UNI_OP
		ops.append(symbol)
		optypes.append(self.UNI_OP)
		# return the new instance. we have only added a single element,
		# so the index increases by 1.
		return self.__class__(ops, index = self.index+1, optypes = optypes)
	
	def __update_func__(self, name):
		# optimization: see above for explanation
		if self.index==len(self.ops):
			ops = self.ops
			optypes = self.optypes 
		else:
			ops = self.ops[:self.index]
			optypes = self.optypes[:self.index]
		# track the (external) operation and note that we have added a FUNC
		ops.append(name)
		optypes.append(self.FUNC)
		# return the new instance. we have only added a single element,
		# so the index increases by 1.
		return self.__class__(ops, index = self.index+1, optypes = optypes)
	
	for name, symbol in op_map.items():
		exec(f'def __{name}__(self,other): '
				f'return self.__update_binary__(other, "{symbol}", {BIN_OP})')
	for name, symbol in op_map_r.items():
# 		print('op_map_r:', name, symbol)
		exec(f'def __r{name}__(self,other): '
				f'return self.__update_binary__(other, "{symbol}", {BIN_OP_R})')
# 		print(eval(f'__r{name}__'))
	for name, symbol in unitary_map.items():
		exec(f'def __{name}__(self,symbol): return self.__update_unary__("{symbol}")')
	for name, fname in func_map.items():
		exec(f'def __{name}__(self): return self.__update_func__("{fname}")')
	
	def __str__(self):
		return f'OpTrack<({id(self)})>[{", ".join(str(v) for v in islice(self.ops, self.index))}]'
# 		return f"OpTrack<{id(self)}>"
	
	def __repr__(self):
		return f'OpTrack<({id(self)})>[{", ".join(str(v) for v in islice(self.ops, self.index))}]'
# 		return f"OpTrack<{id(self)}>"
	
	def __hash__(self): return hash(id(self))
	
	def evaluate(self, VERBOSE=VERBOSE, *, mscalar = np.min_scalar_type, CASTING = 'safe', can_cast = np.can_cast):
		if VERBOSE:
			global VERBOSE_INDENT
			VERBOSE_INDENT += 1
			tab0, tab1 = '\t'*VERBOSE_INDENT, '\t'*(VERBOSE_INDENT+1)
		
		stack = [] # tracks elements in the postfix expression (self.ops)
		mutability = [] # tracks whether elements in `stack` are mutable
		
		OP_TRACK = self.OP_TRACK
		BIN_OP, BIN_OP_R, FUNC, UNI_OP = self.BIN_OP, self.BIN_OP_R, self.FUNC, self.UNI_OP
		FUNCTYPES = self.FUNCTYPES
		
		immutable = {e for e,t in zip(self.ops, self.optypes) if t is OP_TRACK}
		
		self_class = type(self)
		
		for elem, optype in zip(islice(self.ops, self.index), islice(self.optypes, self.index)):
			if VERBOSE: print(f"{tab0}evaluate <{optype}>:", elem)
			if optype in FUNCTYPES:
				if optype==BIN_OP or optype==BIN_OP_R:
					if VERBOSE: print(f'{tab1}binary operation')
					if optype is self.BIN_OP:
						op_right, op_left = stack.pop(), stack.pop()
						op_right_is_mutable, op_left_is_mutable = mutability.pop(), mutability.pop()
					else:
						op_left, op_right = stack.pop(), stack.pop()
						op_left_is_mutable, op_right_is_mutable = mutability.pop(), mutability.pop()
					
					if VERBOSE: print(f'{tab1}operands before class check:', op_left, op_right)
					
					if isinstance(op_left, self_class):
						if VERBOSE:
							print(f"{tab1}op_left evaluating")
							VERBOSE_INDENT += 1
						op_left = op_left.evaluate()
						if VERBOSE:
							VERBOSE_INDENT -= 1
					if isinstance(op_right, self_class):
						if VERBOSE:
							VERBOSE_INDENT += 1
							print(f"{tab1}op_right evaluating")
						op_right = op_right.evaluate()
						if VERBOSE:
							VERBOSE_INDENT -= 1
					
					if VERBOSE: print(f'{tab1}operands after class check:', op_left, op_right)
					
					try:
						mscalar_left, mscalar_right = mscalar(op_left), mscalar(op_right)
						
						if   op_left_is_mutable \
						     and \
						     can_cast(mscalar_right, mscalar_left, casting=CASTING) \
						:
							result = bin_op_to_np[elem](op_left, op_right, out=op_left, casting=CASTING)
						
						elif op_right_is_mutable \
						     and \
						     can_cast(mscalar_left, mscalar_right, casting=CASTING) \
						:
							result = bin_op_to_np[elem](op_left, op_right, out=op_right, casting=CASTING)
						else:
							result = bin_op_to_np[elem](op_left, op_right)
					except (TypeError, ValueError, UFuncTypeError):
						"""
						* TypeError can result e.g. from true_divide called on
						  two integer arrays even with casting = 'safe', the default
						* ValueError arises e.g. from invalid broadcasting
						* UFuncTypeError arises when the casting rule does not allow an
						  in-place operation, e.g. <int> += <float> for casting = 'safe'.
						"""
						result = bin_op_to_np[elem](op_left, op_right)
					
					
# 					# can't do in-place data transfer between arrays of varying dtypes
# 					except :
# 						result = bin_op_to_np[elem](op_left, op_right)
				elif optype == UNI_OP:
					operand = stack.pop()
					if isinstance(operand, self_class):
						if VERBOSE:
							VERBOSE_INDENT += 1
							print(f"{tab1}unary operand evaluating")
						operand = operand.evaluate()
						if VERBOSE:
							VERBOSE_INDENT -= 1
					if mutability.pop():
						result = un_op_to_np[elem](operand, out=operand)
					else:
						result = un_op_to_np[elem](operand)
				elif optype == FUNC:
# 					result = ... <FUNC(stack.pop().data)>
					raise NotImplementedError("Implement FUNC case")
				stack.append(result)
				mutability.append(True)
			else:
				mutability.append((optype is OP_TRACK) and (elem not in immutable))
				stack.append(elem)
		
		if VERBOSE: VERBOSE_INDENT -= 1
		
		return stack.pop()

if __name__ == '__main__':
	# simple test case
	o = DeforestedArray(data=np.arange(4))
	o1 = o+(o**2)+o*2
	evaluated = o1.evaluate()
	raw = o.data + (o.data**2) + o.data*2
	if VERBOSE:
		print(o1.optypes, o1.ops, sep='\n')
		print("evalated:", evaluated)
		print("raw:", raw)
	assert np.allclose(evaluated, raw)
	
	# extension test case
	o = DeforestedArray(data=np.arange(4))
	o1 = o+(o**2 - o)+o*2
	evaluated = o1.evaluate()
	raw = o.data + (o.data**2 - o.data) + o.data*2
	if VERBOSE:
		print(o1.optypes, o1.ops, sep='\n')
		print("evalated:", evaluated)
		print("raw:", raw)
	assert np.allclose(evaluated, raw)
	
	# test __radd__
	o = DeforestedArray(data=np.arange(4))
	o1 = 1+o
	evaluated = o1.evaluate()
	raw = 1+o.data
	if VERBOSE:
		print(o1.optypes, o1.ops, sep='\n')
		print("evalated:", evaluated)
		print("raw:", raw)
	assert np.allclose(evaluated, raw)
	
	# extension test case
	o = DeforestedArray(data=np.arange(4))
	o1 = o+(2+(o-1)**2 - o)+o*2
	evaluated = o1.evaluate()
	raw = o.data + (2+(o.data-1)**2 - o.data) + o.data*2
	if VERBOSE:
		print(o1.optypes, o1.ops, sep='\n')
		print("evalated:", evaluated)
		print("raw:", raw)
	assert np.allclose(evaluated, raw)
	
	# using two arrays
	a = DeforestedArray(data=np.arange(4))
	b = DeforestedArray(data=np.arange(4)+1)
	o1 = a + b
	evaluated = o1.evaluate()
	raw = a.data + b.data
	if VERBOSE:
		print(o1.optypes, o1.ops, sep='\n')
		print("evalated:", evaluated)
		print("raw:", raw)
	assert np.allclose(evaluated, raw)
	
	# using two arrays and another operation
	a = DeforestedArray(data=np.arange(4))
	b = DeforestedArray(data=np.arange(4)+1)
	o1 = a*2 + b
	evaluated = o1.evaluate()
	raw = a.data*2 + b.data
	if VERBOSE:
		print(o1.optypes, o1.ops, sep='\n')
		print("evalated:", evaluated)
		print("raw:", raw)
	assert np.allclose(evaluated, raw)
	
	print('tests passed')