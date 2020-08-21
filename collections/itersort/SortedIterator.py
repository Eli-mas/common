"""
The SortedIterator class (hosted in this script by the same name) allows for
fundamental set operations on sorted data (union, intersection, difference,
symmetric difference) without having to convert the sorted objects into sets.
It exploits the sorted structure in the data to minimize operations.

* IMPORTANT *
Although the algorithm at work here might be theoretically efficient,
this class cannot be considered production-ready because it is *very* slow.
I am still peeling apart the reasons why this is, but I suspect it may come
from two sources: (1) the deep nesting of function calls/yield statements
that compose the class' functionality; (2) the interpreted nature of Python,
which makes things all that much slower. There also could be sub-optimal
design/implementation choices on my part aside from the nesting issue, though
nothing blatant has revealed itself to me yet.

If this could be rewritten in pure Python to be faster that would be nice. 
Using Cython or Pypy might help, but I am not experienced enough with these
to say. I don't think Numba would help in this case, because this class
makes heavy use of Python-specific features and generators. A simplified
version of this class supporting comparisons between only simple types,
such as primitives and arrays of primitives, could theoretically be written
entirely in Numba, possibly requiring rewriting without yield statements
(Numba's current generator support is still early and experimental from what
I understand). Rewriting this class in a compiled language should also help,
though I am unable to do this at the moment.
"""

import operator
from ..collection import empty_deque, counter, takewhile, consume
from collections import deque
from functools import wraps, partial
from .sorted_collections import SimpleSortedList

NULL = object()
_generator = type(i for i in range(1))

def trynext(iterable):
	try: return next(iterable)
	except StopIteration: return NULL

def closest_value_in_sorted_base(target_value,iterable,op):
	"""
	Get next value in iterable > target_value OR >= target_value,
	depending on the operator provided to op.
	"""
	# takewhile stops upon finding a value (not < v), i.e. a value >= v
	t = takewhile(lambda v: not op(v,target_value), iterable)
	# iterate through the takewhile
	consume(t)
	try:
		# if we found an element, return it
		return t.last
	except AttributeError:
		# if no matching elements were found, `t` has no 'last' attribute
		return NULL

closest_value_in_sorted_ge = partial(closest_value_in_sorted_base, op = operator.ge)
closest_value_in_sorted_gt = partial(closest_value_in_sorted_base, op = operator.gt)
'''
def closest_value_in_sorted_ge(target_value,iterable):
	"""
	Get next value in iterable >= target value.
	Return NULL if iterable exhausts.
	"""
	# takewhile stops upon finding a value (not < v), i.e. a value >= v
	t = takewhile(lambda v: v<target_value, iterable)
	# iterate through the takewhile
	consume(t)
	try:
		# if we found an element, return it
		return t.last
	except AttributeError:
		# if no matching elements were found, `t` has no 'last' attribute
		return NULL

def closest_value_in_sorted_gt(target_value,iterable):
	"""
	Get next value in iterable >= target value.
	Return NULL if iterable exhausts.
	"""
	# takewhile stops upon finding a value (not < v), i.e. a value >= v
	t = takewhile(lambda v: v<=target_value, iterable)
	# iterate through the takewhile
	consume(t)
	try:
		# if we found an element, return it
		return t.last
	except AttributeError:
		# if no matching elements were found, `t` has no 'last' attribute
		return NULL
'''
def swap_delete(i,*lists):
	"""
	In lists where the order of items does not matter,
	an item can be removed by swapping it with the last item
	in the list, and then deleting the last item.
	
	This yields a performance improvement, since the other
	items in the list do not have to be shifted to accommodate
	this deletion.
	
	This function performs this sort of deletion
	on an arbitrary number of sets. 'i' is an index or
	an iterable of indices to be removed.
	"""
	
	if isinstance(i,int):
		for l in lists:
			# swap with last item
			l[i] = l[-1]
			# now delete last item
			del l[-1]
	
	else:
		for ind in sorted(i,reverse=True):
			for l in lists:
				# swap with last item
				l[ind] = l[-1]
				# now delete last item
				del l[-1]

class SortedIterator:
	"""
	** Make sure to look at the documentation for this module for performance notes
	** This class is fully functional based on some testing I have done,
	** but far too slow to be practically useful
	
	Iterator over sorted iterables that processes, at each iteration,
	the next unique minimal item across all input iterables, and the
	sets containing this item. I.e., this function repeatedly finds the
	minimal value that is available across all iterables (denoted 'v'),
	and collects all iterables whose current value matches this value
	into a container 'matching_sets' (type(matching_sets) = list).
	The iterables may contain any objects that support comparisons
	between one another, including other iterables.
	
	This class allows performing different iteration types on the provided
	iterables, which are analgous to Python's set operations:
		* 'union': get elements in all sets regardless of repetition between sets
		* 'intersection': get elements common to at least some number of sets
		* 'symmetric_difference': get elements common to at most some number of sets
		* 'difference': get elements present in some sets and not in other sets
	
	Each of these operations has accompanying it:
		* A method named 'iter_{operation}', which yields elements as an iterator
		* classmethod functions for both the plain function and generator-function
		  versions, which have the same names but with a leading underscore added;
		  these accept iterables and call instance methods on them without exposing
		  the instance to the user, simply returning the result
	
	Motivation: the generator methods offer a memory advantage that Python's
	sets do not provide without extra wrapping. Moreover, when the iterables
	in question contain unhashable objects,
	sets are not an option, unless those objects can be converted to
	hashable objects and then placed in sets, but this is not always feasible.
	This class does not utilize hashing on the iterated objects
	in any way, only ordered comparisons.
	
	-- -- -- --
	One caveat: iterables can be equal, but if they are in fact the same
	objects, you will run into trouble if iterating over an object mutates it.
	This is because this class will call iter() on the same object twice
	and attempt to iterate over both generators simultaneously, yielding
	indeterminate results. In the event that iterables are not mutated
	over the course of iteration, then this concern is moot, but
	beware: this class does not check this condition, so it is up to the user
	to be cautious on this point.
	-- -- -- --
	
	The functionality of the aforementioned operations are enhanced beyond what
	pure set operations offer, while still maintaining minimal memory usage.
	In particular, each method works by passing arguments to the 'iterate'
	method, which determine how the elements of the above operations are yielded.
	Although there are different arguments that can be modified, there is
	primarily one that is meant to be modified by the user: the 'how' argument.
	At present there are four options:
		* 'unique': true set-like operation, yield each unique value only once.
		  This is the default for all set operations (union, intersection,
		  difference, symmetric difference).
		* 'pooled': yield each unique element a number of times equal to the number
		  of sets it occurs in.
		* 'most': yield each unique element a number of times equal to
		  the highest count (# of occurrences) it has in any of the given iterables.
		* 'all': yield each element the number of times it occurs
		  collectively across all iterables.
	
	You can also pass 'count' = True or 'with_sets' = True; both are False
	by default (passing both as True has the same effect as passing only
	'with_sets' = True). The behavior is:
		* If 'count' is True, then no matter what argument was passed
		  for 'how', the generator will yield each unique iterable only once.
		  However, this value v is yielded in a tuple, (v,N), where N is the number
		  of times that value would have been yielded had 'count' been False.
		* If 'with_sets' is True, each unique value v is yielded once in a tuple
		  (v,S) where S is a list of references to the original iterables
		  that contained that value: specifically, S=[s._original for s in matching_sets].
	
	The other arguments to 'iterate' are handled internally, as they comprise
	the architecture on which the set-like operations are built. For
	completeness, since they are left exposed in the public interface,
	they are specified here. There are three: {threshold, continuer, condition}.
	
	`threshold`: While iterating through iterables, a list is kept
	of the iterables (called 'sets'). A while-loop iterates through these,
	normally stopping when len(sets)==0. If `threshold` is provided, this changes:
	iteration stops when len(sets)<=threshold. If threshold is greater
	than the number of iterables provided, iteration immediately returns
	without consuming values from any of the iterables.
	
	`continuer`: in addition to the threshold argument, other conditions
	may be specified for determining whether the while-loop should continue.
	The `continuer` argument encapsulates this. If specified, it must
	return a boolean (or a result that can be intepreted as a boolean).
	Iteration continue if it returns True; if it returns False,
	the while-loop breaks, and iteration is finished. The `continuer`
	will receive only one argument, which is the 'sets' list identified
	above.
	
	`condition`: this is a callable that tells whether or not anything
	will be yielded at a current iteration; it should return True or False,
	or something whose results may be interpreted as boolean. It is called on
	three arguments identified above: the current 'v', the corresponding
	'matching_sets' list, and the full 'sets' list. If `condition` returns
	False, nothing is yielded, and the iterator moves to the next smallest value
	in each iterable. If it returns True, then the result of output(v, matching) is yielded.
	
		** Note, the 'matching_sets' list contains objects of type
		   SortedTrackingIter, an inner class to this class. SortedTrackingIter
		   maintains a reference to the original iterable on which iter() is called
		   to generate the iterator used by this class' iteration methods.
		   To get the original iterables, use the `_original` attribute on
		   a SortedTrackingIter instance.
	
	By default, condition(v, matching_sets, sets) is set to (lambda v: True),
	and continuer(v, matching, sets) is set to (lambda v, matching: v), so that
	this iterator returns the unique values in the union of sorted iterables.
	When you call iter() on this class, this default behavior is invoked,
	and the 'how' argument recieves 'unique', so that the default iteration behavior
	is a unique union of input iterables.
	
	The iterables provided may be containers, generators, or any other
	kind of iterable, and they may hold various kinds of object.
	The only requirement is that for each, iter(iterable) yields
	its elements in sorted order, implying that the iterables
	must contain objects that are completely orderable.
	
	This class operates on iterables of the provided inputs
	by calling iter() on them and storing these generator objects
	for operating on; copies of the data are not made. So, if one passes
	iterables that mutate over the course of iteration (e.g. generators),
	those objects will be left in an indeterminate state--i.e.,
	partially or completely consumed.
	
	Note: at the moment, calling next() on an iterator generated by this
	class always yields a single element. For design simplicity
	this made sense. It also made sense because allowing yielding
	of arbitary outputs could result in the resultant iterator
	generating values in a non-sorted order. Of course, if a sensible reason
	arises, such functionality may be added going forward.
	
	To do: another iterator like this one, but for iterables that
	are in reverse-sorted order
	"""
	
	class SortedTrackingIter:
		"""
		A minimal class that maintains references to a container and
		an iterable over that container. Used in conjunction with the
		outer 'SortedIterator' class.
		"""
		def __init__(self,iterable,_yield = None):
			self._original = iterable
			self._iter = iter(iterable)
			self._yield = iterable if _yield is None else _yield
			
			# optimization notes: see next_gt
# 			self._take = takewhile(lambda v: not op(v,target_value), self._iter)
		
		def __iter__(self):
			return self
		
		def __next__(self):
			v = self._next()
			if v is NULL: raise StopIteration
			return v
		
		def _next_gt(self, value):
# 			self.current_value = closest_value_in_sorted_gt(value, self._iter)
			# optimization notes: this function creates a new takewhile instance
			# upon every time being called. Inefficient
			while True:
				try:
					v = next(self._iter)
					if v > value:
						self.current_value = v
						return v
				except StopIteration:
					# is this the right behavior?
					self.current_value = NULL
					return NULL
		
		def _next_ge(self, value):
			self.current_value = closest_value_in_sorted_ge(value, self._iter)
			return self.current_value
		
		def _next(self):
			try:
				self.current_value = next(self._iter)
				return self.current_value
			except StopIteration:
				# is this the right behavior?
				self.current_value = NULL
				# raise
			return self.current_value
		
	class SortedTrackingContainerIter:
		def __init__(self, container, _yield = None):
			if isinstance(container, SimpleSortedList):
				self._original = container
			else:
				self._original = SimpleSortedList(container, already_sorted=True)
			
			if _yield is None:
				self._yield = container
			else:
				self._yield = _yield
			
			self.current_position = 0
		
		def __next__(self):
			if self.current_position == len(self._original):
				self.current_value = NULL
				raise StopIteration
			self.current_value = self._original[self.current_position]
			self.current_position += 1
			return self.current_value
		
		def _next_gt(self, value):
			try:
				self.current_position, self.current_value = \
					self._original.finds_gt(self.current_value, low = self.current_position)
			except ValueError:
				self.current_value = NULL
			return self.current_value
		
		def _next_ge(self, value):
			if self.current_position == len(self._original):
				return NULL
			
			self.current_position, self.current_value = \
				self._original.finds_ge(self.current_value, low = self.current_position)
	
	def __init__(self,sorted_iterables, preserve=False, preserve_keep=False, _yields = None):
		"""
		Provide sorted iterables and tell whether or not to preserve them
		for later re-iteration.
		
		If preserve: keep iterables around so that they can be iterated over repeatedly
			if preserve_keep: store the original iterable objects internally
			else: keep tuple(i) for each original iterable 'i' provided
		
		_yields:
			Normally if with_sets is specified True in self.iterate,
				unique return values are returned along with the iterables that generated them
			
			Sometimes, however, we are not interested in the iterables directly,
				but want some object with which to iterable was associated,
				especially when the iterable was a generator made from another
				object
			
			If `_yields` is specified, it achieves this behavior:
				when you specify with_sets=True in self.iterate,
				instead of yielding the generating iterables,
				the corresponding objects in `_yields` will be yielded
				
				Note: `_yields` must have same number of elements as `sorted_iterables`
		"""
		self.preserve = preserve
		if _yields is not None:
			_yields = list(_yields)
			if len(_yields) != len(sorted_iterables):
				raise ValueError(
					'if passed, `_yields` must have same length as `sorted_iterables`;'
					f'passed len(_yields)={len(_yields)}, len(sorted_iterables)={len(sorted_iterables)}'
				)
		self._yields = _yields
		
		if preserve:
			if preserve_keep: self.preserved_iterables = tuple(sorted_iterables)
			else: self.preserved_iterables = tuple(map(tuple,sorted_iterables))
			self.sets=[None]*len(self.preserved_iterables)
		else:
			self.prepare(sorted_iterables)
	
	def __len__(self):
		return len(self.preserved_iterables) if self.preserve else len(self.sets)

	def prepare(self,sorted_iterables):
		"""
		Initialization; prepare the class for iteration by storing
		iter(i) for each i in iterables and generating their intial
		values. Empty iterables are removed at this point.
		"""
		# for consistency with SimpleSortedList names
		if self.preserve: tracker = self.SortedTrackingContainerIter
		else: tracker = self.SortedTrackingIter
		if self._yields is not None:
			sets = [tracker(i,_yield = y) for i,y in zip(sorted_iterables,self._yields)]
		else:
			sets = [tracker(i) for i in sorted_iterables]
		
		# track current values in each iterable
		current_values = [None]*len(sets)
		
		current_values[:] = map(trynext,sets)
		
		# get next value for each container
		# have we exhausted any iterables?
		inds = (i for i,v in enumerate(current_values) if v is NULL)
		
# 		print('prepare: nullifying sets at indices:',[i for i,v in enumerate(current_values) if v is NULL])
		
		# If so, delete exhausted ones
		swap_delete(inds,sets,current_values)
		
		self.sets, self.current_values = sets, current_values
# 		print(' '.join('+' for _ in range(16)))
# 		print('before iteration:',
# 			f'len(sets)={len(sets)}',
# # 			f'sets={sets}',
# 			f'originals={[i._original for i in sets]}',
# 			f'continuer={self.continuer(sets)}',
# 			sep='\n\t'
# 		)
# 		print(' '.join('#' for _ in range(16)))
	
	def verify_arguments(self,
		how='unique', condition=None, continuer=None, threshold=0,
		lt_bound = None, le_bound = None, ge_bound = None, gt_bound=None):	
		"""
		Inner-mechanics method to verify that arguments make sense before
		iteration proceeds.
		"""
		# method naming convention follows how argument
		try:
			iterator = getattr(self,f'_iterator_{how}')
		except AttributeError:
			raise ValueError(
				f'invalid name "{how}" passed for `how` argument; '
				"pass one of ('unique', 'most', 'pooled', 'all')"
			)
		
		# type-checking
		if not isinstance(threshold,int):
			raise TypeError(f'`threshold` must be of type int: passed {threshold}')
		if threshold < 0:
			raise ValueError(f'`threshold` must be non-negative: passed {threshold}')
		self.threshold = threshold
		
		# more type-checking
		if condition is None:
			condition = lambda v, matching_sets, sets: True
		elif not callable(condition):
			raise TypeError(f'`condition` argument must be callable: passed {condition}')
		if continuer is None:
			continuer = lambda sets: True
		elif not callable(continuer):
			raise TypeError(f'`continuer` argument must be callable: passed {continuer}')
		
		low = high = low_func = high_func = None
		if lt_bound is not None:
			high = lt_bound
			high_func = None # TO DO
		if le_bound is not None:
			high = le_bound
			high_func = None # TO DO
		if gt_bound is not None:
			low = gt_bound
			low_func = SortedTrackingIter._next_gt
		if ge_bound is not None:
			low = ge_bound
			low_func = SortedTrackingIter._next_ge
		
		if (low is not None) and (high is not None) and low > high:
			raise ValueError(f'provided upper bound ({high}) < lower bound ({low})')
		
		return (iterator, condition, continuer, threshold), (low, low_func, high, high_func)
	
	def process_low_bound(self, low, low_func):
		""""""
		# advance iterators until lower bound is escaped
		indices_to_remove_from_sets = deque()
		
		for i,s in enumerate(self.sets):
			v = low_func(s,low)
			if v is NULL: # if exhausted, mark for removal
				indices_to_remove_from_sets.append(i)
			else:
				current_values[i] == v
		
		# remove exhausted iterators from sets
		swap_delete(indices_to_remove_from_sets, sets, current_values)
	
	def iterate(self, *a, count = False, with_sets = False, **kw):
		"""
		The generator function responsible for repeteadly yielded the
		smallest value available across all iterables, and calling
		upon another generator to tell it what to yield from
		the iterables.
		
		If 'count' is True, then instead of yielding each value
		the number of times the generator would normally yield it,
		it yields once for each unique value a tuple with that value
		and the number of times that value would be yielded
		if 'count' were False
		
		if 'with_sets' is True, again each unique value is yielded
		once in the first element of a tuple, but instead of a count
		the second element is now a list of reference to the original
		iterables that contained those values. Nothing is guaranteed
		about the order in which these iterables occur in the list.
		"""
		if self.preserve:
			self.prepare(self.preserved_iterables)
		
		(iterator, condition, continuer, threshold), \
			(low, low_func, high, high_func) \
			= self.verify_arguments(*a, **kw)
		# `iterator` is the method that determines the set-like operation
		# to be performed: union, intersection, difference, sym. diff.
		
		sets, current_values = self.sets, self.current_values
		
		if low is not None:
			self.process_low_bound(self,low,low_func)
		# TO DO: implement functionality for upper bound
		# consider: let _next_ge/_next_gt or derivatives of closest_value_in_sorted_base
		# support an upper_bound argument, which automatically declares the iterator
		# exhausted upon finding a value higher than upper bound
		
		"""
		note: instead of separate iterator code blocks,
		'count' and 'with_sets' functionality might be accomplishable
		by having a single block that calls an `output` method
		on the generator
		
		the problem is, as seen below, the original generator
		<iterator(v, condition ...)> must be consumed
		
		so let's not throw a wrench into that now, these options
		should be more than enough
		"""
		if with_sets:
			# this has a bug
			# `yield_value` is always yielded without checking to see
			# if the condition was actually met
			while len(sets)>threshold and continuer(sets):
				v, matching, matching_sets = self._get_target_value_and_matching(sets, current_values)
				# every value coming out of this iterator is the same, namely 'v'
				# so we can yield 'v' and the sets that matched it
				yield_value = (v, [s._yield for s in matching_sets])
				
				# but be sure to consume the original iterable,
				# otherwise changes that the iterator make to attributes on self
				# will not occur
				consume(iterator(v, condition, matching, matching_sets, sets, current_values))
				yield yield_value
			
			return
		
		if count:
			while len(sets)>threshold and continuer(sets):
				v, matching, matching_sets = self._get_target_value_and_matching(sets, current_values)
				
				# every value coming out of this iterator is the same, namely 'v'
				# so we can yield 'v' and the number times it is generated
				yield (v, sum(1 for _ in iterator(v, condition, matching, matching_sets, sets, current_values)))
			
			return
		
		while len(sets)>threshold and continuer(sets):
			v, matching, matching_sets = self._get_target_value_and_matching(sets, current_values)
			
			# every value coming out of this iterator is the same, namely 'v'
			# so just yield v for anything that this iterator generates
# 			print_update('yielding',v)
			for _ in iterator(v, condition, matching, matching_sets, sets, current_values):
				yield v
				
# 		print(' '.join('*' for _ in range(16)),end='\n\n')
	
	def _get_target_value_and_matching(self, sets, current_values):
		"""
		Get current smallest available value across iterables
		to match on, and all iterables that match this value.
		
		Returns 'matching', a list of indices corresponding to
		the contained iterables, and 'matching_sets',
		the corresponding list of SortedTrackingIter objects.
		"""
		
# 		sets, current_values = self.sets, self.current_values
		
		# set target value to the minimal current value
		v = min(current_values)
# 		print('current values, target:',self.current_values,v)
		
		# get all sets with this current value
		matching = [i for i,_v in enumerate(current_values) if _v==v]
		matching_sets = [sets[i] for i in matching]
# 		print('\tmatching (inds, ids, ids(originals)):',
# 			matching,
# 			list(id(i) for i in matching_sets),
# 			list(id(i._original) for i in matching_sets)
# 		)
		return v, matching, matching_sets
	
	def update_via_matching(self, v, matching, sets, current_values, indices_to_remove_from_sets=None):
		"""
		When we are done with a particular value v for any iterables,
		we have to find the next greatest value those iterables have
		to offer. This method calls on SortedTrackingIter._next_gt to
		accomplish that. If any iterables are exhausted in the process,
		they are removed from self.sets.
		"""
# 		sets, current_values = self.sets, self.current_values
# 		if indices_to_remove_from_sets is None:
# 			indices_to_remove_from_sets = deque()
		indices_to_remove_from_sets = deque()
		for i in matching:
			# find next value greater than this iterable's current value
			new_v = sets[i]._next_gt(v) # closest_value_in_sorted_gt(v,sets[i]) #
			if new_v is NULL: # iterable exhausted
				indices_to_remove_from_sets.append(i)
			else: # set to current value for this iterable
				current_values[i] = new_v
		
		# remove the sets marked for deletion, as well as the corresponding
		# values in 'current_values'
		swap_delete(indices_to_remove_from_sets, sets, current_values)
	
	def _iterator_unique(self, v, condition, matching, matching_sets, sets, current_values):
		"""
		A unique iterator over values contained across all iterables.
		When a new lowest value is found, this iterator simply yields it
		once, then updates any iterables that matched it to their
		respective next available values.
		"""
		# determine whether to yield anything
		if condition(v, matching_sets, sets):
# 			print('yielding:',output(v, matching_sets))
			yield v
		
		# any matching sets should now be updated with a new value
		# in the process, some might be exhausted
		self.update_via_matching(v, matching, sets, current_values)
	
	def _iterator_pooled(self, v, condition, matching, matching_sets, sets, current_values):
		"""
		The 'pooled' iterator finds the iterables that match a given value,
		and yields that value once for each iterable where it occurs.
		It then updates those iterables to their next available values.
		
		Results equivalent to:
		>>> for _ in range(len(matching_sets)):
		>>> 	yield v
		"""
		# determine whether to yield anything
# 		print(f'in pooled iterator with v={v}, originals:',[s._original for s in sets])
		if condition(v, matching_sets, sets):
# 			print('yielding:',output(v, matching_sets))
# 			print(f'\tpooled iterator: v={v}, len(matching_sets)={len(matching_sets)}')
			for _ in matching_sets:
				yield v
		
		# any matching sets should now be updated with a new value
		# in the process, some might be exhausted
		self.update_via_matching(v, matching, sets, current_values)
	
	def _iterator_most(self, v, condition, matching, matching_sets, sets, current_values):
		"""
		The 'most' iterator finds the iterables that match a given value,
		and then yields that value the highest number of times it occurs
		in any one of the matched iterables. It does this by iterating
		over all matched iterables simultaneously, pushing each iterable
		forward one iteration at a time, yielding the target value
		so long as any iterables are still matching, and breaking
		when no more iterables match.
		
		Results equivalent to:
		>>> m = max(s.count(v) for s in matching_sets)
		>>> for _ in range(m):
		>>> 	yield v
		"""
		# determine whether to yield anything
		indices_to_remove_from_sets = deque()
		if condition(v, matching_sets, sets):
			indices_to_remove_from_matching = deque()
			
			# most iterator: try pushing each set forward by one
			# and testing for continuity of current target value or exhaustion
			# every time at least one set has a match, yield the target value
			# in this way, we yield the target value the highest number of times
			# it occurs in any single iterable
			current_values = self.current_values
# 			print(f'_iterator_most: v={v}, matching: {matching},current values:',current_values)
			while True:
				matched = False
# 				print('\twhile start: matching:',matching)
				for ind,i in enumerate(matching):
					# find next value greater than this iterable's current value
# 					print(f'\t\tindexing current_values {current_values} at i={i}')
					if current_values[i] == v:
						matched=True
						try:
							current_values[i] = next(sets[i])
# 							print(f'\t\tfound {v} in sets[{i}]; new current value is {current_values[i]}')
						except StopIteration:
# 							print(f'\t\tfound {v} in sets[{i}]; STOP')
							indices_to_remove_from_matching.append((ind,i))
# 					else:
# 						print(f'\t\tabsent: {v} in sets[{i}]')
				swap_delete((e[0] for e in indices_to_remove_from_matching),matching)
				indices_to_remove_from_sets.extend(
					e[1] for e in empty_deque(indices_to_remove_from_matching)
				)
				if not matched: break
				yield v
			
			swap_delete(indices_to_remove_from_sets,sets,current_values)
		else:
			self.update_via_matching(v, matching, sets, current_values)
	
	def _iterator_all(self, v, condition, matching, matching_sets, sets, current_values):
		"""
		The 'all' iterator yields the target value once for each time it
		occurs in each set where it occurs. Thus the total number of
		yields of this value is equal to the total number of times
		the value occurs across all iterables.
		
		Results equivalent to:
		>>> for s in matching_sets:
		>>> 	for _ in range(list(s).count(v)):
		>>> 		yield v
		"""
		if condition(v, matching_sets, sets):
			# all iterator: yield the value for every time it occurs
			# across all sets
			current_values = self.current_values
			indices_to_remove_from_sets = deque()
# 			print(f'_iterator_all: v={v}')
			for i,s in zip(matching, matching_sets):
				# if we refactor things into a class(es),
				# I can use my takewhile here
				s = sets[i]
# 				print(f'\titerating over sets[{i}]: {sets[i]._original}',end='')
				while True:
					if current_values[i]==v:
						yield v
					try:
						current_values[i] = next(s)
						if current_values[i] != v:
# 							print()
							break
					except StopIteration:
						indices_to_remove_from_sets.append(i)
# 						print(' exhausted')
						break
			swap_delete(empty_deque(indices_to_remove_from_sets),sets,current_values)
		else:
			# condition not met for this value,
			# so update iterables to new value
			self.update_via_matching(v, matching, sets, current_values)
	
	@wraps(iterate)
	def iter_union(self,*a,**kw):
		"""
		Generator over set-like union on iterables.
		See class documentation for behavior of keyword arguments.
		"""
		return self.iterate(*a,**kw)
	
	@wraps(iter_union)
	def union(self,*a,**kw):
		"""
		Set-like union on iterables; see 'iter_union' method.
		"""
		return list(self.iter_union(*a,**kw))
	
	@wraps(iter_union)
	@classmethod
	def _iter_union(cls, sorted_iterables, *a, **kw):
		"""
		Generator on set-like union on iterables;
		this is a class method.
		"""
		return cls(sorted_iterables).iter_union(*a, **kw)
	
	@wraps(union)
	@classmethod
	def _union(cls, sorted_iterables, *a, **kw):
		"""
		Set-like union on iterables;
		this is a class method.
		"""
		return cls(sorted_iterables).union(*a, **kw)
	
	@wraps(iterate)
	def iter_intersection(self, *a, threshold=2, **kw):
		"""
		Generator over set-like intersection on iterables.
		`threshold` tells the minimal number of iterables required to contain
		a value before it is yielded. Default is 2, meaning any value
		that appears in more than one iterable is yielded.
		
		If `threshold` is set to 1, this is just a union, and 'iter_union'
		is called instead.
		
		If `threshold` is 0 or negative, it works like negative indexing
		with modularity over len(self); i.e., threshold=0 is equivalent
		to threshold=len(self), so all iterables must contain elements
		to be yielded; threshold=-1 means all but one must contain values to be
		yielded; and so forth.
		
		If threshold <= -len(self), a ValueError is raised.
		"""
		if not (-len(self) < threshold <= len(self)):
			raise ValueError(
				'specify threshold such that '
				'-len(self) < threshold <= len(self);'
				f' specified threshold={threshold} but'
				f' there are {len(self)} iterables'
			)
		if threshold<=0:
			threshold+=len(self)
		if threshold==1:
			return self.iter_union(*a,**kw)
		
		kw['condition'] = lambda v, matching_sets, sets: len(matching_sets) >= threshold
		# use threshold-1 below because if len(self) < threshold,
		# it is not possible for a value to be contained in `threshold`
		# iterables
		return self.iterate(*a, threshold = threshold-1, **kw)
	
	@wraps(iter_intersection)
	def intersection(self, *a, **kw):
		"""
		Set-like intersection on sorted iterables;
		see 'iter_intersection' method.
		"""
		return list(self.iter_intersection(*a,**kw))
	
	@wraps(iter_intersection)
	@classmethod
	def _iter_intersection(cls, sorted_iterables, *a, **kw):
		"""
		Generator on set-like intersection on sorted iterables;
		this is a class method.
		"""
		return cls(sorted_iterables).iter_intersection(*a, **kw)
	
	@wraps(intersection)
	@classmethod
	def _intersection(cls, sorted_iterables, *a, **kw):
		"""
		Set-like intersection on sorted iterables;
		this is a class method.
		"""
		return cls(sorted_iterables).intersection(*a, **kw)
	
	def iter_symmetric_difference(self, *a, threshold = 1, **kw):
		"""
		Generator on set-like symmetric difference on iterables.
		The threshold tells the maximal number of sets in which
		a value can be a member before being denied. The default is 1,
		meaning that each yielded value is unique to a particular set.
		If threshold = 2, a value is yielded if it is in no more than
		two iterables; and so forth.
		
		As with intersection, 0 and negative values are allowed,
		and len(self) is added to such values.If threshold =
		len(self), this is just a union, and iter_union is called.
		If threshold is not in the bounds (-len(self), len(self)],
		a ValueError is raised.
		"""
		if not (-len(self) < threshold <= len(self)):
			raise ValueError(
				'specify threshold such that '
				'-len(self) < threshold <= len(self);'
				f' specified threshold={threshold} but'
				f' there are {len(self)} iterables'
			)
		if threshold<=0:
			threshold+=len(self)
		if threshold==len(self):
			return self.iter_union(*a,**kw)
		
		kw['condition'] = lambda v, matching_sets, sets: len(matching_sets) <= threshold
		return self.iterate(*a, **kw)
	
	@wraps(iter_symmetric_difference)
	def symmetric_difference(self, *a, **kw):
		"""
		Set-like symmetric difference on sorted iterables;
		see 'iter_symmetric_difference' method.
		"""
		return list(self.iter_symmetric_difference(*a, **kw))
	
	@wraps(iter_symmetric_difference)
	@classmethod
	def _iter_symmetric_difference(cls, sorted_iterables, *a, **kw):
		"""
		Generator on set-like symmetric difference on sorted iterables;
		this is a class method.
		"""
		return cls(sorted_iterables).iter_symmetric_difference(*a, **kw)
	
	@wraps(symmetric_difference)
	@classmethod
	def _symmetric_difference(cls, sorted_iterables, *a, **kw):
		"""
		Set-like symmetric difference on sorted iterables;
		this is a class method.
		"""
		return cls(sorted_iterables).symmetric_difference(*a, **kw)
	
	def iter_difference(self, *, n=1, indices_include = None, indices_exclude = None, **kw):
		"""
		Generator on set-like difference between sets marked for
		inclusion and exclusion; i.e. get all elements that are present
		in the inclusion iterables but not the exclusion iterables.
		
		If `indices_include` and `indices_exclude` are not provided,
		the iterables marked for inclusion are the first `n` iterables,
		and the remaining iterables are marked for exclusion.
		Default value of n is 1, meaning get all values that are
		in the first iterable but no others.
		
		If either or both are provided, they are collections specifying
		which iterables stored by this instance will be marked for inclusion or exclusion,
		with respect to the order in which they were originally provided.
		If only inclusion indices are provided, all other iterables are marked for exclusion.
		If only inclusion iterables are provided, all other iterables are marked for inclusion.
		It is permitted to pass indices for inclusion and exclusion such that
		their union does not cover every iterable referenced by this instance.
		The only requirement is that no single index is marked both for inclusion
		and exclusion; this raises a ValueError.
		
		This method, unlike others, does not accept positional arguments.
		"""
		i_none, e_none = indices_include is None, indices_exclude is None
		
# 		print(f'iter_difference: len(self)={len(self)}, n={n}, how={kw["how"]}')
		
		if n==1:
			return self.__iter_difference_single(**kw)
		elif i_none and e_none:
			return self.__iter_difference_multi(range(n),range(n,len(self)), **kw)
		elif e_none and not i_none:
			indices_exclude = set(range(len(self)))
			indices_exclude.difference_update(indices_include)
		elif i_none and not e_none:
			indices_include = set(range(len(self)))
			indices_include.difference_update(indices_exclude)
		else:
			indices_include, indices_exclude = set(indices_include), set(indices_exclude)
			if indices_include & indices_exclude:
				raise ValueError(
					f'{self.__class__.__name__}.difference: '
					'cannot provide indices to include and to exclude that overlap: '
					f'provided include={indices_include}, exclude={indices_exclude}'
				)
		
		# if every iterable marked for exclusion, nothing can be yielded
		if not indices_include:
			return (_ for _ in range(0))
		
		# if every iterable marked for inclusion, equivalent to a union
		if not indices_exclude:
			return self.iter_union(**kw)
		
		# if only a subset of iterables is selected,
		# it wastes time to search through the other iterables not incorporated
		# when their results do not affect the difference being taken
		# so use the corresponding class method to call difference on
		# only the selected sets
		if len(indices_include) + len(indices_exclude) != len(self):
			return self.__class__._iter_difference(
				(
					*(self.sets[i]._original for i in indices_include),
					*(self.sets[i]._original for i in indices_exclude)
				),
				n = len(indices_include)
			)
		
		return self.__iter_difference_multi(indices_include, indices_exclude, **kw)
	
	def __iter_difference_single(self, **kw):
		"""Handles the special case of 'iter_difference' where n=1."""
# 		print(f'__iter_difference_single: len(self)={len(self)}, how={kw["how"]}')
		# as long as the first iterable is present in sets, it will be at index 0
		reference = self.preserved_iterables[0] if self.preserve else self.sets[0]._original
		kw['continuer'] = lambda sets: sets[0]._original is reference
		
		# it should be the only iterable of matching_sets to have the value yielded
		kw['condition'] = lambda v, matching_sets, sets: \
			len(matching_sets)==1 and (matching_sets[0]._original is reference)
		
		return self.iterate(**kw)
	
	def __iter_difference_multi(self, indices_include, indices_exclude, **kw):
		"""Handles the general case of 'iter_difference'."""
		if self.preserve:
			ref = self.preserved_iterables
		else:
			ref = [i._original for i in self.sets]
		include = set(id(ref[i]) for i in indices_include)
		exclude = set(id(ref[i]) for i in indices_exclude)
		
# 		print(
# 			f'__iter_difference_multi: len(self)={len(self)}, how={kw["how"]}'
# 			f'include={[ref[i] for i in indices_include]}, '
# 			f', exclude={[ref[i] for i in indices_exclude]}'
# 		)
		
		# require the matching value to be in at least one inclusion iterable
		# and none of the exclusion iterables
		kw['condition'] = lambda v, matching_sets, sets: \
			bool(matching_sets) and not any(id(i._original) in exclude for i in matching_sets)
		
		# stop when all the inclusion iterables are exhausted
		kw['continuer'] = lambda sets: any(id(i._original) in include for i in sets)
		
		return self.iterate(**kw)
	
	@wraps(iter_difference)
	def difference(self,**kw):
		"""
		Set-like difference between sets marked for inclusion/exclusion;
		see `iter_difference` method for details.
		"""
		return list(self.iter_difference(**kw))
	
	@wraps(iter_difference)
	@classmethod
	def _iter_difference(cls, sorted_iterables, **kw):
		"""
		Generator on set-like difference on sorted iterables;
		this is a class method.
		"""
		return cls(sorted_iterables).iter_difference(**kw)
	
	@wraps(difference)
	@classmethod
	def _difference(cls, sorted_iterables, **kw):
		"""
		Set-like symmetric on sorted iterables;
		this is a class method.
		"""
		return cls(sorted_iterables).difference(**kw)


if __name__ == '__main__':
	from functools import reduce
	from ..collection import collect_dicts
	from random import randint
	from ...common import print_update
	
	class SortedIteratorVerifier:
		class MultiSetOperator:
			def __init__(self,iterables,keep=False):
				self.keep = keep
				self.iterables = iterables if keep else tuple(map(tuple,iterables))
				self.__union = None
				self.__intersection = None
# 				for i in self.iterables:
# 					print('iterable:',i,'counts:',counter(i))
				self.counts = collect_dicts(map(counter,self.iterables))
				self.occurrences = {k:len(v) for k,v in self.counts.items()}
# 				self.print_info()
			
			def print_info(self):
				print(
					f'__init__:\n\titerables = {self.iterables}'
					f'\n\tunion={tuple(SortedIterator(self.iterables).iter_union())}'
					f'\n\tcounts={self.counts}'
					f'\n\toccurrences={self.occurrences}'
				)
			
			@property
			def union(self):
				if self.__union is None:
					self.__union = reduce(set.union, self.iterables, set())
				return self.__union
			
			@property
			def intersection(self):
				if self.__intersection is None:
					self.__intersection = reduce(set.intersection, self.iterables, self.union)
				return self.__intersection
			
			def _intersection(self,threshold=2):
				return set(k for k,v in self.occurrences.items() if v >= threshold)
			
			def _symmetric_difference(self,threshold=1):
				return set(k for k,v in self.occurrences.items() if v <= threshold)
			
			def _difference(self, n=1):
				u1 = self.__class__(self.iterables[:n], keep=self.keep).union
				u2 = self.__class__(self.iterables[n:], keep=self.keep).union
				return u1 - u2
		
		def __init__(self,iterables, **kw):
			self.iterables = tuple(map(tuple,iterables))
			self.iterator = SortedIterator(self.iterables, **kw)
			self.operator = self.MultiSetOperator(iterables, keep=True)
			self.counts = self.operator.counts
			self.kw = kw
		
		@property
		def _iterator(self):
			return SortedIterator(map(iter,self.iterables),**self.kw)
		
		count_aggregators = {
			'all' : sum,
			'most' : max,
			'pooled' : lambda values: sum(1 for _ in values),
			'unique' : lambda values: 1
		}
		
		def aggregate_counts(self, how):
			return {k:self.count_aggregators[how](v) for k,v in self.counts.items()}
		
		def verify_iterator_base(self, result, how, keys=None):
			counts = counter(result) # dict: keys = unique values in result, values = respective counts in result
			iter_counts = self.aggregate_counts(how)
			if keys is not None:
				iter_counts = {k:iter_counts[k] for k in keys}
			try: assert counts == iter_counts
			except AssertionError:
				count_str = "\n".join(str(t) for t in map(tuple,sorted(counts.items())))
				iter_count_str = "\n".join(str(t) for t in map(tuple,sorted(iter_counts.items())))
				raise AssertionError(
					f'\nresult (how={how}):\n{result}'
					f'\ncounts (for result)\n{count_str}:'
					f'\niter_counts:\n{iter_count_str}'
					f''
				)
		
		def verify_union_base(self,how):
			result = tuple(self.iterator.iter_union(how=how))
			# assert self.operator.merged == set(result)
			# above is unnecessary; it will happen in self.verify_iterator_base
			self.verify_iterator_base(result, how)
			_result = tuple(self._iterator.iter_union(how=how))
			assert _result == result, (
				f'union error, how={how}:'
				f'\n\t_result (from self._iterator):\n\t{_result}'
				f'\n\tresult (from self.iterator):\n\t{result}'
			)
		
		def verify_intersection_base(self, how, threshold=2):
			result = tuple(self.iterator.iter_intersection(how=how, threshold=threshold))
			self.verify_iterator_base(
				result, how, keys=self.operator._intersection(threshold=threshold)
			)
			_result = tuple(self._iterator.iter_intersection(how=how, threshold=threshold))
			assert _result == result, (
				f'intersection error, how={how}, threshold={threshold}:'
				f'\n\t_result (from self._iterator):\n\t{_result}'
				f'\n\tresult (from self.iterator):\n\t{result}'
			)
		
		def verify_symmetric_difference_base(self, how, threshold=1):
			result = tuple(self.iterator.iter_symmetric_difference(how=how, threshold=threshold))
			self.verify_iterator_base(
				result, how, keys=self.operator._symmetric_difference(threshold=threshold)
			)
			_result = tuple(self._iterator.iter_symmetric_difference(how=how, threshold=threshold))
			assert _result == result, (
				f'symmetric difference error, how={how}, threshold={threshold}:'
				f'\n\t_result (from self._iterator):\n\t{_result}'
				f'\n\tresult (from self.iterator):\n\t{result}'
			)
		
		def verify_difference_base(self, how, n = 1):
# 			print(f'verify_difference_base: how={how}, n={n}')
			result = tuple(self.iterator.iter_difference(how=how, n=n))
			self.verify_iterator_base(
				result, how, keys=self.operator._difference(n=n)
			)
			_result = tuple(self._iterator.iter_difference(how=how, n=n))
			assert _result == result, (
				f'difference error, how={how}, n={n}:'
				f'\n\t_result (from self._iterator):\n\t{_result}'
				f'\n\tresult (from self.iterator):\n\t{result}'
			)
		
		def verify_union_unique(self):
			return self.verify_union_base('unique')
		def verify_union_pooled(self):
			return self.verify_union_base('pooled')
		def verify_union_most(self):
			return self.verify_union_base('most')
		def verify_union_all(self):
			return self.verify_union_base('all')
		
		def verify_intersection_unique(self,*a,**kw):
			return self.verify_intersection_base('unique',*a,**kw)
		def verify_intersection_pooled(self,*a,**kw):
			return self.verify_intersection_base('pooled',*a,**kw)
		def verify_intersection_most(self,*a,**kw):
			return self.verify_intersection_base('most',*a,**kw)
		def verify_intersection_all(self,*a,**kw):
			return self.verify_intersection_base('all',*a,**kw)
		
		def verify_symmetric_difference_unique(self,*a,**kw):
			return self.verify_symmetric_difference_base('unique',*a,**kw)
		def verify_symmetric_difference_pooled(self,*a,**kw):
			return self.verify_symmetric_difference_base('pooled',*a,**kw)
		def verify_symmetric_difference_most(self,*a,**kw):
			return self.verify_symmetric_difference_base('most',*a,**kw)
		def verify_symmetric_difference_all(self,*a,**kw):
			return self.verify_symmetric_difference_base('all',*a,**kw)
		
		def verify_difference_unique(self,*a,**kw):
			return self.verify_difference_base('unique',*a,**kw)
		def verify_difference_pooled(self,*a,**kw):
			return self.verify_difference_base('pooled',*a,**kw)
		def verify_difference_most(self,*a,**kw):
			return self.verify_difference_base('most',*a,**kw)
		def verify_difference_all(self,*a,**kw):
			return self.verify_difference_base('all',*a,**kw)
		
		def main(self):
			self.verify_union_unique()
			self.verify_union_pooled()
			self.verify_union_most()
			self.verify_union_all()
			for threshold in range(1, len(self.iterables)+1):
				# note: still have to check to 0 and negative values work as expected
				# and that values out of bounds raise ValueError
				self.verify_intersection_unique(threshold)
				self.verify_intersection_pooled(threshold)
				self.verify_intersection_most(threshold)
				self.verify_intersection_all(threshold)
			for threshold in range(1, len(self.iterables)+1):
				# note: still have to check to 0 and negative values work as expected
				# and that values out of bounds raise ValueError
				self.verify_symmetric_difference_unique(threshold)
				self.verify_symmetric_difference_pooled(threshold)
				self.verify_symmetric_difference_most(threshold)
				self.verify_symmetric_difference_all(threshold)
			for n in range(0, len(self.iterables)):
				# note: still have to check to 0 and negative values work as expected
				# and that values out of bounds raise ValueError
				self.verify_difference_unique(n)
				self.verify_difference_pooled(n)
				self.verify_difference_most(n)
				self.verify_difference_all(n)
	
	import numpy as np
	import itertools
	lists = ([0,0,1,2,4,4,4,4],[1,1,2,3],[2,3,3,3,4])
	ranges = (range(20), range(0,20,3), range(0,20,2))
	strings = ['abcde','aabcde','aacccccefgg']
	"""
	print('lists:',lists)
	for method in ('_union','_intersection'):
		print(f'\t{method}')
		method = getattr(SortedIterator,method)
		for how in ('all','most','pooled','unique'):#
# 			s = SortedIterator(lists)
# 			print(f'\t{how}:',s.union(how=how))
			print(f'\t\t{how}:', method(lists,how))
	
	print('ranges:',ranges)
	for method in ('_union','_intersection'):
		print(f'\t{method}')
		method = getattr(SortedIterator,method)
		for how in ('all','most','pooled','unique'):#
# 			s = SortedIterator(ranges)
# 			print(f'\t{how}:',s.union(how=how))
			print(f'\t\t{how}:', method(ranges,how))
	"""
	
	def random_counts(iterable,low=0,high=10):
		return tuple(v for v in iterable for _ in range(randint(low,high)))
	
	iterable_sets = (
		lists,
		ranges,
		[(),(),(),[],[]],
		[[]],
		['a'],
		['','a'],
		[[4]],
		[[],[4]],
		[[3],[4]],
		[[3],[3,4],[3,4],[4],[4]],
		[((3,),(4,)),((3,),(4,))],
		[(),((3,),),((4,),)],
		[(),((),),((),(3,),),((),(4,),),((),(3,),(4,),)],
		strings,
		*(tuple(map(random_counts, lists)) for _ in range(100)),
		*(tuple(map(random_counts, ranges)) for _ in range(100)),
# 		*(str(map(random_counts, strings)) for _ in range(100)),
	)
	
	if True:
		for i,s in enumerate(iterable_sets):
			print_update(f'verifying sets[{i}]')
			siv = SortedIteratorVerifier(s, preserve=True, preserve_keep=True)
			try: siv.main()
			except AssertionError:
				print(f'\n\n!\t!\t!\n\nassertion error on sets[{i}], iterables:')
				siv.operator.print_info()
				for i in s: print(f'\t{i}')
				print('re-raising:')
				raise
		print_update('all sets verified\n')
	
	s = 2*10**5
	large_lists = tuple(tuple(itertools.compress(range(s), np.random.randint(0,2,size=s))) for _ in range(4))
	print(f'len(large_lists)={len(large_lists)}, sizes:',tuple(map(len,large_lists)))
	
	#import time
	#t0 = time.process_time()
	#u1 = SortedIterator._union(large_lists)
	#tf = time.process_time()
	#print_update('')
	#print('t (SortedIterator):',tf-t0)
	#t0 = time.process_time()
	#u2 = reduce(set.union, map(set,large_lists), set())
	#tf = time.process_time()
	#print('t (reduce::set.union):',tf-t0)
	#assert set(u1)==u2
	
	from line_profiler import LineProfiler
	l = LineProfiler(SortedIterator.union, SortedIterator.iterate, SortedIterator._iterator_unique, SortedIterator.update_via_matching, SortedIterator._get_target_value_and_matching, SortedIterator.SortedTrackingIter._next_gt)
	l.run('SortedIterator._union(large_lists)')
	with open('SortedIterator_large_lists_timing.file','w') as f:
		l.print_stats(f)
	
