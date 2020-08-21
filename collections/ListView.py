from itertools import islice
from .collection import consume

def slice_size(s,size):
	s = slice(*s.indices(size))
	return s, slice_indices_size(s.start, s.stop, s.step, size)

def slice_indices_size(start, stop, step, size):
	if start is None: start = 0
	if stop is None: stop = size
	if step is None: step = 1
	
	return max((stop-start+step-1)//step, 0)

def make_view_slice(s, view):
	start, stop, step = s.start, s.stop, s.step
	if step==0:
		raise ValueError('cannot have slice step == 0')
	if step is None:
		step = 1
	if step==1:
		if stop is None: stop = len(view)
		elif stop < 0:
			raise ValueError('make_view_slice: negative indexing not supported')
		else: stop = min(len(view), stop)
		if start is None: start = 0
		elif start < 0:
			raise ValueError('make_view_slice: negative indexing not supported')
		else: start = min(len(view), start)
	
# 	elif view.step==-1:
# 		#if s.start is None: s.start = len()
# 		#if s.stop is None: s.stop = len(view)
# 		if s.step is None: s.step = -1
# 	...
	
	return slice(start+ view.low, stop + view.low, step), max((stop-start+step-1)//step, 0)

class ViewedList(list):
	def __init__(self, a):
		super().__init__(a)
# 		self. # FINISH
	
	def identify_overlapping_and_later_views(self, low, high):
		"""
		given low, high, get the views that overlap with
		and occur after this index range
		"""
		...
	
	def add_view(self, view):
		...
	
	def pop(self, index):
		super().pop()
		...

class ListView:
	"""
	To do:
		ListView only supports views without gaps, i.e. views formed from step=1
		what about other values of step?
		
		This class decidedly does not support negative indexing,
			because Python's negative indexing has an inconsistency
			such that [l[i] for i in range(start, stop, step)] does not always
			yield the same results as l[start:stop:step] when step is negative
			or when start and stop have different signs.
			
			And I am okay with this, because negative indexing is a Python
			convenience, not an essential language feature, nor one that
			generalizes to all languages.
		
		If the source object is modified externally, a given ListView instance
			is not aware of this;
			
			So we have to make a special class for the source object
			that maintains a list of views that are present on it,
			which updates these views when it is updated
		
		__setitem__: size_replacing not defined on iterables that do not support __len__ method
	
	DONE:
		__getitem__ should return a ListView instance, not a list slice
	
	"""
	step = 1
	
	def __init__(self, source, low, high):
		if low < 0:
			raise IndexError('ListView.__init__: negative indexing not permitted')
		if high > len(source):
			raise IndexError(f'ListView.__init__: index {high} beyond bounds of source with len {len(source)}')
		if low > high:
			raise IndexError(f'ListView.__init__: high {high} < low {low}')
		self.low = low
		self.high = high
		self.source = source
		self._compute_size()
	
	def _compute_size(self):
		"""Set the number of elements covered by this view in the source."""
		self._size = slice_indices_size(self.low, self.high, self.step, len(self.source))
	
	def __len__(self): return self._size
	
	def __iter__(self):
		return self._islice()
	
	def __insert_slice(self, where, value):
		"""start = where.start
		if start is None: start = 0
		elif start > len(self):
			start = self.high # mimics behavior of Python list
		
		stop = where.stop
		if stop is None:
			stop = 0
		
		where = slice(start + self.low, stop + self.low, where.step)
		size_replaced, where = slice_size(where,len(self.source))
		"""
		where, size_replaced = make_view_slice(where, self)
		
		size_replacing = len(value) # doesn't work if type(value) does not define __len__
		
		self.source[where] = value
		
		# modify these attributes only after
		# assignment succeeds
		dif = (size_replacing - size_replaced)
		self.high += dif
		self._size += dif
	
	def __setitem__(self, where, value):
		"""Set items in source and update bounds of this view"""
		if isinstance(where, slice):
			self.__insert_slice(where, value)
		elif isinstance(where,int):
			if where >= len(self):
				raise IndexError(f'ListView.__setitem__: index {where} beyond bounds of {self}')
			elif where < 0:
				raise IndexError(f'ListView.__setitem__: negative indexing not permitted')
			
			self.source[where + self.low] = value
		
		else:
			raise TypeError(
				f'indexing object {where} of unrecognized type {type(where)}: '
				'pass slice or int'
			)
	
	def __bool__(self):
		return bool(self.high - self.low)
	
	def __getitem__(self, where):
		"""
		Retrieve elements from source
		"""
		
		if isinstance(where, slice):
			where, _ = make_view_slice(where, self)
			# make_view_slice verifies and updates bounds accordingly
			return self.__class__(self.source, where.start, where.stop)
		
		elif isinstance(where, int):
			# require 0 <= where < len(self), as with list type
			if where >= len(self):
				raise IndexError(f'ListView.__getitem__: index {where} beyond bounds of {self}')
			elif where < 0:
				raise IndexError(f'ListView.__getitem__: negative indexing not permitted')
			return self.source[where + self.low]
		
		raise TypeError(f'indexing object {where} of unrecognized type {type(where)}: pass slice or int')
	
	def get(self, where):
		return list(self[where])
	
	def __delitem__(self, where):
		"""
		Delete from the source, and update the bounds of this view
		"""
		if isinstance(where, slice):
			# make_view_slice verifies and updates bounds accordingly
			where, size_removed = make_view_slice(where, self)
			
			del self.source[where]
			self.high -= size_removed
			self._size -= size_removed
		
		elif isinstance(where,int):
			# pop handles bound checking, don't repeat here
			self.pop(where)
		
		else:
			raise TypeError(f'indexing object {where} of unrecognized type {type(where)}: pass slice or int')
	
	def insert(self, index, iterable):
		# require 0 <= where < len(self), as with list type
		if index > self.high - self.low:
			raise IndexError(f'ListView.pop: index {index} beyond bounds of {self}')
		elif index < 0:
			raise IndexError(f'ListView.pop: negative indexing not permitted')
		
		self.source.insert(index + self.low, iterable) # automatically handles type-checking
		# modify these attributes only after insertion succeeds
		# though insertion ought always to be successful
		self.high += 1
		self._size += 1
	
	def pop(self, index):
		"""
		Given an index relative to the bounds of this view,
		pop an item from the list.
		"""
		# require 0 <= where < len(self), as with list type
		if index >= self.high - self.low:
			raise IndexError(f'ListView.pop: index {index} beyond bounds of {self}')
		elif index < 0:
			raise IndexError(f'ListView.pop: negative indexing not permitted')
		
		self.source.pop(index + self.low) # automatically handles type-checking
		# modify these attributes only after pop succeeds
		self.high -= 1
		self._size -= 1
	
	def index(self, value):
		"""
		Return index of a value relative to the bounds of this view.
		"""
		# list.index offers low/high bound options, very convenient here
		return self.source.index(value, self.low, self.high) - self.low
	
	def remove(self, value):
		"""
		Remove a value if present in this view.
		"""
		# list.remove does not provide bound options as list.index does,
		# so we take the indirect route, and as expected
		# ValueError is still raised if the value is missing
		i = self.index(value)
		
		self.source.pop(i + self.low)
		# modify these attributes only after removal succeeds
		self.high -= 1
		self._size -= 1
	
	def append(self,value):
		"""
		Append to this view by inserting `value`
		at index <self.high> in the source collection
		"""
		self.source.insert(self.high, value)
		# modify these attributes only after insertion succeeds
		# though insertion ought always to be successful
		self.high += 1
		self._size += 1
	
	def extend(self, iterable):
		"""
		Extend this view by inserting the contents of `iterable`
		into the source collection starting at the index <self.high>
		"""
		# defer to __setitem__;
		# a more efficient implementation may be possible,
		# but this deferral ensures that this
		# view's size and high bound is updated
		# and gets the indexing right
		self[len(self):] = iterable
	
	def count(self, value):
		"""count number of occurences of value in this view"""
		return sum(1 for v in self._islice() if v==value)
		# return sum(filter(v==value for v in self._islice())) # more efficient?
	
	def __repr__(self):
		return f'ListView<({self.low}, {self.high}, {self.step}): len = {len(self)}>'
	
	def __str__(self):
		return repr(self) # f'ListView<[", ".join(map(str,self._islice()))]>'
	
	def _islice(self):
		"""return islice(self.source, self.low, self.high, self.step)"""
		return islice(self.source, self.low, self.high, self.step)
	
	def _range(self):
		"""return range(self.low, self.high, self.step)"""
		return range(self.low, self.high, self.step)
	
	def _slice(self):
		return slice(self.low, self.high, self.step)
	
	def _swap(self, a, b):
		"""
		swap values on opposite sides of the view's center
		"""
		print(f'swapping inds {a}, {b}')
		self.source[a], self.source[b] = \
			self.source[b], self.source[a]
	
	def reverse(self):
		"""
		reverse view in place, i.e.
		reverse the window of the source covered by the view
		"""
		# don't have to defer to __setitem__ since size does not change
		l = len(self)
		
		# swap items on opposite sides of center until view is reversed
		if l % 2:
			consume(
				self._swap(a,b) for a,b in zip(
					range(self.low + l//2 - 1, -1, -1),
					range(self.low + l//2 + 1, self.high)
				)
			)
		else:
			consume(
				self._swap(a,b) for a,b in zip(
					range(self.low + l//2 - 1, -1, -1),
					range(self.low + l//2, self.high)
				)
			)
	
	def copy(self):
		"""Return another ListView with the same <source, low, high>."""
		return ListView(self.source, self.low, self.high)
	
	def clear(self):
		"""
		Clear all items from this view, i.e. delete the window bounded
		by this view from the source.
		"""
		del self.source[self.low:self.high:self.step]
		# view has nothing left in it
		self._size = 0
		self.high = self.low
	
	def __contains__(self, value):
		return (v in self._islice())
		# return bool(self.count(value)) # more efficient?

if __name__ == '__main__':
	def make_lv():
		l = list(range(10)); v = ListView(l, 2, 7)
		print('v:',v,'<-->',list(v))
		return l,v
	
	def test_lv(command, get=False):
		l,v = make_lv()
		c = l[:]
		cv = list(v)
		try:
			if get:
				result = eval(command)
				if isinstance(result, ListView):
					print(f'result: {command} = {result} <--> {list(result)}\n-- -- -- --\n')
				else:
					print(f'result: {command} = {result}\n-- -- -- --\n')
				assert c==l
				assert list(v) == cv
			else:
				exec(command)
				print(f'{command}: {v} <--> {list(v)}, l: {l}\n-- -- -- --\n')
		except (IndexError, ValueError) as e:
			print(f'{command}: --> {e.__class__.__name__}')
# 			print(f'v: {v} <--> {list(v)}, l: {l}\n-- -- -- --\n')
			assert c==l
			assert list(v) == cv
	
	r = range(100,500,100)
	print(f'r: {r}\n*  *  *\n')
	
	test_lv('v.pop(-1)')
	test_lv('v.pop(0)')
	test_lv('v.pop(1)')
	test_lv('v.pop(2)')
	test_lv('v.pop(3)')
	test_lv('v.pop(4)')
	test_lv('v.pop(5)')
	test_lv('v.pop(6)')
	test_lv('del v[0]')
	test_lv('del v[1]')
	test_lv('del v[2]')
	test_lv('del v[3]')
	test_lv('del v[4]')
	test_lv('del v[5]')
	test_lv('del v[6]')
	test_lv('del v[:2]')
	test_lv('del v[2:]')
	test_lv('del v[1:4]')
	test_lv('del v[1:5]')
	test_lv('del v[1:6]')
	test_lv('del v[1:10]')
	test_lv('del v[6:]')
	test_lv('del v[:]')
	test_lv('v.clear()')
	
	test_lv('v.remove(0)')
	test_lv('v.remove(1)')
	test_lv('v.remove(4)')
	test_lv('v.remove(8)')
	test_lv('v.remove(9)')
	test_lv('v.insert(-1,1000)')
	test_lv('v.insert(0,1000)')
	test_lv('v.insert(1,1000)')
	test_lv('v.insert(2,1000)')
	test_lv('v.insert(3,1000)')
	test_lv('v.insert(4,1000)')
	test_lv('v.insert(5,1000)')
	test_lv('v.insert(6,1000)')
	test_lv('v.append(1000)')
	test_lv('v.extend(r)')
	test_lv('v[7:] = r')
	test_lv('v[5:] = r')
	
	test_lv('v[4] = 1000')
	test_lv('v[5] = 1000')
	test_lv('v[0] = 1000')
	test_lv('v[-1] = 1000')
	test_lv('v.reverse()')
	
	test_lv('v[4]',True)
	test_lv('v[5]',True)
	test_lv('v[-1]',True)
	test_lv('v[4:]',True)
	test_lv('v[1:3]',True)
	test_lv('v[2:10]',True)
	test_lv('v[10:]',True)
	test_lv('bool(v)',True)
	test_lv('v.count(1)',True)
	test_lv('v.copy()',True)
	test_lv('v.index(1)',True)
	test_lv('v.index(2)',True)
	test_lv('v.index(3)',True)
	test_lv('v.index(4)',True)
	test_lv('v.index(5)',True)
	test_lv('v.index(6)',True)
	test_lv('v.index(7)',True)