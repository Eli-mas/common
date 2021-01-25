import re

non_identifier_pattern = re.compile('[^a-zA-Z0-9\_]+')
non_identifer_char_pattern = re.compile('[^a-zA-Z0-9\_]')
identifier_pattern = re.compile('(?<![a-zA-Z0-9\_])[a-zA-Z\_][a-zA-Z0-9\_]*')
identifier_match = re.compile('^(?<![a-zA-Z0-9\_])[a-zA-Z\_][a-zA-Z0-9\_]*$')

whole_word_expr = '(?<![a-zA-Z0-9\_]){}(?![a-zA-Z0-9\_])'

whole_symbol_expr = '(?<![^a-zA-Z0-9\_]){}(?![^a-zA-Z0-9\_])'
	
class InvalidPatternError(ValueError):
	"""Indicates that a given string does not match an expected pattern"""
	def __init__(self, bad_pattern, msg=None):
		self.bad_pattern = bad_pattern
		self.msg = msg
	
	def __str__(self):
		if self.msg is None:
			return f'"{bad_pattern}" is not a valid pattern'
		return f'"{bad_pattern}": {self.msg}'

def assert_no_iter_wrap(ErrorClass):
	if not issubclass(ErrorClass, Exception):
		raise TypeError("`ErrorClass` should inherit from `Exception`")
	
	def assert_no_iter(it, *a):
		"""Given an iterable `it`, assert that it is empty by trying to
		remove an element and catching a stop exception.
		`ErrorClass` is the error type to be raised upon failure.
		`*a` are passed to ErrorClass()."""
		try:
			next(it)
			raise ErrorClass(*a)
		except StopIteration:
			pass
	
	return assert_no_iter

def assert_iter_wrap(ErrorClass):
	if not issubclass(ErrorClass, Exception):
		raise TypeError("`ErrorClass` should inherit from `Exception`")

	def assert_iter(it, *a):
		"""Assert that an iterable is non-empty by retrieving
		the next element.
		`ErrorClass` is the error type to be raised upon failure.
		`*a` are passed to ErrorClass()."""
		try:
			next(it)
		except StopIteration:
			raise ErrorClass(*a)
	
	return assert_iter

assert_no_iter_pattern = assert_no_iter_wrap(InvalidPatternError)

assert_iter_pattern = assert_iter_wrap(InvalidPatternError)


def finditer_whole_words(expr, *, ipf=identifier_pattern.finditer):
	"""Given any arbitrary Python expression as a string, return a
	re.finditer iterator over the unique valid Python identifiers
	located within this string."""
	return ipf(expr)

def finditer_whole_word(expr, search_string, *, wwf = whole_word_expr.format):
	"""NOT YET DEBUGGED"""
	raise NotImplementedError
	return re.compile(wwf(expr)).finditer(search_string)

def finditer_whole_symbol(symbol, search_string, wsf = whole_symbol_expr.format):
	return re.compile(wsf(re.escape(symbol))).finditer(search_string)

def finditer_identifier(expr, search_string, *, wwf = whole_word_expr.format):
	"""Given a valid Python identifier, return a re.finditer iterator
	over occurrences of the expression as a whole word (i.e., not
	embedded within another valid Python identifier)."""
	if identifier_match.match(expr) is None:
		raise InvalidPatternError(expr, "invalid Python identifier")
	
	return re.compile(wwf(expr)).finditer(search_string)

def reparse(expression, repstr, pattern=None, attributes=None):
	"""Given an `expression` (str) and a replacement string `repstr` (str),
	use a regex pattern to replace occurrences of specific attributes
	with strings indicating the getting of those attributes on the object
	represented by `repstr`. I.e., for a matched attribute 'a', replace
	occurences of 'a' with f'{repstr}.a' in `expression`.
	
	The attributes are either defined by `attributes` (iterable of str),
	or by `pattern`, which should be a re.Pattern instance that matches
	any of the desired attributes. At least one of these parameters must
	be provided. If both are provided, only `pattern` is considered.
		
	Though I could offer the option for `pattern` to be a string, I did not
	want to do this was since it might encourage one to submit a string to
	this function repeatedly, having re.compile called on it repeatedly,
	which is wasteful. A pattern should only be compiled once if it is to
	be used repeatedly.
	
	Limitations at the moment:
		* If `attributes` is supplied, the default behavior is to form a
		  regex pattern by joining the attributes with '|', so that the
		  pattern is an OR-join on the attributes. This can be improved
		  to match the attributes only when they exist as entire words,
		  not as part of other words. The '|'-joined pattern does not
		  accomplish this.
	
	This implementation is efficient because it makes a single pass on
	`expression` (due to pattern.finditer). It then uses the results of
	pattern.finditer to isolate start and end regions that need replacing,
	compiles these into a list, and then calls ''.join() to form the
	new string. The O(1)-pass implementation means O(n*L) time overall,
	where n = # of attributes and L=size of longest attribute, assuming
	that pattern.finditer operates in linear time. If pattern.finditer
	operates in super-linear time, so does this function.
	
	Examples:
	>>> expr = '(a + b) * (c + d)'
	>>> reparse(expr, 'T', re.compile('a|b|c|d'))	# '(T.a + T.b) * (T.c * T.d)'
	>>> reparse(expr, 'T', attributes = 'abcd')		# '(T.a + T.b) * (T.c * T.d)'
	>>> class T: a,b,c,d = 1,2,3,4
	>>> eval(reparse(expr, 'T', attributes = 'abcd')) # 21
	
	Limitations/extensions:
		* This is a specific kind of batch replacement but could be
		  more generalized; here we are replacing expressions with
		  attribute accesses only, there are more options
		* Could also have different replacement options for different
		  patterns, in which case a mapping of constants/callables
		  could be supplied
		* Does not check whether attributes are substrings of one another,
		  which is likely to lead to undefined behavior (if not a crash)
		  -- this is a bug
	"""
	if pattern is None:
		try: pattern = re.compile("|".join(attributes))
		except TypeError: raise ValueError(
			"`reparse`: if `pattern` is None, pass an iterable "
			"of strings for `attributes` parameter"
		)
	elif not isinstance(pattern, re.Pattern):
		raise TypeError("`reparse`: pass a re.Pattern instance for `pattern` parameter")
	regions = ((m.start(), m.end()) for m in pattern.finditer(expression))
	
	out = []
	start = 0
	for _start, _end in regions:
		out.append((False, start, _start))
		out.append((True, _start, _end))
		start = _end
	out.append((False, _end, len(expression)))
	
	return ''.join(f'{repstr}.{expression[s:e]}' if r else expression[s:e]
				   for r,s,e in out)

__all__ = ('InvalidPatternError', 'finditer_whole_words', 'finditer_whole_word', 'reparse')