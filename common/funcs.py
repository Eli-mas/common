"""Functions for common operations;
at the moment a small module with mostly file-related utilities."""

import os
import re
from pathlib import Path
from os.path import join

HOMEFOLDER = os.path.expanduser('~')

def asPath(s):
	"""If `s` is a Path, return it, otherwise return Path(s)."""
	return s if isinstance(s,Path) else Path(s)

def makepath(p):
	"""Ensure that a directory exists, if not create it; returns the path.
	
	See https://stackoverflow.com/questions/273192/ for guidance."""
	path = join(os.getcwd(), p)
	os.makedirs(path, exist_ok=True)
	return path

def makepath_from_file(f):
	"""Given a filename, ensure that the directory specified in its path
	exists by sending it through `makepath`."""
	return makepath(asPath(f).parent)

def get_filetype(f):
	"""Get the filetype, which is taken to be the string of characters
	following the last period character in the filename. If there is
	no period in the filename, return an empty str."""
	i = f.rfind('.')
	if i==-1: return ''
	return f[i+1:]

def change_filetype(f, out):
	"""Return a new Path object with the suffix of `f` replaced by `out`."""
	return asPath(f).with_suffix(out)

def get_filename(f,ret_ext=False):
	"""Separate a filename from its containing directory;
	if ret_ext is true, return both, otherwise return only the filename."""
	f = asPath(f)
	if ret_ext: return f.name,f.parent
	return f.name

def filename_insert(f,insert):
	"""Given a Path, insert `insert` after the filename before the suffix."""
	f = asPath(f)
	return str(f.with_name(f'{f.name}{insert}').with_sufix(f.suffix))

def with_filename(f, name):
	f = asPath(f)
	p = f.parent / name
	if f.suffix=='': return str(p.with_sufix(f.suffix))
	return str(p)

def print_update(*p):
	"""Printing this to a terminal erases any text on the current line
	and then prints whatever arguments are passed, passing <end=''> to
	the print function."""
	print('\x1b[2K\r',flush=True,end='')
	print(*p,flush=True,end='')

def getattrs(obj,*attrs):
	"""return [getattr(obj, a) for a in attrs]"""
	return [getattr(obj, a) for a in attrs]

def filternone(obj, replacement):
	"""Return replacement if obj is None else obj"""
	return replacement if obj is None else obj

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
		* If `attributes` is suppled, the default behavior is to form a
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

__all__ = ['asPath','makepath','makepath_from_file','get_filetype',
'change_filetype','print_update','HOMEFOLDER', 'getattrs','filternone',
'with_filename', 'reparse']