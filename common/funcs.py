"""Functions for common operations;
at the moment a small module with mostly file-related utilities."""

import os
import re
from pathlib import Path
from os.path import join
from ..collections import consume

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

def tprint(*objs, s='\t'):
	"""Print an object with some space string `s` inserted at each line.
	Note: uses '\n' to find newline, this might not be fully portable.
	"""
	for obj in objs:
		if not isinstance(obj, str): obj = str(obj)
		consume(print(f'{s}{l}') for l in obj.split('\n'))

__all__ = ['asPath','makepath','makepath_from_file','get_filetype',
'change_filetype','print_update','HOMEFOLDER', 'getattrs','filternone',
'with_filename', 'tprint']