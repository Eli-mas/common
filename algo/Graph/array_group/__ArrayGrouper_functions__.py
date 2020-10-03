import numpy as np
from numpy import inf
from numba import njit, typed, types
from numba.typed import List, Dict
from numba.types import UniTuple, ListType, DictType
from numba import int_, bool_
from numba.experimental import jitclass

def generic_init(self, array):
	self.array = array
	self.rows, self.cols = array.shape
	self.groups = np.full_like(array, -1) # to hold the label values
	
	self.counter = 0
	self.has_been_run = False
	
	# mapping <group id: 2-column array where each row is a cell coordinate>
	# note: we would rather have Set(UniTuple(int_, 2)) for the value type,
	# but at the moment numba (v0.51.2) rejects with an error:
	# "set(UniTuple(int64 x 2)) as value is forbidden"
	bordered_by = {}
	bordered_by[-1] = np.empty((2,2),int_) # gets numba to infer signature accurately
	self.bordered_by = bordered_by
	self.bordered_by.pop(-1)
	
	# mapping <group id: the value in self.array that the group refers to>
	self.group_referenced_values = Dict.empty(int_,int_)
	
	# mapping <group id: bool(does the group border the outside of the array?)>
	self.borders_outside = Dict.empty(int_,bool_)
	
	# mapping <group id: unique group_id integers of bordering groups>
	# can't specify empty Dict here, leads to compilation errors--
	# presumably a numba issue having to do with the value type being an array
	self.bordering_groups = {-1:np.empty(0,int_)}
	self.bordering_groups.pop(-1)

def _evaluate_symmetric_pattern(self, pattern):
	"""Evaluate whether a given pattern is centrosymmetric. Used internally."""
	r,c = pattern.shape
	
	if not ((r%2) or (c%2)):
		print('pattern is required to have odd dimensions: has dimensions ('
			  + str(r) + ', ' + str(c)+')')
		raise ValueError('pattern dimensions are not odd')
	
	# center coordinate of the array
	center_r, center_c = r//2, c//2
	
	# this receives row & column indices if an asymmetry is found
	error = List.empty_list(int_)
	
	# for all rows up to middle rows, scan all columns
	for i in range(center_r):
		for j in range(c):
			if pattern[i,j] != pattern[2*center_r - i, 2*center_c - j]:
				error.append(i)
				error.append(j)
				break
		else:
			# if we didn't break, go to the next iteration of the outer loop
			continue
		# if we did break, then break outer loop as well
		break
	# don't keep scanning if we already found a mismatch
	if not error:
		# for middle row, scan only up to before center column
		for j in range(center_c):
			if pattern[center_r,j] != pattern[center_r, 2*center_c - j]:
				error.append(center_r)
				error.append(j)
				break
	if error:
		i,j = error
		error_message_pieces = List([
			'error: pattern not symmetric: p[', str(i),', ',str(j),']:',
			str(pattern[i,j]),' != ','p[', str(2*center_r - i),
			', ',str(2*center_c),']:',
			str(pattern[2*center_r - i, 2*center_c - j])
		])
		print(''.join(error_message_pieces))
		raise ValueError('pattern not symmetric')

def _borders_outside(self, i, j):
	"""For a cell at coordinate (i,j), tell whether this cell
	borders the outside of the array."""
	return (i==0) or (i==self.rows-1) or (j==0) or (j==self.cols-1)

'''
def _neighbors_of_cell(self, ij, neighbor_searcher):
	"""
	Get neighbors of cell, both matching and nonmatching,
	and return together in a 2-column array, where the last
	row in the array tells the number of nonmatching and matching
	neighbors.
	
	Parameters:
	:: ij (UniTuple(-int_, 2)) :: coordinate of the cell
	"""
	i,j = ij
	value = self.array[i,j]
	rows,cols = self.rows, self.cols
	
	neighbors = []
	# for each neighbor, keep a boolean value to tell if its value matches
	match = List.empty_list(bool_)
	
	for r,c in neighbor_searcher:
		# neighbor_searcher gives relative coordinates; add to get cell coordiantes
		neighbor_r, neighbor_c = i+r, j+c
		
		# ensure coordinate points inside the array
		if (0<=neighbor_r<self.rows) and (0<=neighbor_c<self.cols):
			# matching or not, the neighbor is added
			neighbors.append((neighbor_r, neighbor_c))
			if self.array[neighbor_r, neighbor_c]==value:
				match.append(True)
			else:
				match.append(False)
	
	return neighbors, match
'''
def _find_group_and_neighbors(self, i, j):
	"""Determine and associate the group for a cell at coordinate (i,j).
	This operation mutates the instance's internal data structures."""
	
	# -1 means cell has not yet been evaluated; anything else means it has
	if self.groups[i,j] != -1:
		return
	
	value = self.array[i,j]
	
	# stack is efficiently implemented as a list, so go this route
	stack = List([(i,j)])
	
	# set of matching cells; to start, include the input cell by default
	matching = set([(i,j)])
	nonmatching = set([(-1,-1)]) # allows numba to auto-detect signature
	nonmatching.remove((-1,-1))
	
	neighbor_searcher = self._get_potential_neighbors(value)
	
	# DFS traversal to find contiguous, matching cells
	# at the end of this loop, matching contains all members of this group,
	# and nonmatching contains all the unique neighbor cells of this group
	while len(stack)!=0:
		current = stack.pop()
		neighbors, match = self._neighbors_of_cell(current, neighbor_searcher)
		for n,m in zip(neighbors,match):
			if m:
				# here we have to check explicitly, because the stack
				# only receives the cell if it is not already present
				if n not in matching:
					matching.add(n)
					stack.append(n)
			else:
				# here we don't have to check explicitly--the .add method
				# handles both cases when cell is or is not present
				nonmatching.add(n)
	
	# self.counter increments for each new group
	group_id = self.counter
	
	# set the label (group id) for all cells
	for i,j in matching:
		self.groups[i,j] = group_id
	
	# convert to arrays to match signature
	self.bordered_by[group_id] = np.array(list(nonmatching), dtype=int_)
	# 'value' is the value in the original array, 'group_id' is the label
	self.group_referenced_values[group_id] = value
	
	# find whether or not this group borders the array's outside
	for i,j in matching:
		if self._borders_outside(i,j):
			# all we need is one outer cell to yield True
			self.borders_outside[group_id] = True
			break
	else:
		# only get here if we didn't break out of the above loop
		self.borders_outside[group_id] = False
	
	self.counter += 1

def _search_with_border_check(self, group_id, value, neighbor_searcher, stack, nonmatching, borders_outside):
	"""DFS traversal to find contiguous, matching cells.
	Short-circuits in the case that one of the cells is found to
	border the array outside."""
	while (len(stack)!=0) and (not borders_outside):
		i,j = stack.pop()
		for r,c in neighbor_searcher:
			# neighbor_searcher gives relative coordinates; add to get cell coordiantes
			neighbor_r, neighbor_c = i+r, j+c
		
			# ensure coordinate points inside the array
			if (0<=neighbor_r<self.rows) and (0<=neighbor_c<self.cols):
				# matching or not, the neighbor is added
				if self.array[neighbor_r, neighbor_c]==value:
					if self.groups[neighbor_r, neighbor_c]!=group_id:
						stack.append((neighbor_r, neighbor_c))
						self.groups[neighbor_r, neighbor_c] = group_id
						if self._borders_outside(i,j):
							borders_outside = True
				else:
					nonmatching.add((neighbor_r, neighbor_c))
	return borders_outside

def _search(self, group_id, value, neighbor_searcher, stack, nonmatching):
	"""DFS traversal to find contiguous, matching cells.
	Does not check whether cells border outside because this has
	already been handled by `_search_with_border_check`."""
	while len(stack)!=0:
		i,j = stack.pop()
		for r,c in neighbor_searcher:
			# neighbor_searcher gives relative coordinates; add to get cell coordiantes
			neighbor_r, neighbor_c = i+r, j+c
			
			# ensure coordinate points inside the array
			if (0<=neighbor_r<self.rows) and (0<=neighbor_c<self.cols):
				# matching or not, the neighbor is added
				if self.array[neighbor_r, neighbor_c]==value:
					if self.groups[neighbor_r, neighbor_c]!=group_id:
						stack.append((neighbor_r, neighbor_c))
						self.groups[neighbor_r, neighbor_c] = group_id
				else:
					nonmatching.add((neighbor_r, neighbor_c))

def _find_group_and_neighbors(self, i, j):
	"""Determine and associate the group for a cell at coordinate (i,j).
	This operation mutates the instance's internal data structures."""
	
	# -1 means cell has not yet been evaluated; anything else means it has
	if self.groups[i,j] != -1:
		return
	
	value = self.array[i,j]
	
	# self.counter increments for each new group
	group_id = self.counter
	
	# stack is efficiently implemented as a list, so go this route
	stack = List([(i,j)])
	
	# set of matching cells; to start, include the input cell by default
	nonmatching = set([(-1,-1)]) # allows numba to auto-detect signature
	nonmatching.remove((-1,-1))
	
	neighbor_searcher = self._get_potential_neighbors(value)
	
	borders_outside = self._search_with_border_check(
		group_id, value, neighbor_searcher, stack, nonmatching,
		self._borders_outside(i,j)
	)
	self._search(group_id, value, neighbor_searcher, stack, nonmatching)
	# by this point, all members of this group have had the
	# group_id assigned to the respective cells in self.groups,
	# and nonmatching contains all the unique neighbor cells of this group
	
	self.groups[i,j] = group_id
	
	self.borders_outside[group_id] = borders_outside
	
	# normally we wouldn't have to do this check, because taking list()
	# of an empty set would make an empty list; but numba
	# raises an IndexError if `nonmatching` is empty
	if len(nonmatching) != 0:
		# convert to arrays to match signature
		self.bordered_by[group_id] = np.array(list(nonmatching), dtype=int_)
	else:
		self.bordered_by[group_id] = np.empty((0,0), dtype=int_)
	# 'value' is the value in the original array, 'group_id' is the label
	self.group_referenced_values[group_id] = value
	
	self.counter += 1

def label(self):
	"""Given the instance's input array, find its labeled array:
	find all contiguous groups in the array (as done by scipy.ndimage.label),
	but also generate data structures required to make a containment
	tree from the labeled array (not provided by scipy's function).
	"""
	# associate every cell with a group, caching results along the way
	# to maintain linear time complexity
	for i in range(self.rows):
		for j in range(self.cols):
			self._find_group_and_neighbors(i,j)
	
	# now we get the unique labels that border each group
	for group_id, neighbors_array in self.bordered_by.items():
		s = set([-1]) # numba type signature determination
		s.pop()
		# get unique labels that border current group
		for i,j in neighbors_array:
			s.add(self.groups[i,j])
		
		self.bordering_groups[group_id] = np.array(list(s))
	
	# mark the iteration as complete
	self.has_been_run = True

def generate_implicit_tree(self):
	"""Generate an implicit tree for the labeled array. See the class
	documentation for details of what is returned."""
	if not self.has_been_run:
		raise RuntimeError('cannot generate tree before instance has been processed')
	
	tree = {} # mapping <label: array of labels with one-greater depth>
	outer = np.array( # the outer groups comprise the top level of the tree
		[group_id for group_id, b in self.borders_outside.items() if b],
		dtype=int_
	)
	tree[-1] = outer
	
	# keep track of what not to add to the queue
	visited = Dict.empty(int_, int_)
	
	# q functions as a queue;
	# the fact that it is a queue, not a stack, is important
	q=set(outer)
	
	# top-level depth = 0 by definition
	depth = 0
	while q:
# 		print('generate_implicit_tree: while loop')
		for label in q:
			visited[label] = depth
		
		# `new` holds labels at the next depth that are discovered below
		new = set([-1]) # numba type signature detection
		new.remove(-1)
		
		# temp = List.empty_list(int_) # this doesn't work
		#     NotImplementedError: ListType[int64] cannot be represented as a Numpy dtype
		
		temp = [-1] # workaround
		temp.pop()
		# `temp` stores new labels discovered at the next depth
		# ultimately, these will be added to the queue
		
		# for each label in q, get its children labels and assign them
		# to 'new' so that they can go into q for the next while iteration
		for label in q:
			for neighboring_label in self.bordering_groups[label]:
				value = (neighboring_label in visited)
# 					if n not in visited: # numba bug? This doesn't work
				
				# we add the value to the queue in two cases:
				
				# case 1: the value has not been seen before
				if value==False:
					temp.append(neighboring_label)
				# case 2: the value has been seen and has a greater depth.
				# In this case, it will still only be evaluated once as
				# a member of q, because each level (depth) of the tree
				# is processed one at a time.
				elif visited[neighboring_label] > depth:
					temp.append(neighboring_label)
			
			# for this label, assign its children as the value in the tree
			tree[label] = np.array(temp, dtype=int_)
			# temp contains children of the current node, which are all at
			# the next depth level, so add them to 'new'
			new.update(temp)
			# clear the list so it can be reused,
			# rather than create a new list at each iteration
			temp.clear()
		# 'new' contains the labels at the next depth level, so assign to q
		q = new
		depth+=1
		
	return tree

__functions__ = {
	'generic_init': generic_init,
	'_evaluate_symmetric_pattern': _evaluate_symmetric_pattern,
	'_borders_outside': _borders_outside,
# 	'_neighbors_of_cell': _neighbors_of_cell,
	'_search_with_border_check': _search_with_border_check,
	'_search': _search,
	'_find_group_and_neighbors': _find_group_and_neighbors,
	'label': label,
	'generate_implicit_tree': generate_implicit_tree
}