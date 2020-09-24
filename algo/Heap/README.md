# Heap

Heap routines implemented in numba-compiled Python. Two functional modules at the moment:
- `Heap`: numba-/numpy-driven implementation of a heap data structure: a left-complete, binary tree where every parent is <= its children. The factory provides generic heap functionality for heaps of primitive types (ints, floats, chars/strings). For large arrays this runs faster than Python's native heapq module-- see `Heap_timing.py` (custom Heap becomes quicker somewhere between array sizes of 10^5 and 10^6).
- `ArrayHeap` allows for merging sorted arrays via a modified implementation of the routines in `Heap.py`. Not yet timed. Currently tested on arrays of int and float types.
