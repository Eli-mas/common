# common

Collection of miscellaneous routines that have proven useful in my [research project](https://github.com/Eli-mas/ResearchProject-RamPressure-EM-JK) or other work of mine.

Notable elements:
<<<<<<< HEAD
* `algo.Heap`: numba/numpy implementation of heap routines. `Heap.py` provides generic heap functionality for heaps of primitive types (ints, floats, strings), and `ArrayHeap.py` allows for merging sorted arrays. Timings generated from `Heap_timing.py` show that `Heap` runs faster than Python's native `heapq` module for large arrays (custom Heap becomes quicker somewhere between array sizes of 10^5 and 10^6).
* `algo.Heap`: numba/numpy implementation of heap routines. `Heap.py` provides generic heap functionality for heaps of primitive types (ints, floats, strings), and `ArrayHeap.py` allows for merging sorted arrays. Timings generated from `Heap_timing.py` show that `Heap` runs faster than Python's native `heapq` module for large arrays (custom Heap becomes quicker somewhere between array sizes of 10^5 and 10^6).
* `arrays.roll_wrap_funcs`: take rolling sums/means of arrays of any dimensionality along any dimension, with the ability to wrap around this array along this dimension (functionality may be added for obtaining rolling aggregations along multiple axes simultaneously), and allowing for applying weights while rolling.
* `collections`: provides assorted simple but useful utilities for working with collections of objects. Also incorporates routines dealing with sorting. One script of interest is `collections.SortedIterator`, which handles logic for performing set-like operations on sorted data. This module is currently not usable in production code because, although the algorithms are theoretically efficient, the practical implementation is too slow; this may have to do with Python mechanics--see the documentation of that module. One thing I would like to do is rewrite the module via numba, as I did with algo.Heap -- this should make it quite fast.
* decorators: decorators of interest. Two scripts at the moment:
    - `parameter`: provides decorators for acting dynamically on a function's parameters when the function is called. In particular, this can be used for type verification, for which functionality is already provided via the `validate_parameter_types` decorator. Other functions exist for more general functionality: `operate_on_params_dynamically` allows for performing any operations on parameters when a function is called, which can be used e.g. to implement callbacks, and `modify_parameters_dynamically` can be used to modify the parameters and then pass the modified versions to the function in question. This is utilized in the `makeax` decorator, which allows for generating a matplotlib axis automatically when a function is called if one is not passed explicitly.
    - `wrap`: at the moment, provides only one decorator, but a handy one: `wrap.add_doc` allows you to automatically inject the documentation of multiple functions into the documentation of another function at runtime.
* `parallel`: at the moment contains a very simple class `threading.Queuer` for running a function in parallel across threads. 'threading' is not a good name choice for a module (collision with built-in), so at some point I will fix this.