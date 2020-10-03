# ArrayGrouper

Label an array of integers and generate a containment tree for it via the `ArrayGrouper` and `ArrayGrouperMultiPattern` classes.

Labeling an array means the same thing as in, e.g., [`scipy.ndimage.label`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html): identify all the contiguous groups of that array and assign to each group a unique label, in this case returned as another array of identical shape to the input array, but filled with the derived integer labels. This functionality is identical to that provided by scipy.ndimage.label.

The novel functionality provided here is in the containment tree. The containment tree indicates the order in which one would encounter groups in the array proceeding inwards from outside the array. A containment tree satisfies the following properties:
* Acyclic; pointers are from parents --> children only; children always have depth 1 greater than their parents.
* It contains each unique generated label only once.
* The depth at which a label is contained in this tree is the *minimal separation degree* from the outside of the array. E.g. if a group borders the outside, its minimal separation degree is 0, because to get to the outside from that group does not reqiure traversing through any other groups. If instead you are at a group and you have to go through at least one other group in order to get to the outside, then group 1 has a minimal separation of 1, and is found at depth 1 in the tree.
* Parents and children are related in a depth-wise sense: a child has a minimal separation degree exactly one greater than its parent (compatible with the fact that it has depth one greater than its parent).
* Parents may have multiple children, children may have multiple parents. Parents may also have no children, but every node beneath the topmost level has at least one parent.

For example, if we have the following array:
![sample array 1](https://github.com/Eli-mas/common/blob/master/algo/Graph/array_group/imgs/sample%20array%201.png)
The groups are computed as follows:
![sample groups 1](https://github.com/Eli-mas/common/blob/master/algo/Graph/array_group/imgs/sample%20groups%201.png)
And the containment tree is:
![sample groups 1](imgs/https://github.com/Eli-mas/common/blob/master/algo/Graph/array_group/imgs/sample%20containment%20tree%201.png)

A somewhat more complex eaxmple: an array is given as
![sample array 2](https://github.com/Eli-mas/common/blob/master/algo/Graph/array_group/imgs/sample%20array%202.png)
The groups are computed as follows:
![sample groups 2](https://github.com/Eli-mas/common/blob/master/algo/Graph/array_group/imgs/sample%20groups%202.png)
And the containment tree is:
![sample groups 2](imgs/https://github.com/Eli-mas/common/blob/master/algo/Graph/array_group/imgs/sample%20containment%20tree%202.png)

The `ArrayGrouper` class allows for specifying a single pattern that tells what cells may be considered neighbors of a given cell. The `ArrayGrouperMultiPattern` class allows for specifying different patterns for particular values, as well as a default pattern. The meaning of 'pattern' here is identical to the `structure` parameter passed to [`scipy.ndimage.label`].

At the moment the containment tree may be generated in two ways:
* **implicitly**: a flat dictionary where each key in the dict (a label) points to a conntainer of other keys (labels) in the dict, such that the pointers do not violate the aforementioned properties of the containment tree. Because the structure of this tree is fully known at compile time, this functionality is implemented as a compiled method on the grouper classes; see the `generate_implicit_tree` method in `__ArrayGrouper_functions__.py`.
* **explicitly**: the tree is returned as nested dicts, where dict keys are integers representing label nodes and dict values are dicts representing children of the respective labels. Because of numba type-inference issues, this is not compiled, and therefore is not a method of this class; the module-level function `generate_tree` in `ArrayGrouper.py` can be called on an instance to achieve this. Note, since each label exists only once in the tree, if two different labels point to a common child label, they will point to the same dictionary objects in memory.