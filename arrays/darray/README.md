# darray

Wrapper around the numpy array that automatically deforests chained operations.

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

But this is much more verbose than the one-liner `result = a + b * (c**2 + d**2)`

This module defines the DeforestedArray type, which allows for
keeping concise syntax while preventing unnecessary intermediate
arrays from being created. This is accomplished by lazy evaluation
and an `evaluate` method:
	result = (a + b * (c**2 + d**2)).evaluate()