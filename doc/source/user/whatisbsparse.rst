
What is bsparse?
================
bsparse implements sparse data containers, that always have an
equivalent dense 2D array representation. As such, bsparse data
containers can store ...

- ... 2D dense arrays of arbitrary shape and ``dtype``.
- ... `scipy.sparse` matrices of arbitrary shape and ``dtype``.
- ... bsparse matrices of arbitrary shape and ``dtype``.
- - ... a mixture of all of the above.

An important caveat is that matrix rows and columns *must not be
ragged*, i. e. each data container row and column always has a uniform
size.

All bsparse data containers implement...

- ... basic arithmetic operations, i. e. addition, subtraction,
  multiplication, and division.
- ... dedicated matrix multiplication.
- ... (conjugate) transposition.
- ... symmetry flags for square symmetric and Hermitian matrices to
  enable a higher compression.
- ... NumPy-like indexing, data modification, as well as changes to the
  sparsity structure.
- ... conversion routines between different storage formats.
- ... a number of instantiation and conversion routines.
- ... loading and saving data structures to disk.

It is the ideal framework if you want to access and modify block-wise
sparse data, and implement algorithms exploiting this structure.

What bsparse is not
-------------------
As it stands, bsparse is not particularly fast. It merely takes
advantage of NumPy's low-level accelerations. 

Unlike `scipy.sparse`, it also currently does not give you access to
high-performance linear algebra routines (although we would like it to
do so in the future).

Why bsparse?
------------
In our work we often encounter large, sparse, often diagonally dominant,
block matrices. What we couldn't find was a suitable data structure in
Python, hence bsparse.
