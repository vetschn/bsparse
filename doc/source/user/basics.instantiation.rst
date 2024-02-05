
Instantiation
=============

There are several different mechanisms for instantiating ``sparse`` and
``bsparse`` matrices. This is a description of the supported mechanisms.

Explicit Instantiation
----------------------
Sparse matrices can be instantiated by calling the ``__init__()`` method
of the respective matrix class, this is however not recommended unless
you know precisely what you are doing (*especially for DIA matrices*).

>>> import bsparse as bsp
>>> import bsparse.sparse as sp
>>> bsp.BCOO(
...     rows=[1, 3],
...     cols=[1, 2],
...     data=[np.ones((10, 10)), np.ones((2, 5))]
... )
BCOO(bshape=(4, 3), bnnz=2 | shape=(14, 16), nnz=110)


From ndarray
------------
You can create a sparse matrix from an `numpy.ndarray` aby calling the
``.from_array()`` classmethod of the desired matrix format.

>>> sp.DIA.from_array(np.eye(10, k=3))
DIA(shape=(10, 10), nnz=10, dtype=float64)


From sparray
------------
One can also create bsparse matrices from `scipy.sparse` matrices using
the ``.from_sparray()`` classmethod.

>>> bsp.sparse.CSR.from_sparray(scipy.sparse.random(100,100))
CSR(shape=(100, 100), nnz=100, dtype=float64)


Zeros
-----
You can create all-zero (``.nnz == 0``) bsparse matrices by calling the
``bsp.zeros()`` routine.

>>> bsp.zeros((10, 10))
BCOO(bshape=(10, 10), bnnz=0 | shape=(10, 10), nnz=0)
>>> bsp.zeros((10, 10), format="bdia")
BDIA(bshape=(10, 10), bnnz=0 | shape=(10, 10), nnz=0)

Identity
--------
Creating identity matrices works just as with NumPy.

>>> bsp.eye((3, 4))
BCOO(bshape=(3, 4), bnnz=3 | shape=(3, 4), nnz=3)
>>> bsp.eye((3, 4)).toarray()
array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.]])


Random
------
Just as with `scipy.sparse` you can also create random sparse matrices
of a given density.

>>> bsp.random((10, 10), density=0.5)
BCOO(bshape=(10, 10), bnnz=50 | shape=(10, 10), nnz=50)
>>> bsp.random((10, 10), density=0.5, format="bcsr")
BCSR(bshape=(10, 10), bnnz=50 | shape=(10, 10), nnz=50)


From Diagonals
--------------
Just as with NumPy you can create matrices with given diagonal values.

>>> bsp.sparse.diag([1, 2, 3, 4])
COO(shape=(4, 4), nnz=4, dtype=int64)
>>> bsp.sparse.diag([1, 2, 3, 4]).toarray()
array([[1, 0, 0, 0],
       [0, 2, 0, 0],
       [0, 0, 3, 0],
       [0, 0, 0, 4]])
>>> bsp.sparse.diag([1, 2, 3, 4], offset=2)
COO(shape=(6, 6), nnz=4, dtype=int64)
>>> bsp.sparse.diag([1, 2, 3, 4], offset=2).toarray()
array([[0, 0, 1, 0, 0, 0],
       [0, 0, 0, 2, 0, 0],
       [0, 0, 0, 0, 3, 0],
       [0, 0, 0, 0, 0, 4],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0]])

