
Indexing
========

Just as with other functionality, indexing resembles that of NumPy a
lot.

You can access and modify individual elements of a bsparse matrix using
the usual square bracket syntax. Unlike with `scipy.sparse`, *changes to
the sparsity structure are allowed!*

>>> bcoo = bsp.zeros((3, 2))
>>> bcoo[0, 0]
COO(shape=(1, 1), nnz=0, dtype=float64)

With this empty matrix of ``bshape == (10, 10)``, you can even modify
the row- and column-sizes of the matrix by setting a block.

>>> bcoo[1,0] = np.ones((2,3))
>>> bcoo
BCOO(bshape=(3, 2), bnnz=1 | shape=(4, 4), nnz=6)

When calling ``bsp.zeros()``, all block-sizes are set to ``1``
automatically. Since there are no elements that constrain the sizes, we
can just update the them to accommodate the block.


We can also use slicing syntax to get a sub-block of the matrix.

>>> bcoo[1:, :2]
BCOO(bshape=(2, 2), bnnz=1 | shape=(3, 4), nnz=6)
>>> bcoo[1:, 2::-1]
BCOO(bshape=(2, 2), bnnz=1 | shape=(3, 4), nnz=6)

Another fun thing to do is mixing dense and sparse arrays.

>>> bcoo[0,-1] = bsp.sparse.random((2,2), density=0.9)
>>> bcoo.toarray()
array([[0.        , 0.        , 0.        , 0.53923029, 0.91937085],
       [0.        , 0.        , 0.        , 0.69792673, 0.        ],
       [1.        , 1.        , 1.        , 0.        , 0.        ],
       [1.        , 1.        , 1.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ]])
