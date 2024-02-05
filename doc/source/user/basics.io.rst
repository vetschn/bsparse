
I/O
===

You can write bsparse matrices to a file to save them for later use.

>>> import bsparse.sparse as sp
>>> csr = sp.random((100, 100), format="csr")
>>> csr
CSR(shape=(100, 100), nnz=1000, dtype=float64)
>>> csr.save_npz("./csr.npz")

Once you want it back, just load it using ``.load_npz()``

>>> sp.load_npz("test.npz")
CSR(shape=(100, 100), nnz=1000, dtype=float64)
