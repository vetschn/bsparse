import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike

from bsparse.bsparse import BSparse


class BCSR(BSparse):
    """A sparse matrix in Compressed Sparse Row format.

    The `CSR` class represents a sparse matrix using three arrays:

    - ``indptr`` contains the index of the first element of each row.
    - ``indices`` contains the column indices of each non-zero element.
    - ``data`` contains the values of each non-zero element.

    Parameters
    ----------
    indptr : array_like, optional
        The index of the first element of each row.
    indices : array_like, optional
        The column indices of each non-zero element.
    data : array_like, optional
        The values of each non-zero element.
    shape : tuple, optional
        The shape of the matrix. If not given, it is inferred from
        ``indptr``.
    dtype : data-type, optional
        The data type of the matrix. If not given, it is inferred from
        the data type of ``data``.
    symmetry : str, optional
        The symmetry of the matrix. If not given, it is assumed to be
        ``None``. Possible values are:

        - ``None``
        - ``"symmetric"``
        - ``"hermitian"``
        - ``"skew-symmetric"``
        - ``"skew-hermitian"``

    """

    def __init__(
        self,
        indptr: ArrayLike,
        indices: ArrayLike,
        data: ArrayLike,
        shape: tuple | None = None,
        dtype: np.dtype | None = None,
        symmetry: str | None = None,
    ) -> None:
        self.indptr = np.asarray(indptr, dtype=int)
        self.indices = np.asarray(indices, dtype=int)
        self.data = np.asarray(data, dtype=dtype)

        if shape is None:
            shape = (self.indptr.max(), self.indices.max() + 1)
        self._shape = shape

    @property
    def shape(self) -> tuple[int, int]:
        """The shape of the matrix."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """The data type of the matrix."""
        return self.data.dtype

    @property
    def nnz(self) -> int:
        """The number of non-zero elements in the matrix."""
        return self.data.size

    def _get_cols(self, row: int) -> np.ndarray:
        """Returns the column indices for the given row."""
        return self.indices[self.indptr[row] : self.indptr[row + 1]]

    def _get_data(self, row: int) -> np.ndarray:
        """Returns the data values for the given row."""
        return self.data[self.indptr[row] : self.indptr[row + 1]]

    def copy(self) -> "BCSR":
        """Returns a copy of the matrix."""
        return BCSR(
            self.indptr.copy(),
            self.indices.copy(),
            self.data.copy(),
            self.shape,
            self.dtype,
        )

    def to_array(self) -> np.ndarray:
        """Converts the matrix to a dense `numpy.ndarray`."""
        arr = np.zeros(self.shape, dtype=self.dtype)
        for row in range(self.shape[0]):
            cols = self._get_cols(row)
            data = self._get_data(row)
            for col, val in zip(cols, data):
                arr[row, col] = val

        return arr

    def to_coo(self) -> "BSparse":
        """Converts the matrix to `COO` format."""
        from bsparse import BCOO

        rows = np.repeat(np.arange(self.shape[0]), np.diff(self.indptr))
        return BCOO(rows, self.indices, self.data, self.shape, self.dtype)

    def to_csr(self) -> "BCSR":
        """Converts the matrix to `CSR` format."""
        return self

    def to_dia(self) -> "BSparse":
        """Converts the matrix to `DIA` format."""
        raise NotImplementedError

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "BCSR":
        """Creates a `CSR` matrix from a dense `numpy.ndarray`."""
        from bsparse import BCOO

        return BCOO.from_array(arr).to_csr()

    @classmethod
    def from_spmatrix(cls, mat: sp.spmatrix) -> "BCSR":
        """Creates a `CSR` matrix from a `scipy.sparse.spmatrix`."""
        mat = sp.csr_array(mat)
        return cls(mat.indptr, mat.indices, mat.data, mat.shape, mat.dtype)
