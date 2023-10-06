import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike

from bsparse._base import BSparse


class BDIA(BSparse):
    """A sparse matrix in DIAgonal format.

    The ``BDIA`` class represents a sparse matrix using two arrays:

    - ``offsets`` contains the offsets of the diagonal elements.
    - ``data`` contains the values of the diagonal elements.

    Parameters
    ----------
    offsets : array_like, optional
        The offsets of the diagonal elements.
    data : array_like, optional
        The values of the diagonal elements.
    shape : tuple, optional
        The shape of the matrix. If not given, it is inferred from
        `offsets`.
    dtype : data-type, optional
        The data type of the matrix. If not given, it is inferred from
        the data type of `data`.

    """

    def __init__(
        self,
        offsets: ArrayLike,
        data: ArrayLike,
        shape: tuple | None = None,
        dtype: np.dtype | None = None,
    ) -> None:
        self.offsets = np.asarray(offsets, dtype=int)
        self.data = np.asarray(data, dtype=dtype)

        if shape is None:
            shape = (self.offsets.max() + 1, self.data.shape[1])
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

    def copy(self) -> "BDIA":
        """Returns a copy of the matrix."""
        return BDIA(self.offsets.copy(), self.data.copy(), self.shape, self.dtype)

    def to_array(self) -> np.ndarray:
        """Converts the matrix to a dense `numpy.ndarray`."""
        arr = np.zeros(self.shape, dtype=self.dtype)
        for i, offset in enumerate(self.offsets):
            np.fill_diagonal(arr[offset:], self.data[i])
        return arr

    def to_coo(self) -> "BDIA":
        """Converts the matrix to `COO` format."""
        pass

    def to_csr(self) -> "BDIA":
        """Converts the matrix to `CSR` format."""
        pass

    def to_dia(self) -> "BDIA":
        """Converts the matrix to `DIA` format."""
        pass

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "BDIA":
        """Creates a sparse matrix from a dense `numpy.ndarray`."""

        data = []
        offsets = []
        for i in range(-arr.shape[0] + 1, arr.shape[1]):
            diag = np.diag(arr, i)
            if np.any(diag):
                data.append(diag)
                offsets.append(i)

        return cls(offsets, data, arr.shape, arr.dtype)

    @classmethod
    def from_spmatrix(cls, mat: sp.spmatrix) -> "BDIA":
        """Creates a sparse matrix from a `scipy.sparse.spmatrix`."""
        mat = sp.dia_array(mat)
        return cls(mat.offsets, mat.data, mat.shape, mat.dtype)
