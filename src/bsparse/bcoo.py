from warnings import warn

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike

from bsparse._base import BSparse


class BCOO(BSparse):
    """A sparse matrix container in COOrdinate format.

    The ``BCOO`` class represents a sparse matrix using three arrays:

    - ``rows`` contains the row coordinates of the non-zero elements.
    - ``cols`` contains the column coordinates of the non-zero elements.
    - ``data`` contains the values of the non-zero elements.

    Upon creation, the matrix is sorted (lexicographically) by rows and
    columns. Duplicate elements are not allowed.

    Parameters
    ----------
    rows : array_like
        The row coordinates of the non-zero elements.
    cols : array_like
        The column coordinates of the non-zero elements.
    data : array_like
        The values of the non-zero constituents. This is an array of
        dtype object.
    bshape : tuple, optional
        The shape of the matrix container. If not given, it is inferred
        from ``rows`` and ``cols``.
    shape : tuple, optional
        The shape of the matrix. If not given, it is inferred from the
        cumulative shape of the matrix elements.
    symmetry : str, optional
        The symmetry of the matrix. If not given, it is assumed to be
        ``None``. Possible values are:

        - ``None``
        - ``"symmetric"``
        - ``"hermitian"``
        - ``"skew-symmetric"``
        - ``"skew-hermitian"``

        Upon applying symmetry, the lower triangular part of the matrix
        is discarded.

    """

    def __init__(
        self,
        rows: ArrayLike,
        cols: ArrayLike,
        data: ArrayLike,
        bshape: tuple[int, int] | None = None,
        symmetry: str | None = None,
    ) -> None:
        """Initializes a ``BCOO`` matrix."""
        self.rows = np.asarray(rows, dtype=int)
        self.cols = np.asarray(cols, dtype=int)
        self.data = np.asarray(data, dtype=object)

        if bshape is None:
            bshape = (self.rows.max() + 1, self.cols.max() + 1)
        if self.rows.size != 0:  # Allows empty matrices.
            if self.rows.max() >= bshape[0] or self.cols.max() >= bshape[1]:
                raise ValueError("Matrix has out-of-bounds indices.")
        self._bshape = bshape

        self._check_data()

        self._symmetry = symmetry
        if symmetry in ("symmetric", "hermitian", "skew-symmetric", "skew-hermitian"):
            if self.bshape[0] != self.bshape[1]:
                raise ValueError("Symmetry is only applicable to square matrices.")
            skew = -1 if symmetry.startswith("skew") else 1
            if symmetry.endswith("symmetric"):

                def _map(value):
                    if np.isscalar(value):
                        return skew * value
                    return self * value.T

                self._symmetry_map = _map

            if symmetry.endswith("hermitian"):

                def _map(value):
                    if np.isscalar(value):
                        return skew * np.conjugate(value)
                    if isinstance(value, np.ndarray):
                        return skew * value.conj().T
                    return self * value.H

                self._symmetry_map = _map

            self._discard_subdiagonal()
        elif symmetry is not None:
            raise ValueError(
                "Invalid symmetry. Possible values are: None, symmetric, "
                "hermitian, skew-symmetric, skew-hermitian."
            )

        self.sort_indices()

    def _check_data(self) -> None:
        """Checks that the data adheres to the specification."""
        # Check that there are no duplicate elements.
        if self.data.size != len(set(zip(self.rows, self.cols))):
            raise ValueError("Matrix has duplicate elements.")

        # Check that the matrix elements are scalars or two dimensional.
        for b in self.data:
            if not np.isscalar(b) and b.ndim != 2:
                raise ValueError("Matrix elements must be scalars or two dimensional.")

        # Check that the matrix rows are aligned.
        for row in self.rows:
            data = self.data[self.rows == row]
            if len(set([1 if np.isscalar(b) else b.shape[0] for b in data])) != 1:
                raise ValueError("Matrix rows are not aligned.")

        # Check that the matrix columns are aligned.
        for col in self.cols:
            data = self.data[self.cols == col]
            if len(set([1 if np.isscalar(b) else b.shape[0] for b in data])) != 1:
                raise ValueError("Matrix columns are not aligned.")

    def _discard_subdiagonal(self) -> None:
        """Takes symmetry into account by removing the lower triangular part."""
        if any(self.rows > self.cols):
            warn("Matrix is not upper triangular. Lower triangular part is discarded.")
            ind = self.rows <= self.cols
            self.rows = self.rows[ind]
            self.cols = self.cols[ind]
            self.data = self.data[ind]

    def _unsign_indices(self, row: int, col: int) -> tuple:
        """Adjusts the sign to allow negative indices and checks bounds."""
        # Unsign the indices.
        if row < 0:
            row = self.bshape[0] + row
        if col < 0:
            col = self.bshape[1] + col
        if not (0 <= row < self.bshape[0] and 0 <= col < self.bshape[1]):
            raise IndexError("Block index out of bounds.")

        return row, col

    def __getitem__(
        self, key: int | slice | list | tuple
    ) -> "np.ndarray | int | float | complex | BCOO":
        """Returns a matrix element or a submatrix."""
        if isinstance(key, (int, slice)):
            return self[key, :]

        if isinstance(key, list):
            return [self[k] for k in key]

        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Invalid number of indices.")

            row, col = key
            if isinstance(row, int) and isinstance(col, int):
                row, col = self._unsign_indices(row, col)
                apply_symmetry_map = self.symmetry is not None and row > col
                row, col = (col, row) if apply_symmetry_map else (row, col)

                value = self.data[self.rows == row & self.cols == col]
                if value.size == 0:
                    value = [np.zeros((self.row_sizes[row], self.col_sizes[col]))]
                if apply_symmetry_map:
                    value = self._symmetry_map(value)
                return value[0]

            if isinstance(row, int) and isinstance(col, slice):
                cols = np.arange(*col.indices(self.bshape[1]))
                trafos, __, cols = zip(*[self._unsign_indices(row, c) for c in cols])
                row = self.rows == row
                col = np.isin(self.cols, cols)
                value = BCOO(
                    np.zeros(len(cols), dtype=int),
                    self.cols[col] - cols[0],
                    [trafo(b) for trafo, b in zip(trafos, self.data[col])],
                    (1, len(cols)),
                )
                return value

            if isinstance(row, slice) and isinstance(col, int):
                rows = np.arange(*row.indices(self.bshape[0]))
                trafos, rows, __ = zip(*[self._unsign_indices(r, col) for r in rows])
                row = np.isin(self.rows, rows)
                col = self.cols == col
                value = BCOO(
                    self.rows[row] - rows[0],
                    np.zeros(len(rows), dtype=int),
                    [trafo(b) for trafo, b in zip(trafos, self.data[row])],
                    (len(rows), 1),
                )
                return value

            if isinstance(row, slice) and isinstance(col, slice):
                rows = np.arange(*row.indices(self.bshape[0]))
                cols = np.arange(*col.indices(self.bshape[1]))
                trafos, rows, cols = zip(
                    *[self._unsign_indices(r, c) for r in rows for c in cols]
                )
                row = np.isin(self.rows, rows)
                col = np.isin(self.cols, cols)
                value = BCOO(
                    self.rows[row] - rows[0],
                    self.cols[col] - cols[0],
                    [trafo(b) for trafo, b in zip(trafos, self.data[row & col])],
                    (len(rows), len(cols)),
                )
                return value

        raise NotImplementedError

    def __setitem__(
        self,
        key: int | slice | tuple[int | slice, int | slice],
        value: "np.ndarray | int | float | complex | BCOO",
    ):
        """Sets a matrix element or a submatrix."""
        if not isinstance(value, (np.ndarray, int, float, complex, BCOO)):
            raise TypeError("Value must be a matrix or a scalar.")

        if isinstance(key, int):
            row = self.rows == key
            self.data[row] = value
            return

    def __add__(self, other: "np.number | BSparse") -> "BCOO":
        """Adds another matrix or a scalar to this matrix."""
        raise NotImplementedError

    def __sub__(self, other: "np.number | BSparse") -> "BCOO":
        """Subtracts another matrix or a scalar from this matrix."""
        raise NotImplementedError

    def __rsub__(self, other: "np.number | BSparse") -> "BCOO":
        """Subtracts this matrix from another matrix or a scalar."""
        raise NotImplementedError

    def __mul__(self, other: "np.number | BSparse") -> "BCOO":
        """Multiplies another matrix or a scalar by this matrix."""
        raise NotImplementedError

    def __truediv__(self, other: "np.number | BSparse") -> "BCOO":
        """Divides this matrix by another matrix or a scalar."""
        raise NotImplementedError

    def __rtruediv__(self, other: "np.number | BSparse") -> "BCOO":
        """Divides another matrix or a scalar by this matrix."""
        raise NotImplementedError

    def __neg__(self) -> "BCOO":
        """Negates this matrix."""
        raise NotImplementedError

    def __matmul__(self, other: "BSparse") -> "BCOO":
        """Multiplies this matrix by another matrix."""
        raise NotImplementedError

    def __rmatmul__(self, other: "BSparse") -> "BCOO":
        """Multiplies another matrix by this matrix."""
        raise NotImplementedError

    @property
    def T(self) -> "BSparse":
        """The transpose of the matrix."""
        raise NotImplementedError

    @property
    def H(self) -> "BSparse":
        """The conjugate transpose of the matrix."""
        raise NotImplementedError

    @property
    def symmetry(self) -> str | None:
        """The symmetry of the matrix."""
        return self._symmetry

    @property
    def bshape(self) -> tuple[int, int]:
        """The block shape of the matrix."""
        return self._bshape

    @property
    def row_sizes(self) -> np.ndarray:
        """The sizes of the row elements."""
        sizes = np.zeros(self.bshape[0], dtype=int)
        for row in range(self.bshape[0]):
            data = self.data[self.rows == row]
            if data.size == 0:
                sizes[row] = 0
            elif np.isscalar(data[0]):
                sizes[row] = 1
            else:
                sizes[row] = data[0].shape[0]
        return sizes

    @property
    def col_sizes(self) -> np.ndarray:
        """The sizes of the column elements."""
        sizes = np.zeros(self.bshape[1], dtype=int)
        for col in range(self.bshape[1]):
            data = self.data[self.cols == col]
            if data.size == 0:
                # No elements in this column.
                sizes[col] = 0
            elif hasattr(data[0], "shape"):
                # Matrix elements.
                sizes[col] = data[0].shape[1]
            else:
                # Scalar elements.
                sizes[col] = 1
        return sizes

    @property
    def bnnz(self) -> int:
        """The number of non-zero elements in the matrix."""
        return self.data.size

    @property
    def nnz(self) -> int:
        """The number of non-zero elements in the matrix."""
        raise NotImplementedError

    def sort_indices(self) -> None:
        """Sorts the matrix by rows and columns."""
        order = np.lexsort((self.cols, self.rows))
        self.rows = self.rows[order]
        self.cols = self.cols[order]
        self.data = self.data[order]

    def copy(self) -> "BCOO":
        """Returns a copy of the matrix."""
        return BCOO(
            self.rows.copy(),
            self.cols.copy(),
            self.data.copy(),
            self.shape,
        )

    def to_coo(self) -> "BCOO":
        """Converts the matrix to `COO` format."""
        return self

    def to_csr(self) -> "BSparse":
        """Converts the matrix to `CSR` format."""
        indptr = np.zeros(self.shape[0] + 1, dtype=int)
        for row in self.rows:
            indptr[row + 1] += 1
        indptr = np.cumsum(indptr)

        # from bsparse import BCSR
        raise NotImplementedError

    def to_dia(self) -> "BSparse":
        """Converts the matrix to `DIA` format."""
        # from bsparse import DIA
        raise NotImplementedError

    def to_array(self) -> np.ndarray:
        """Converts the matrix to a dense `numpy.ndarray`."""
        raise NotImplementedError

    def save_npz(self, filename: str) -> None:
        """Saves the matrix to a `numpy.npz` file."""
        raise NotImplementedError

    @classmethod
    def from_array(cls, arr: ArrayLike, symmetry: str | None = None) -> "BCOO":
        """Creates a `BCOO` matrix from a dense `numpy.ndarray`."""
        arr = np.asarray(arr)
        rows, cols = arr.nonzero()
        data = arr[rows, cols]
        return cls(rows, cols, data, arr.shape, symmetry)

    @classmethod
    def from_spmatrix(cls, mat: sp.spmatrix) -> "BCOO":
        """Creates a `BCOO` matrix from a `scipy.sparse.spmatrix`."""
        mat = sp.coo_array(mat)
        return cls(mat.row, mat.col, mat.data, mat.shape)
