from numbers import Number
from warnings import warn

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike

from bsparse.sparse.sparse import Sparse


class CSR(Sparse):
    """A sparse matrix in Compressed Sparse Row format.

    The `CSR` class represents a sparse matrix using three arrays:

    * `rowptr`: contains the index of the first element of each row.
    * `cols`: contains the column indices of each non-zero element.
    * `data`: contains the values of each non-zero element.

    .. figure:: ../figures/csr.jpg
        :scale: 25%

    Parameters
    ----------
    rowptr : array_like
        The index of the first element of each row.
    cols : array_like
        The column indices of each non-zero element.
    data : array_like
        The values of each non-zero element.
    shape : tuple, optional
        The shape of the matrix. If not given, it is inferred from the
        row and column indices.
    dtype : numpy.dtype, optional
        The data type of the matrix elements. If not given, it is
        inferred from the data array.
    symmetry : str, optional
        The symmetry of the matrix. If not given, no symmetry is
        assumed. This is only applicable for square matrices, where
        possible values are ``'symmetric'`` and ``'hermitian'``. Note
        that when setting a symmetry, the lower triangular part of the
        matrix is discarded.

    Attributes
    ----------
    rowptr : ndarray
        The index of the first element of each row.
    cols : ndarray
        The column coordinates of the non-zero elements.
    data : ndarray
        The values of the diagonals.
    shape : tuple[int, int]
        The shape of the matrix.
    dtype : dtype
        The data type of the matrix.
    symmetry : str
        The symmetry of the matrix.
    nnz : int
        The number of stored elements in the matrix.
    T : CSR
        The transpose of the matrix.
    H : CSR
        The conjugate transpose of the matrix.


    Examples
    --------
    >>> rowptr = [0, 2, 3, 5]
    >>> cols = [0, 2, 1, 0, 2]
    >>> data = [1, 2, 3, 4, 5]
    >>> csr = CSR(rowptr, cols, data, shape=(3, 3))
    >>> csr.toarray()
    array([[1, 0, 2],
           [0, 3, 0],
           [4, 0, 5]])

    """

    def __init__(
        self,
        rowptr: ArrayLike,
        cols: ArrayLike,
        data: ArrayLike,
        shape: tuple[int, int] = None,
        dtype: np.dtype = None,
        symmetry: str | None = None,
    ) -> None:
        pass

        """Initializes the sparse matrix."""
        self.rowptr = np.asarray(rowptr, dtype=int)
        self.cols = np.asarray(cols, dtype=int)
        self.data = np.asarray(data, dtype=dtype)

        self._dtype = self.data.dtype
        self._shape = self._validate_shape(shape)
        self._symmetry = self._validate_symmetry(symmetry)

    def _validate_shape(self, shape: tuple[int, int] | None) -> tuple[int, int]:
        """Validates the shape of the matrix."""
        if shape is None:
            if self.data.size == 0:
                raise ValueError("Cannot instantiate an empty matrix without shape.")
            return (self.rowptr.size - 1, self.cols.max() + 1)
        if self.data.size != 0:
            if self.rowptr.size - 2 >= shape[0] or self.cols.max() >= shape[1]:
                raise ValueError("Matrix has out-of-bounds indices.")
        if shape[0] < 0 or shape[1] < 0:
            raise ValueError("Matrix has negative shape.")
        return tuple(shape)

    def _validate_symmetry(self, symmetry: str | None) -> str | None:
        """Validates the symmetry of the matrix."""
        if symmetry is None:
            return symmetry
        if symmetry not in ("symmetric", "hermitian"):
            raise ValueError("Invalid symmetry.")
        if self.shape[0] != self.shape[1]:
            raise ValueError("Symmetry is only applicable to square matrices.")

        self._discard_subdiagonal()

        return symmetry

    def _discard_subdiagonal(self) -> None:
        """Takes symmetry into account by removing the lower triangular part."""
        rows = self._expand_rows()
        if any(rows > self.cols):
            warn(
                "Symmetric matrix is not upper triangular. "
                "Lower triangular part is discarded."
            )
            mask = rows <= self.cols
            rowptr = np.zeros(self.shape[0] + 1, dtype=int)
            for row in rows[mask]:
                rowptr[row + 1] += 1
            rowptr = np.cumsum(rowptr)
            self.rowptr = rowptr
            self.cols = self.cols[mask]
            self.data = self.data[mask]

    def _desymmetrize(self) -> "CSR":
        """Removes symmetry."""
        if self.symmetry is None:
            return self

        self_rows = self._expand_rows()
        mask = self_rows != self.cols
        rows = np.concatenate((self_rows, self.cols[mask]))
        cols = np.concatenate((self.cols, self_rows[mask]))
        lower_data = (
            self.data[mask] if self.symmetry == "symmetric" else self.data[mask].conj()
        )
        data = np.concatenate((self.data, lower_data))
        from bsparse.sparse import COO

        result = COO(
            rows,
            cols,
            data,
            self.shape,
            self.dtype,
        )
        return result.tocsr()

    def _unsign_index(self, row: int, col: int) -> tuple:
        """Adjusts the sign to allow negative indices and checks bounds."""
        row = self.shape[0] + row if row < 0 else row
        col = self.shape[1] + col if col < 0 else col
        if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
            raise IndexError("Index out of bounds.")

        return row, col

    def _expand_rows(self) -> np.ndarray:
        """Expands the row indices."""
        rows = np.zeros(self.nnz, dtype=int)
        if self.nnz == 0:
            return rows
        for i in range(self.shape[0]):
            rows[self.rowptr[i] : self.rowptr[i + 1]] = i
        return rows

    def _get_cols(self, row: int) -> np.ndarray:
        """Returns the column indices for the given row."""
        return self.cols[self.rowptr[row] : self.rowptr[row + 1]]

    def _get_data(self, row: int) -> np.ndarray:
        """Returns the data values for the given row."""
        return self.data[self.rowptr[row] : self.rowptr[row + 1]]

    def _getitem_symmetry(self, row: int | slice, col: int | slice):
        """Returns the element at the given coordinates."""
        if isinstance(row, int) and isinstance(col, int):
            row, col = self._unsign_index(row, col)
            if row <= col:
                cols = self._get_cols(row)
                value = self._get_data(row)[cols == col]
                if len(value) == 0:
                    return self.dtype.type(0)
                return value[0]

            if self.symmetry == "symmetric":
                return self[col, row]
            if self.symmetry == "hermitian":
                return np.conj(self[col, row])

        if isinstance(row, int):
            row = slice(row, row + 1)
        if isinstance(col, int):
            col = slice(col, col + 1)

        rows = np.arange(*row.indices(self.shape[0]))
        cols = np.arange(*col.indices(self.shape[1]))
        if len(rows) == 0 or len(cols) == 0:
            raise IndexError("Slice index out of bounds.")
        rows = np.array([self._unsign_index(r, 0)[0] for r in rows])
        cols = np.array([self._unsign_index(0, c)[1] for c in cols])

        row_step = row.step if row.step is not None else 1
        col_step = col.step if col.step is not None else 1

        from bsparse.sparse import COO

        if row.start == col.start and row.stop == col.stop and row_step == col_step:
            # If the slice is symmetric, we need to return a symmetric matrix.

            mask = np.isin(self._expand_rows(), rows) & np.isin(self.cols, cols)

            submatrix = COO(
                (self._expand_rows()[mask] - rows[0]) // row_step,
                (self.cols[mask] - cols[0]) // col_step,
                self.data[mask],
                shape=(len(rows), len(cols)),
                dtype=self.dtype,
                symmetry=self.symmetry,
            )
            return submatrix.tocsr()

        submatrix = COO([], [], [], shape=(len(rows), len(cols)), dtype=self.dtype)
        for i, j in np.ndindex(submatrix.shape):
            value = self[int(rows[i]), int(cols[j])]
            if value == 0:
                continue
            submatrix.rows = np.append(submatrix.rows, (rows[i] - rows[0]) // row_step)
            submatrix.cols = np.append(submatrix.cols, (cols[j] - cols[0]) // col_step)
            submatrix.data = np.append(submatrix.data, value)
        return submatrix.tocsr()

    def _getslice(self, row: slice, col: slice):
        """Returns a submatrix."""
        rows = np.arange(*row.indices(self.shape[0]))
        cols = np.arange(*col.indices(self.shape[1]))
        if len(rows) == 0 or len(cols) == 0:
            raise IndexError("Slice index out of bounds.")
        rows = np.array([self._unsign_index(r, 0)[0] for r in rows])
        cols = np.array([self._unsign_index(0, c)[1] for c in cols])
        row_step = row.step if row.step is not None else 1
        col_step = col.step if col.step is not None else 1
        mask = np.isin(self._expand_rows(), rows) & np.isin(self.cols, cols)

        from bsparse.sparse import COO

        submatrix = COO(
            (self._expand_rows()[mask] - rows[0]) // row_step,
            (self.cols[mask] - cols[0]) // col_step,
            self.data[mask],
            shape=(len(rows), len(cols)),
            dtype=self.dtype,
        )
        return submatrix.tocsr()

    def __getitem__(
        self, key: int | slice | tuple[int | slice, int | slice]
    ) -> "Number | CSR":
        """Returns a matrix element or a submatrix."""
        if isinstance(key, (int, slice)):
            key = (key, slice(None))

        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("Invalid index")

        row, col = key
        if not isinstance(row, (int, slice)) or not isinstance(col, (int, slice)):
            raise IndexError("Invalid index")

        if self.symmetry is not None:
            return self._getitem_symmetry(row, col)

        if isinstance(row, int) and isinstance(col, int):
            row, col = self._unsign_index(row, col)
            cols = self._get_cols(row)
            value = self._get_data(row)[cols == col]
            if len(value) == 0:
                return self.dtype.type(0)
            return value[0]

        if isinstance(row, int):
            row = slice(row, row + 1)
        if isinstance(col, int):
            col = slice(col, col + 1)

        return self._getslice(row, col)

    def __setitem__(
        self,
        key: tuple[int, int],
        value: "Number",
    ) -> None:
        """Sets a matrix element."""
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("Invalid index")

        row, col = key
        if not isinstance(row, int) or not isinstance(col, int):
            raise IndexError("Invalid index")

        row, col = self._unsign_index(row, col)
        if self.symmetry is not None and row > col:
            if self.symmetry == "symmetric":
                self[col, row] = value
                return
            if self.symmetry == "hermitian":
                self[col, row] = np.conj(value)
                return

        if any(self._get_cols(row) == col):
            if value == 0:
                mask = ~((self._expand_rows() == row) & (self.cols == col))
                self.rowptr[row + 1 :] -= 1
                self.cols = self.cols[mask]
                self.data = self.data[mask]
                return

            self._get_data(row)[self._get_cols(row) == col] = value
            return

        if value == 0:
            return

        if row not in self._expand_rows():
            self.rowptr = np.insert(self.rowptr, row + 1, self.nnz)
        self.rowptr[row + 1 :] += 1
        self.cols = np.insert(self.cols, self.rowptr[row + 1] - 1, col)
        self.data = np.insert(self.data, self.rowptr[row + 1] - 1, value)

    def __add__(
        self, other: Number | Sparse | np.ndarray | sp.sparray
    ) -> "CSR | np.ndarray":
        """Adds another matrix or a scalar to this matrix."""
        result = self.tocoo() + other
        if isinstance(result, np.ndarray):
            return result
        return result.tocsr()

    def __radd__(
        self, other: Number | Sparse | np.ndarray | sp.sparray
    ) -> "CSR | np.ndarray":
        """Adds this matrix to another matrix or a scalar."""
        return self + other

    def __sub__(
        self, other: Number | Sparse | np.ndarray | sp.sparray
    ) -> "CSR | np.ndarray":
        """Subtracts another matrix or a scalar from this matrix."""
        return self + (-other)

    def __rsub__(
        self, other: Number | Sparse | np.ndarray | sp.sparray
    ) -> "CSR | np.ndarray":
        """Subtracts this matrix from another matrix or a scalar."""
        return other + (-self)

    def __mul__(
        self, other: Number | Sparse | np.ndarray | sp.sparray
    ) -> "CSR | np.ndarray":
        """Multiplies another matrix or a scalar by this matrix."""
        result = self.tocoo() * other
        if isinstance(result, np.ndarray):
            return result
        return result.tocsr()

    def __rmul__(
        self, other: Number | Sparse | np.ndarray | sp.sparray
    ) -> "CSR | np.ndarray":
        """Multiplies this matrix by another matrix or a scalar."""
        return self * other

    def __truediv__(
        self, other: Number | Sparse | np.ndarray | sp.sparray
    ) -> "CSR | np.ndarray":
        """Divides this matrix by another matrix or a scalar."""
        result = self.tocoo() / other
        if isinstance(result, np.ndarray):
            return result
        return result.tocsr()

    def __rtruediv__(
        self, other: Number | Sparse | np.ndarray | sp.sparray
    ) -> "CSR | np.ndarray":
        """Divides another matrix or a scalar by this matrix."""
        result = other / self.tocoo()
        if isinstance(result, np.ndarray):
            return result
        return result.tocsr()

    def __neg__(self) -> "CSR":
        """Negates this matrix."""
        result = CSR(
            self.rowptr.copy(),
            self.cols.copy(),
            -self.data.copy(),
            self.shape,
            self.dtype,
            self.symmetry,
        )
        return result

    def _matmul_dense(self, other: np.ndarray) -> np.ndarray:
        """Multiplies this matrix by a dense matrix."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Incompatible matrix shapes.")

        result = np.zeros((self.shape[0], other.shape[1]), dtype=self.dtype)
        for i in range(self.shape[0]):
            for j, a in zip(self._get_cols(i), self._get_data(i)):
                result[i] += a * other[j]
        return result

    def __matmul__(
        self, other: Number | Sparse | np.ndarray | sp.sparray
    ) -> "CSR | np.ndarray":
        """Multiplies this matrix by another matrix."""
        if isinstance(other, np.ndarray):
            return self._desymmetrize()._matmul_dense(other)
        if isinstance(other, Sparse):
            other = other.tocsr()
        if sp.issparse(other):
            other = CSR.from_sparray(other)

        if not isinstance(other, CSR):
            raise TypeError("Invalid type.")

        if self.shape[1] != other.shape[0]:
            raise ValueError("Incompatible matrix shapes.")

        if self.symmetry is not None or other.symmetry is not None:
            # Products of symmetric matrices are not necessarily symmetric.
            return self._desymmetrize() @ other._desymmetrize()

        result_rowptr = [0]
        result_cols = []
        result_data = []
        for i in range(self.shape[0]):
            data = []
            cols = []
            for j, a in zip(self._get_cols(i), self._get_data(i)):
                for k, b in zip(other._get_cols(j), other._get_data(j)):
                    if k in cols:
                        data[cols.index(k)] += a * b
                    else:
                        cols.append(k)
                        data.append(a * b)
            result_cols.extend(cols)
            result_data.extend(data)
            result_rowptr.append(len(result_cols))

        result = CSR(
            result_rowptr,
            result_cols,
            result_data,
            (self.shape[0], other.shape[1]),
            self.dtype,
        )

        return result

    def __rmatmul__(
        self, other: Number | Sparse | np.ndarray | sp.sparray
    ) -> "CSR | np.ndarray":
        """Multiplies another matrix by this matrix."""
        if isinstance(other, np.ndarray):
            return other @ self.toarray()
        if isinstance(other, Sparse):
            return other.tocsr() @ self
        if sp.issparse(other):
            return CSR.from_sparray(other) @ self

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

    @property
    def symmetry(self) -> str | None:
        """The symmetry of the matrix."""
        return self._symmetry

    @property
    def T(self) -> "CSR":
        """The transpose of the matrix."""
        if self.symmetry == "symmetric":
            return self
        if self.symmetry == "hermitian":
            return self.conjugate()

        from bsparse.sparse import COO

        transpose = COO(
            self.cols,
            self._expand_rows(),
            self.data,
            (self.shape[1], self.shape[0]),
            self.dtype,
            self.symmetry,
        )
        return transpose.tocsr()

    @property
    def H(self) -> "CSR":
        """The conjugate transpose of the matrix."""
        if self.symmetry == "hermitian":
            return self
        if self.symmetry == "symmetric":
            return self.conjugate()

        from bsparse.sparse import COO

        hermitian = COO(
            self.cols,
            self._expand_rows(),
            np.conj(self.data),
            (self.shape[1], self.shape[0]),
            self.dtype,
            self.symmetry,
        )
        return hermitian.tocsr()

    def conjugate(self) -> "CSR":
        """The complex conjugate of the matrix."""
        conjugate = CSR(
            self.rowptr,
            self.cols,
            np.conj(self.data),
            self.shape,
            self.dtype,
            self.symmetry,
        )
        return conjugate

    def diagonal(self, offset: int = 0) -> np.ndarray:
        """Returns the diagonal of the matrix."""
        if not -self.shape[0] < offset < self.shape[1]:
            raise ValueError("Offset out of bounds.")

        if offset < 0:
            if self.symmetry == "hermitian":
                return np.conj(self.diagonal(-offset))
            if self.symmetry == "symmetric":
                return self.diagonal(-offset)

        start = (-offset) * self.shape[1] if offset < 0 else offset

        rows = []
        cols = []
        for flat_ind in range(start, self.shape[0] * self.shape[1], self.shape[1] + 1):
            if flat_ind // self.shape[1] >= self.shape[1] - offset:
                break
            rows.append(flat_ind // self.shape[1])
            cols.append(flat_ind % self.shape[1])

        diag = np.zeros(len(rows), dtype=self.dtype)
        for i, (row, col) in enumerate(zip(rows, cols)):
            value = self.data[(self._expand_rows() == row) & (self.cols == col)]
            if len(value) == 0:
                continue
            diag[i] = value[0]
        return diag

    def copy(self) -> "CSR":
        """Returns a copy of the matrix."""
        new = CSR(
            self.rowptr.copy(),
            self.cols.copy(),
            self.data.copy(),
            self.shape,
            self.dtype,
            self.symmetry,
        )
        return new

    def astype(self, dtype: np.dtype) -> "CSR":
        """Returns a copy of the matrix with a different data type."""
        new = CSR(
            self.rowptr.copy(),
            self.cols.copy(),
            self.data.copy(),
            self.shape,
            dtype,
            self.symmetry,
        )
        return new

    def toarray(self) -> np.ndarray:
        """Converts the matrix to a dense `numpy.ndarray`."""
        arr = np.zeros(self.shape, dtype=self.dtype)
        rows = self._expand_rows()
        arr[rows, self.cols] = self.data
        if self.symmetry is not None:
            mask = rows != self.cols
            if self.symmetry == "symmetric":
                arr[self.cols[mask], rows[mask]] = self.data[mask]
            if self.symmetry == "hermitian":
                arr[self.cols[mask], rows[mask]] = np.conj(self.data[mask])
        return arr

    def tocoo(self) -> "Sparse":
        """Converts the matrix to `COO` format."""
        from bsparse.sparse.coo import COO

        coo = COO(
            self._expand_rows(),
            self.cols,
            self.data,
            self.shape,
            self.dtype,
            self.symmetry,
        )
        return coo

    def tocsr(self) -> "Sparse":
        """Converts the matrix to `CSR` format."""
        return self

    def todia(self) -> "Sparse":
        """Converts the matrix to `DIA` format."""
        from bsparse.sparse.dia import DIA

        offsets, offset_indices = np.unique(
            self.cols - self._expand_rows(), return_inverse=True
        )
        if len(self.data) == 0:
            data = np.zeros((0, 0), dtype=self.dtype)
            return DIA(offsets, data, self.shape, self.dtype, self.symmetry)

        data = np.zeros((len(offsets), self.cols.max() + 1), dtype=self.dtype)
        data[offset_indices, self.cols] = self.data

        return DIA(offsets, data, self.shape, self.dtype, self.symmetry)

    def save_npz(self, filename: str) -> None:
        """Saves the matrix as ``.npz`` archive."""
        np.savez_compressed(
            filename,
            format="csr",
            rowptr=self.rowptr,
            cols=self.cols,
            data=self.data,
            shape=self.shape,
            dtype=self.dtype,
            symmetry=self.symmetry,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray, symmetry: str | None = None) -> "CSR":
        """Creates a sparse matrix from a dense `numpy.ndarray`."""
        arr = np.asarray(arr)
        rows, cols = arr.nonzero()
        data = arr[rows, cols]

        rowptr = np.zeros(arr.shape[0] + 1, dtype=int)
        for row in rows:
            rowptr[row + 1] += 1
        rowptr = np.cumsum(rowptr)
        return cls(rowptr, cols, data, arr.shape, symmetry=symmetry)

    @classmethod
    def from_sparray(cls, mat: sp.sparray, symmetry: str | None = None) -> "CSR":
        """Creates a sparse matrix from a `scipy.sparse.sparray`."""
        mat = sp.csr_array(mat)
        return cls(mat.indptr, mat.indices, mat.data, mat.shape, symmetry=symmetry)
