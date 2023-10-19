from numbers import Integral, Number
from warnings import warn

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike

from bsparse import sparse
from bsparse.bsparse import BSparse


class BCSR(BSparse):
    """A sparse matrix in Compressed Sparse Row format.

    The ``CSR`` class represents a sparse matrix using three arrays:

    * ``rowptr``: contains the index of the first element of each row.
    * ``cols``: contains the column indices of each non-zero element.
    * ``data``: contains the values of each non-zero element.

    Parameters
    ----------
    rowptr : array_like
        The index of the first element of each row.
    cols : array_like
        The column indices of each non-zero element.
    data : array_like
        The values of each non-zero element.
    bshape : tuple, optional
        The shape of the matrix container. If not given, it is inferred
        from ``indptr`` and ``indices``.
    dtype : numpy.dtype, optional
        The data type of the matrix elements. If not given, it is
        inferred from the data array.
    sizes : tuple[np.ndarray, np.ndarray], optional
        The sizes of the blocks. If not given, they are inferred from
        ``data``.
    symmetry : str, optional
        The symmetry of the matrix. If not given, no symmetry is
        assumed. This is only applicable for square matrices, where
        possible values are ``'symmetric'`` and ``'hermitian'``. Note
        that when setting a symmetry, the lower triangular part of the
        matrix is discarded.

    """

    def __init__(
        self,
        rowptr: ArrayLike,
        cols: ArrayLike,
        data: ArrayLike,
        bshape: tuple | None = None,
        dtype: np.dtype | None = None,
        sizes: tuple[np.ndarray, np.ndarray] | None = None,
        symmetry: str | None = None,
    ) -> None:
        """Initializes a ``BCSR`` matrix."""
        self.rowptr = np.asarray(rowptr, dtype=int)
        self.cols = np.asarray(cols, dtype=int)

        data = list(data)
        self.data = self._validate_data(data)

        if dtype is None and len(self.data) != 0:
            dtype = np.result_type(*[b.dtype for b in self.data])
        if dtype is None:
            dtype = np.dtype(float)
        self.data = [b.astype(dtype) for b in self.data]
        self._dtype = dtype

        self._bshape = self._validate_bshape(bshape)
        self._symmetry = self._validate_symmetry(symmetry)

        self._check_alignment()

        if sizes is not None:
            self._row_sizes = np.asarray(sizes[0], dtype=int)
            self._col_sizes = np.asarray(sizes[1], dtype=int)
        else:
            self._row_sizes = self.row_sizes
            self._col_sizes = self.col_sizes

    def _validate_data(self, data: ArrayLike) -> list:
        """Validates the data blocks of the matrix."""
        # Check that the matrix blocks allow an equivalent array representation.
        for b in data:
            if b.ndim != 2:
                raise ValueError("Matrix blocks must be two dimensional.")
            if not hasattr(b, "shape"):
                raise ValueError("Matrix blocks must have a `shape` attribute.")
            if not hasattr(b, "dtype"):
                raise ValueError("Matrix blocks must have a `dtype` attribute.")
            if not hasattr(b, "astype"):
                raise ValueError("Matrix blocks must implement an `astype` method.")

        return data

    def _check_alignment(self) -> None:
        """Checks that the matrix rows and columns are aligned."""
        rows = self._expand_rows()
        for row in rows:
            row_data = [b for b, r in zip(self.data, rows) if r == row]
            if len(set([b.shape[0] for b in row_data])) != 1:
                raise ValueError("Matrix rows are not aligned.")

        # Check that the matrix columns are aligned.
        for col in self.cols:
            col_data = [b for b, c in zip(self.data, self.cols) if c == col]
            if len(set([b.shape[1] for b in col_data])) != 1:
                raise ValueError("Matrix columns are not aligned.")

    def _validate_bshape(self, bshape: tuple[int, int] | None) -> tuple[int, int]:
        """Validates the shape of the matrix."""
        if bshape is None:
            if len(self.data) == 0:
                raise ValueError("Cannot instantiate an empty matrix without bshape.")
            return (self.rowptr.size - 1, self.cols.max() + 1)
        if len(self.data) != 0:
            if self.rowptr.size - 2 >= bshape[0] or self.cols.max() >= bshape[1]:
                raise ValueError("Matrix has out-of-bounds indices.")
        if bshape[0] < 0 or bshape[1] < 0:
            raise ValueError("Matrix has negative bshape.")
        return tuple(bshape)

    def _validate_symmetry(self, symmetry: str | None) -> str | None:
        """Validates the symmetry of the matrix."""
        if symmetry is None:
            return symmetry
        if symmetry not in ("symmetric", "hermitian"):
            raise ValueError("Invalid symmetry.")
        if self.bshape[0] != self.bshape[1]:
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
            rowptr = np.zeros(self.bshape[0] + 1, dtype=int)
            for row in rows[mask]:
                rowptr[row + 1] += 1
            rowptr = np.cumsum(rowptr)
            self.rowptr = rowptr
            self.cols = self.cols[mask]
            self.data = [b for b, m in zip(self.data, mask) if m]

    def _desymmetrize(self) -> "BCSR":
        """Removes symmetry."""
        if self.symmetry is None:
            return self

        return self.tocoo()._desymmetrize().tocsr()

    def _unsign_index(self, row: int, col: int) -> tuple:
        """Adjusts the sign to allow negative indices and checks bounds."""
        row = self.bshape[0] + row if row < 0 else row
        col = self.bshape[1] + col if col < 0 else col
        if not (0 <= row < self.bshape[0] and 0 <= col < self.bshape[1]):
            raise IndexError("Block index out of bounds.")

        return row, col

    def _expand_rows(self) -> np.ndarray:
        """Expands the row indices."""
        rows = np.zeros(self.bnnz, dtype=int)
        if self.bnnz == 0:
            return rows
        for i in range(self.bshape[0]):
            rows[self.rowptr[i] : self.rowptr[i + 1]] = i
        return rows

    def _get_cols(self, row: int) -> np.ndarray:
        """Returns the column indices for the given row."""
        return self.cols[self.rowptr[row] : self.rowptr[row + 1]]

    def _get_data(self, row: int) -> list:
        """Returns the data values for the given row."""
        return self.data[self.rowptr[row] : self.rowptr[row + 1]]

    def _getitem_symmetry(self, row: int | slice, col: int | slice):
        """Returns the element at the given coordinates."""
        if isinstance(row, Integral) and isinstance(col, Integral):
            row, col = self._unsign_index(row, col)
            if row <= col:
                cols = self._get_cols(row)
                ind = np.nonzero(cols == col)[0]
                if ind.size == 0:
                    return sparse.zeros(
                        (self.row_sizes[row], self.col_sizes[col]), self.dtype
                    )
                return self._get_data(row)[ind[0]]

            if self.symmetry == "symmetric":
                return self[col, row]
            if self.symmetry == "hermitian":
                return np.conj(self[col, row])

        if isinstance(row, Integral):
            row = slice(row, row + 1)
        if isinstance(col, Integral):
            col = slice(col, col + 1)

        rows = np.arange(*row.indices(self.bshape[0]))
        cols = np.arange(*col.indices(self.bshape[1]))
        if len(rows) == 0 or len(cols) == 0:
            raise IndexError("Slice index out of bounds.")
        rows = np.array([self._unsign_index(r, 0)[0] for r in rows])
        cols = np.array([self._unsign_index(0, c)[1] for c in cols])

        row_step = row.step if row.step is not None else 1
        col_step = col.step if col.step is not None else 1

        from bsparse import BCOO

        if row.start == col.start and row.stop == col.stop and row_step == col_step:
            # If the slice is symmetric, we need to return a symmetric matrix.
            mask = np.isin(self._expand_rows(), rows) & np.isin(self.cols, cols)
            submatrix = BCOO(
                (self._expand_rows()[mask] - rows[0]) // row_step,
                (self.cols[mask] - cols[0]) // col_step,
                [b for b, m in zip(self.data, mask) if m],
                bshape=(len(rows), len(cols)),
                dtype=self.dtype,
                symmetry=self.symmetry,
            )
            return submatrix.tocsr()

        submatrix = BCOO([], [], [], bshape=(len(rows), len(cols)), dtype=self.dtype)
        for i, j in np.ndindex(submatrix.bshape):
            value = self[int(rows[i]), int(cols[j])]
            if isinstance(value, (sparse.Sparse, sp.spmatrix)) and value.nnz == 0:
                continue
            if isinstance(value, np.ndarray) and np.all(value == 0):
                continue
            submatrix.rows = np.append(submatrix.rows, (rows[i] - rows[0]) // row_step)
            submatrix.cols = np.append(submatrix.cols, (cols[j] - cols[0]) // col_step)
            submatrix.data.append(value)
        return submatrix.tocsr()

    def _getslice(self, row: slice, col: slice):
        """Returns a submatrix."""
        rows = np.arange(*row.indices(self.bshape[0]))
        cols = np.arange(*col.indices(self.bshape[1]))
        if len(rows) == 0 or len(cols) == 0:
            raise IndexError("Slice index out of bounds.")
        rows = np.array([self._unsign_index(r, 0)[0] for r in rows])
        cols = np.array([self._unsign_index(0, c)[1] for c in cols])
        row_step = row.step if row.step is not None else 1
        col_step = col.step if col.step is not None else 1
        mask = np.isin(self._expand_rows(), rows) & np.isin(self.cols, cols)

        from bsparse import BCOO

        submatrix = BCOO(
            (self._expand_rows()[mask] - rows[0]) // row_step,
            (self.cols[mask] - cols[0]) // col_step,
            [b for b, m in zip(self.data, mask) if m],
            bshape=(len(rows), len(cols)),
            dtype=self.dtype,
        )
        return submatrix.tocsr()

    def __getitem__(
        self, key: int | slice | tuple[int | slice, int | slice]
    ) -> np.ndarray | sparse.Sparse | BSparse:
        """Returns a matrix element or a submatrix."""
        if isinstance(key, (Integral, slice)):
            key = (key, slice(None))

        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("Invalid index")

        row, col = key
        if not isinstance(row, (Integral, slice)) or not isinstance(
            col, (Integral, slice)
        ):
            raise IndexError("Invalid index")

        if self.symmetry is not None:
            return self._getitem_symmetry(row, col)

        if isinstance(row, Integral) and isinstance(col, Integral):
            row, col = self._unsign_index(row, col)
            cols = self._get_cols(row)
            ind = np.nonzero(cols == col)[0]
            if ind.size == 0:
                return sparse.zeros(
                    (self.row_sizes[row], self.col_sizes[col]), self.dtype
                )
            return self._get_data(row)[ind[0]]

        if isinstance(row, Integral):
            row = slice(row, row + 1)
        if isinstance(col, Integral):
            col = slice(col, col + 1)

        return self._getslice(row, col)

    def __setitem__(
        self,
        key: tuple[int, int],
        value: np.ndarray | sparse.Sparse | BSparse,
    ) -> None:
        """Sets a matrix element."""
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("Invalid index")

        if not isinstance(value, (np.ndarray, sparse.Sparse, BSparse)):
            raise ValueError("Invalid value.")

        value = value.astype(self.dtype)

        row, col = key
        if not isinstance(row, int) or not isinstance(col, int):
            raise IndexError("Invalid index")

        row, col = self._unsign_index(row, col)
        if self.symmetry is not None and row > col:
            if self.symmetry == "symmetric":
                self[col, row] = value
                return
            if self.symmetry == "hermitian":
                self[col, row] = value.conjugate()
                return

        all_zero = (
            isinstance(value, (sparse.Sparse, sp.spmatrix))
            and value.nnz == 0
            or isinstance(value, np.ndarray)
            and np.all(value == 0)
        )

        mask = (self._expand_rows() == row) & (self.cols == col)

        if any(mask):
            if all_zero:
                mask = ~mask
                self.rowptr[row + 1 :] -= 1
                self.cols = self.cols[mask]
                self.data = [b for b, m in zip(self.data, mask) if m]
                self._row_sizes[row] = value.shape[0]
                self._col_sizes[col] = value.shape[1]
                return
            ind = np.nonzero(mask)[0][0]
            self.data[ind] = value
            self._row_sizes[row] = value.shape[0]
            self._col_sizes[col] = value.shape[1]
            return

        if all_zero:
            return

        if row not in self._expand_rows():
            self.rowptr = np.insert(self.rowptr, row + 1, self.bnnz)
        self.rowptr[row + 1 :] += 1
        self.cols = np.insert(self.cols, self.rowptr[row + 1] - 1, col)
        self._validate_data(
            self.data[: self.rowptr[row + 1] - 1]
            + [value]
            + self.data[self.rowptr[row + 1] - 1 :]
        )
        self.data.insert(self.rowptr[row + 1] - 1, value)
        self._check_alignment()
        self._row_sizes[row] = value.shape[0]
        self._col_sizes[col] = value.shape[1]

    def __add__(
        self, other: Number | BSparse | np.ndarray | sp.spmatrix
    ) -> "BCSR | np.ndarray":
        """Adds another matrix or a scalar to this matrix."""
        result = self.tocoo() + other
        if isinstance(result, np.ndarray):
            return result
        return result.tocsr()

    def __radd__(
        self, other: Number | BSparse | np.ndarray | sp.spmatrix
    ) -> "BCSR | np.ndarray":
        """Adds this matrix to another matrix or a scalar."""
        return self + other

    def __sub__(
        self, other: Number | BSparse | np.ndarray | sp.spmatrix
    ) -> "BCSR | np.ndarray":
        """Subtracts another matrix or a scalar from this matrix."""
        return self + (-other)

    def __rsub__(
        self, other: Number | BSparse | np.ndarray | sp.spmatrix
    ) -> "BCSR | np.ndarray":
        """Subtracts this matrix from another matrix or a scalar."""
        return other + (-self)

    def __mul__(
        self, other: Number | BSparse | np.ndarray | sp.spmatrix
    ) -> "BCSR | np.ndarray":
        """Multiplies another matrix or a scalar by this matrix."""
        result = self.tocoo() * other
        if isinstance(result, np.ndarray):
            return result
        return result.tocsr()

    def __rmul__(
        self, other: Number | BSparse | np.ndarray | sp.spmatrix
    ) -> "BCSR | np.ndarray":
        """Multiplies this matrix by another matrix or a scalar."""
        return self * other

    def __truediv__(
        self, other: Number | BSparse | np.ndarray | sp.spmatrix
    ) -> "BCSR | np.ndarray":
        """Divides this matrix by another matrix or a scalar."""
        result = self.tocoo() / other
        if isinstance(result, np.ndarray):
            return result
        return result.tocsr()

    def __rtruediv__(
        self, other: Number | BSparse | np.ndarray | sp.spmatrix
    ) -> "BCSR | np.ndarray":
        """Divides another matrix or a scalar by this matrix."""
        result = other / self.tocoo()
        if isinstance(result, np.ndarray):
            return result
        return result.tocsr()

    def __neg__(self) -> "BCSR":
        """Negates this matrix."""
        result = BCSR(
            self.rowptr.copy(),
            self.cols.copy(),
            [-b for b in self.data],
            self.bshape,
            self.dtype,
            (self.row_sizes, self.col_sizes),
            self.symmetry,
        )
        return result

    def _matmul_dense(self, other: np.ndarray) -> np.ndarray:
        """Multiplies this matrix by a dense matrix."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("Incompatible matrix shapes.")

        row_offsets = np.cumsum(self.row_sizes) - self.row_sizes
        col_offsets = np.cumsum(self.col_sizes) - self.col_sizes
        result = np.zeros(
            (self.shape[0], other.shape[1]), dtype=np.result_type(self, other)
        )
        for i in range(self.bshape[0]):
            for j, a in zip(self._get_cols(i), self._get_data(i)):
                result[row_offsets[i] : row_offsets[i] + a.shape[0]] += (
                    a @ other[col_offsets[j] : col_offsets[j] + a.shape[1]]
                )

        return result

    def __matmul__(
        self, other: Number | BSparse | np.ndarray | sp.spmatrix
    ) -> "BCSR | np.ndarray":
        """Multiplies this matrix by another matrix."""
        if isinstance(other, np.ndarray):
            return self._desymmetrize()._matmul_dense(other)
        if isinstance(other, BSparse):
            other = other.tocsr()
        if isinstance(other, sp.spmatrix):
            warn(
                "Automatically inferring block sizes from sparse matrix. "
                "This may result in unexpected behavior. Consider "
                "converting to a `BSparse` matrix."
            )
            other = BCSR.from_spmatrix(other, (self.col_sizes, [1] * other.shape[1]))

        if not isinstance(other, BCSR):
            raise TypeError("Invalid type.")

        if self.bshape[1] != other.bshape[0]:
            raise ValueError("Incompatible matrix shapes.")
        if np.any(self.col_sizes != other.row_sizes):
            raise ValueError("Incompatible block sizes.")

        if self.symmetry is not None or other.symmetry is not None:
            return self._desymmetrize() @ other._desymmetrize()

        result_rowptr = [0]
        result_cols = []
        result_data = []
        for i in range(self.bshape[0]):
            data = []
            cols = []
            for j, a in zip(self._get_cols(i), self._get_data(i)):
                for k, b in zip(other._get_cols(j), other._get_data(j)):
                    if k in cols:
                        data[cols.index(k)] += a @ b
                    else:
                        cols.append(k)
                        data.append(a @ b)
            result_cols.extend(cols)
            result_data.extend(data)
            result_rowptr.append(len(result_cols))

        result = BCSR(
            result_rowptr,
            result_cols,
            result_data,
            (self.bshape[0], other.bshape[1]),
            self.dtype,
            (self.row_sizes, other.col_sizes),
        )
        return result

    def __rmatmul__(
        self, other: Number | BSparse | np.ndarray | sp.spmatrix
    ) -> "BCSR | np.ndarray":
        """Multiplies another matrix by this matrix."""
        if isinstance(other, np.ndarray):
            return other @ self.toarray()
        if isinstance(other, BSparse):
            return other.tocoo() @ self
        if isinstance(other, sp.spmatrix):
            warn(
                "Automatically inferring block sizes from sparse matrix. "
                "This may result in unexpected behavior. Consider "
                "converting to a `BSparse` matrix."
            )
            return (
                BCSR.from_spmatrix(other, ([1] * other.shape[0], self.row_sizes)) @ self
            )

    @property
    def bshape(self) -> tuple[int, int]:
        """The block shape of the matrix."""
        return self._bshape

    @property
    def row_sizes(self) -> np.ndarray:
        """The sizes of the row elements."""
        if hasattr(self, "_row_sizes"):
            return self._row_sizes

        sizes = np.zeros(self.bshape[0], dtype=int)
        rows = self._expand_rows()
        for row in range(self.bshape[0]):
            row_data = [b for b, r in zip(self.data, rows) if r == row]
            if len(row_data) == 0:
                sizes[row] = 1
                continue
            sizes[row] = row_data[0].shape[0]

        return sizes

    @property
    def col_sizes(self) -> np.ndarray:
        """The sizes of the column elements."""
        if hasattr(self, "_col_sizes"):
            return self._col_sizes

        sizes = np.zeros(self.bshape[1], dtype=int)
        for col in range(self.bshape[1]):
            col_data = [b for b, c in zip(self.data, self.cols) if c == col]
            if len(col_data) == 0:
                sizes[col] = 1
                continue
            sizes[col] = col_data[0].shape[1]

        return sizes

    @property
    def dtype(self) -> np.dtype:
        """The data type of the matrix elements."""
        return self._dtype

    @property
    def bnnz(self) -> int:
        """The number of non-zero elements in the matrix."""
        return len(self.data)

    @property
    def nnz(self) -> int:
        """The number of non-zero elements in the matrix."""
        return sum([b.size if hasattr(b, "size") else b.nnz for b in self.data])

    @property
    def symmetry(self) -> str | None:
        """The symmetry of the matrix."""
        return self._symmetry

    @property
    def T(self) -> "BCSR":
        """The transpose of the matrix."""
        if self.symmetry == "symmetric":
            return self
        if self.symmetry == "hermitian":
            return self.conjugate()

        from bsparse import BCOO

        transpose = BCOO(
            self.cols,
            self._expand_rows(),
            [b.T for b in self.data],
            (self.bshape[1], self.bshape[0]),
            self.dtype,
            (self.col_sizes, self.row_sizes),
            self.symmetry,
        )
        return transpose.tocsr()

    @property
    def H(self) -> "BCSR":
        """The conjugate transpose of the matrix."""
        if self.symmetry == "hermitian":
            return self
        if self.symmetry == "symmetric":
            return self.conjugate()

        from bsparse import BCOO

        hermitian = BCOO(
            self.cols,
            self._expand_rows(),
            [b.conjugate().T for b in self.data],
            (self.bshape[1], self.bshape[0]),
            self.dtype,
            (self.col_sizes, self.row_sizes),
            self.symmetry,
        )
        return hermitian.tocsr()

    def conjugate(self) -> "BCSR":
        """The complex conjugate of the matrix."""
        conjugate = BCSR(
            self.rowptr,
            self.cols,
            [b.conjugate() for b in self.data],
            self.bshape,
            self.dtype,
            (self.row_sizes, self.col_sizes),
            self.symmetry,
        )
        return conjugate

    def diagonal(self, offset: int = 0) -> np.ndarray:
        """Returns the diagonal of the matrix."""
        if not -self.bshape[0] < offset < self.bshape[1]:
            raise ValueError("Offset out of bounds.")

        if offset < 0:
            if self.symmetry == "hermitian":
                return [b.conjugate().T for b in self.diagonal(-offset)]
            if self.symmetry == "symmetric":
                return [b.T for b in self.diagonal(-offset)]

        start = (-offset) * self.bshape[1] if offset < 0 else offset

        rows = []
        cols = []
        for flat_ind in range(
            start, self.bshape[0] * self.bshape[1], self.bshape[1] + 1
        ):
            if flat_ind // self.bshape[1] >= self.bshape[1] - offset:
                break
            rows.append(flat_ind // self.bshape[1])
            cols.append(flat_ind % self.bshape[1])

        diag = []
        for row, col in zip(rows, cols):
            ind = np.nonzero((self._expand_rows() == row) & (self.cols == col))[0]
            if ind.size == 0:
                diag.append(
                    sparse.zeros((self.row_sizes[row], self.col_sizes[col]), self.dtype)
                )
                continue
            diag.append(self.data[ind[0]])
        return diag

    def copy(self) -> "BCSR":
        """Returns a copy of the matrix."""
        return BCSR(
            self.rowptr.copy(),
            self.cols.copy(),
            self.data.copy(),
            self.bshape,
            self.dtype,
            (self.row_sizes, self.col_sizes),
            self.symmetry,
        )

    def astype(self, dtype: np.dtype) -> "BCSR":
        """Returns a copy of the matrix with a different data type."""
        new = BCSR(
            self.rowptr.copy(),
            self.cols.copy(),
            self.data.copy(),
            self.bshape,
            dtype,
            (self.row_sizes, self.col_sizes),
            self.symmetry,
        )
        return new

    def toarray(self) -> np.ndarray:
        """Converts the matrix to a dense `numpy.ndarray`."""
        arr = np.zeros(self.shape, dtype=self.dtype)
        row_offsets = np.cumsum(self.row_sizes) - self.row_sizes
        col_offsets = np.cumsum(self.col_sizes) - self.col_sizes

        for row, col, b in zip(self._expand_rows(), self.cols, self.data):
            arr[
                row_offsets[row] : row_offsets[row] + b.shape[0],
                col_offsets[col] : col_offsets[col] + b.shape[1],
            ] = (
                b if isinstance(b, np.ndarray) else b.toarray()
            )

        if self.symmetry == "symmetric":
            temp = arr.T.copy()
            temp[np.nonzero(arr)] = 0
            arr += temp
        if self.symmetry == "hermitian":
            temp = arr.conj().T.copy()
            temp[np.nonzero(arr)] = 0
            arr += temp

        return arr

    def tocoo(self) -> "BSparse":
        """Converts the matrix to `BCOO` format."""

        from bsparse.bcoo import BCOO

        bcoo = BCOO(
            self._expand_rows(),
            self.cols,
            self.data,
            self.bshape,
            self.dtype,
            (self.row_sizes, self.col_sizes),
            self.symmetry,
        )
        return bcoo

    def tocsr(self) -> "BCSR":
        """Converts the matrix to `BCSR` format."""
        return self

    def todia(self) -> "BSparse":
        """Converts the matrix to `BDIA` format."""
        return self.tocoo().todia()

    def save_npz(self, filename: str) -> None:
        """Saves the matrix as ``.npz`` archive."""
        np.savez_compressed(
            filename,
            format="bcsr",
            rowptr=self.rowptr,
            cols=self.cols,
            data=self.data,
            bshape=self.bshape,
            dtype=self.dtype,
            row_sizes=self.row_sizes,
            col_sizes=self.col_sizes,
            symmetry=self.symmetry,
        )

    @classmethod
    def from_array(
        cls,
        arr: np.ndarray,
        sizes: tuple[ArrayLike, ArrayLike],
        symmetry: str | None = None,
    ) -> "BCSR":
        """Creates a sparse matrix from a dense `numpy.ndarray`."""
        arr = np.asarray(arr)
        row_sizes, col_sizes = sizes
        if arr.shape != (np.sum(row_sizes), np.sum(col_sizes)):
            raise ValueError("Array shape does not match block sizes.")
        row_offsets = np.cumsum(row_sizes) - row_sizes
        col_offsets = np.cumsum(col_sizes) - col_sizes

        rows = []
        cols = []
        data = []
        for i, rr in enumerate(row_offsets):
            for j, cc in enumerate(col_offsets):
                b = arr[rr : rr + row_sizes[i], cc : cc + col_sizes[j]]
                if np.all(b == 0):
                    continue
                rows.append(i)
                cols.append(j)
                data.append(b)

        bshape = (len(row_sizes), len(col_sizes))

        rowptr = np.zeros(bshape[0] + 1, dtype=int)
        for row in rows:
            rowptr[row + 1] += 1
        rowptr = np.cumsum(rowptr)
        return cls(rowptr, cols, data, bshape, arr.dtype, sizes, symmetry)

    @classmethod
    def from_spmatrix(
        cls,
        mat: sp.spmatrix,
        sizes: tuple[ArrayLike, ArrayLike],
        symmetry: str | None = None,
    ) -> "BCSR":
        """Creates a sparse matrix from a `scipy.sparse.spmatrix`."""
        mat = sp.lil_array(mat)
        row_sizes, col_sizes = sizes
        if mat.shape != (np.sum(row_sizes), np.sum(col_sizes)):
            raise ValueError("Matrix shape does not match block sizes.")
        row_offsets = np.cumsum(row_sizes) - row_sizes
        col_offsets = np.cumsum(col_sizes) - col_sizes

        rows = []
        cols = []
        data = []
        for i, rr in enumerate(row_offsets):
            for j, cc in enumerate(col_offsets):
                b = mat[rr : rr + row_sizes[i], cc : cc + col_sizes[j]]
                if b.nnz == 0:
                    continue
                rows.append(i)
                cols.append(j)
                data.append(b)

        bshape = (len(row_sizes), len(col_sizes))

        rowptr = np.zeros(bshape[0] + 1, dtype=int)
        for row in rows:
            rowptr[row + 1] += 1
        rowptr = np.cumsum(rowptr)
        return cls(rowptr, cols, data, bshape, mat.dtype, sizes, symmetry)
