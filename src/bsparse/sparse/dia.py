from numbers import Number
from warnings import warn

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike

from bsparse.sparse.sparse import Sparse


class DIA(Sparse):
    """A sparse matrix in DIAgonal format.

    The ``DIA`` class represents a sparse matrix using two arrays:

    * ``offsets``: the offsets of the diagonals, where zero indicates
      the main diagonal, positive values are above the main diagonal,
      and negative values are below the main diagonal.
    * ``data``: the values of the diagonals, padded with zeros so that
      each diagonal has the same length.

    Parameters
    ----------
    offsets : array_like
        The offsets of the diagonals.
    data : array_like
        The values of the diagonals.
    shape : tuple[int, int], optional
        The shape of the matrix. If not given, it is inferred from the
        offsets and data.
    dtype : dtype, optional
        The data type of the matrix. If not given, it is inferred from
        the data.
    symmetry : str, optional
        The symmetry of the matrix. If not given, no symmetry is
        assumed. This is only applicable for square matrices, where
        possible values are ``'symmetric'`` and ``'hermitian'``. Note
        that when setting a symmetry, the lower triangular part of the
        matrix is discarded.

    """

    def __init__(
        self,
        offsets: ArrayLike,
        data: ArrayLike,
        shape: tuple[int, int] | None = None,
        dtype: np.dtype | None = None,
        symmetry: str | None = None,
    ) -> None:
        """Initialize a sparse matrix."""
        self.offsets = np.asarray(offsets, dtype=int)
        self.data = np.asarray(data, dtype=dtype)

        if self.data.shape[0] != len(self.offsets):
            raise ValueError("Offsets and data must have the same length.")

        self._dtype = self.data.dtype
        self._shape = self._validate_shape(shape)
        self._symmetry = self._validate_symmetry(symmetry)

        self._sort_diagonals()

    def _validate_shape(self, shape: tuple[int, int] | None) -> tuple[int, int]:
        """Validate the shape of the matrix."""
        if shape is None:
            if self.data.size == 0:
                raise ValueError("Cannot instantiate empty matrix without shape.")
            return (self.data.shape[1], self.data.shape[1])
        if self.data.size != 0:
            if self.offsets.min() <= -shape[0] or self.offsets.max() >= shape[1]:
                raise ValueError("Offsets exceed shape.")
        if shape[0] < 0 or shape[1] < 0:
            raise ValueError("Matrix has negative shape.")
        return shape

    def _validate_symmetry(self, symmetry: str | None) -> str | None:
        """Validate the symmetry of the matrix."""
        if symmetry is None:
            return symmetry
        if symmetry not in ("symmetric", "hermitian"):
            raise ValueError("Invalid symmetry.")
        if self.shape[0] != self.shape[1]:
            raise ValueError("Symmetry is only applicable for square matrices.")

        self._discard_subdiagonal()

        return symmetry

    def _discard_subdiagonal(self) -> None:
        if any(self.offsets < 0):
            warn(
                "Symmetric matrix is not upper triangular. "
                "Lower triangular part is discarded."
            )
            mask = self.offsets >= 0
            self.offsets = self.offsets[mask]
            self.data = self.data[mask]

    def _desymmetrize(self) -> "DIA":
        """Remove symmetry from the matrix."""
        if self.symmetry is None:
            return self.copy()

        mask = self.offsets > 0
        if not np.any(mask):
            return DIA(self.offsets.copy(), self.data.copy(), self.shape, self.dtype)

        lower_offsets = -self.offsets[mask]
        lower_data = (
            self.data[mask] if self.symmetry == "symmetric" else self.data[mask].conj()
        )
        lower_data = np.array(
            [np.roll(diag, offset) for offset, diag in zip(lower_offsets, lower_data)],
            ndmin=2,
        )

        offsets = np.concatenate((self.offsets, lower_offsets))
        data = np.concatenate((self.data, lower_data))
        return DIA(offsets, data, self.shape, self.dtype)

    def _sort_diagonals(self) -> None:
        """Sort the diagonals by offset."""
        ind = np.argsort(self.offsets)
        self.offsets = self.offsets[ind]
        self.data = self.data[ind]

    def _unsign_index(self, row: int, col: int) -> tuple:
        """Adjusts the sign to allow negative indices and checks bounds."""
        row = self.shape[0] + row if row < 0 else row
        col = self.shape[1] + col if col < 0 else col
        if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
            raise IndexError("Index out of bounds.")

        return row, col

    def _getitem_symmetry(self, row: int | slice, col: int | slice):
        """Returns the element at the given coordinates."""
        if isinstance(row, int) and isinstance(col, int):
            row, col = self._unsign_index(row, col)
            if row <= col:
                offset = col - row
                if offset not in self.offsets:
                    return self.dtype.type(0)
                data = self.data[self.offsets == offset][0]
                return data[col]

            if self.symmetry == "symmetric":
                return self[col, row]
            if self.symmetry == "hermitian":
                return np.conj(self[col, row])

        if isinstance(row, int):
            row = slice(row, row + 1)
        if isinstance(col, int):
            col = slice(col, col + 1)

        coo = self.tocoo()
        coo = coo[row, col]
        return coo.todia()

    def _getslice(self, row: slice, col: slice):
        """Returns a submatrix."""
        coo = self.tocoo()
        coo = coo[row, col]
        return coo.todia()

    def __getitem__(
        self, key: int | slice | tuple[int | slice, int | slice]
    ) -> "Number | DIA":
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
            offset = col - row
            if offset not in self.offsets:
                return self.dtype.type(0)
            data = self.data[self.offsets == offset][0]
            return data[col]

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

        offset = col - row
        if offset in self.offsets:
            self.data[self.offsets == offset, col] = value
            return

        if value == 0:
            return

        self.offsets = np.append(self.offsets, offset)
        data = np.zeros(self.shape[1], dtype=self.dtype)
        data[col] = value
        if self.data.size == 0:
            self.data = np.array(data, ndmin=2)
            return
        self.data = np.vstack((self.data, data))

    def __add__(
        self, other: Number | Sparse | np.ndarray | sp.sparray
    ) -> "DIA | np.ndarray":
        """Adds another matrix or a scalar to this matrix."""
        if isinstance(other, Number):
            if other == 0:
                return self.copy()
            raise NotImplementedError
        if isinstance(other, np.ndarray):
            return self.toarray() + other
        if isinstance(other, Sparse):
            other = other.todia()
        if sp.issparse(other):
            other = DIA.from_sparray(other)

        if not isinstance(other, DIA):
            raise TypeError("Invalid type.")

        if self.shape != other.shape:
            raise ValueError("Incompatible matrix shapes.")

        if self.symmetry != other.symmetry:
            return self._desymmetrize() + other._desymmetrize()

        offsets = np.unique(np.concatenate((self.offsets, other.offsets)))
        data = np.zeros(
            (len(offsets), self.shape[1]), dtype=np.result_type(self, other)
        )
        for offset in offsets:
            if offset in self.offsets:
                data[offsets == offset] += self.data[self.offsets == offset]
            if offset in other.offsets:
                data[offsets == offset] += other.data[other.offsets == offset]

        return DIA(
            offsets, data, self.shape, np.result_type(self, other), self.symmetry
        )

    def __radd__(
        self, other: Number | Sparse | np.ndarray | sp.sparray
    ) -> "DIA | np.ndarray":
        """Adds this matrix to another matrix or a scalar."""
        return self + other

    def __sub__(
        self, other: Number | Sparse | np.ndarray | sp.sparray
    ) -> "DIA | np.ndarray":
        """Subtracts another matrix or a scalar from this matrix."""
        return self + (-other)

    def __rsub__(
        self, other: Number | Sparse | np.ndarray | sp.sparray
    ) -> "DIA | np.ndarray":
        """Subtracts this matrix from another matrix or a scalar."""
        return (-self) + other

    def __mul__(
        self, other: Number | Sparse | np.ndarray | sp.sparray
    ) -> "DIA | np.ndarray":
        """Multiplies another matrix or a scalar by this matrix."""
        if isinstance(other, Number):
            if self.symmetry == "hermitian" and np.iscomplexobj(other):
                return self._desymmetrize() * other
            result = DIA(
                self.offsets.copy(),
                self.data.copy() * other,
                self.shape,
                np.result_type(self, other),
                self.symmetry,
            )
            return result
        if isinstance(other, np.ndarray):
            return self.toarray() * other
        if isinstance(other, Sparse):
            other = other.todia()
        if sp.issparse(other):
            other = DIA.from_sparray(other)

        if not isinstance(other, DIA):
            raise TypeError("Invalid type.")

        if self.shape != other.shape:
            raise ValueError("Incompatible matrix shapes.")

        if self.symmetry != other.symmetry:
            return self._desymmetrize() * other._desymmetrize()

        offsets = np.intersect1d(self.offsets, other.offsets)
        data = np.zeros(
            (len(offsets), self.shape[1]), dtype=np.result_type(self, other)
        )
        for i, offset in enumerate(offsets):
            data[i] = (
                self.data[self.offsets == offset] * other.data[other.offsets == offset]
            )

        return DIA(
            offsets, data, self.shape, np.result_type(self, other), self.symmetry
        )

    def __rmul__(
        self, other: Number | Sparse | np.ndarray | sp.sparray
    ) -> "DIA | np.ndarray":
        """Multiplies this matrix by another matrix or a scalar."""
        return self * other

    def __truediv__(
        self, other: Number | Sparse | np.ndarray | sp.sparray
    ) -> "DIA | np.ndarray":
        """Divides this matrix by another matrix or a scalar."""
        if isinstance(other, Number):
            if self.symmetry == "hermitian" and np.iscomplexobj(other):
                self = self._desymmetrize()
            result = DIA(
                self.offsets.copy(),
                self.data.copy() / other,
                self.shape,
                np.result_type(self, other),
                self.symmetry,
            )
            return result
        if isinstance(other, Sparse) or sp.issparse(other):
            other = other.toarray()

        if not isinstance(other, np.ndarray):
            raise TypeError("Invalid type.")

        return self.toarray() / other

    def __rtruediv__(
        self, other: Number | Sparse | np.ndarray | sp.sparray
    ) -> "DIA | np.ndarray":
        """Divides another matrix or a scalar by this matrix."""
        if isinstance(other, Number):
            return other / self.toarray()
        if isinstance(other, Sparse) or sp.issparse(other):
            other = other.toarray()

        if not isinstance(other, np.ndarray):
            raise TypeError("Invalid type.")

        return other / self.toarray()

    def __neg__(self) -> "DIA":
        """Negates this matrix."""
        result = DIA(
            self.offsets.copy(),
            -self.data.copy(),
            self.shape,
            self.dtype,
            self.symmetry,
        )
        return result

    def __matmul__(
        self, other: Number | Sparse | np.ndarray | sp.sparray
    ) -> "DIA | np.ndarray":
        """Multiplies this matrix by another matrix."""
        if isinstance(other, np.ndarray):
            return self.toarray() @ other
        if isinstance(other, Sparse):
            other = other.todia()
        if sp.issparse(other):
            other = DIA.from_sparray(other)

        if not isinstance(other, DIA):
            raise TypeError("Invalid type.")

        if self.shape[1] != other.shape[0]:
            raise ValueError("Incompatible matrix shapes.")

        if self.symmetry is not None or other.symmetry is not None:
            # Products of symmetric matrices are not necessarily symmetric.
            return self._desymmetrize() @ other._desymmetrize()

        offsets = sorted(
            set(
                i + j
                for i in self.offsets
                for j in other.offsets
                if -self.shape[0] < i + j < other.shape[1]
            )
        )
        data = np.zeros(
            (len(offsets), other.shape[1]), dtype=np.result_type(self, other)
        )

        shape_diff = other.shape[1] - self.shape[1]

        for i in self.offsets:
            for j in other.offsets:
                if i + j not in offsets:
                    continue
                self_ind = np.where(self.offsets == i)[0][0]
                other_ind = np.where(other.offsets == j)[0][0]
                if shape_diff < 0:
                    self_data = np.roll(self.data[self_ind], j)[:shape_diff]
                else:
                    self_data = np.roll(
                        np.concatenate(
                            (self.data[self_ind], np.zeros(shape_diff, data.dtype))
                        ),
                        j,
                    )
                data[offsets.index(i + j)] += self_data * other.data[other_ind]

        return DIA(
            offsets,
            data,
            (self.shape[0], other.shape[1]),
            np.result_type(self, other),
            None,
        )

    def __rmatmul__(
        self, other: Number | Sparse | np.ndarray | sp.sparray
    ) -> "DIA | np.ndarray":
        """Multiplies another matrix by this matrix."""
        if isinstance(other, np.ndarray):
            return other @ self.toarray()
        if isinstance(other, Sparse):
            return other.todia() @ self
        if sp.issparse(other):
            return DIA.from_sparray(other) @ self

    @property
    def shape(self) -> tuple[int, int]:
        """The shape of the matrix."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """The data type of the matrix elements."""
        return self._dtype

    @property
    def nnz(self) -> int:
        """The number of stored elements in the matrix."""
        return self.data.size

    @property
    def symmetry(self) -> str | None:
        """The symmetry of the matrix."""
        return self._symmetry

    @property
    def T(self) -> "DIA":
        """The transpose of the matrix."""

        if self.symmetry == "symmetric":
            return self
        if self.symmetry == "hermitian":
            return self.conjugate()

        offsets = -self.offsets
        rr = np.arange(len(offsets))[:, None]
        cc = np.arange(self.shape[0]) - (offsets % max(self.shape))[:, None]
        padding = max(0, max(self.shape) - self.data.shape[1])
        data = np.hstack(
            (self.data, np.zeros((self.data.shape[0], padding), dtype=self.dtype))
        )
        data = data[rr, cc]

        transpose = DIA(
            offsets,
            data,
            (self.shape[1], self.shape[0]),
            self.dtype,
            self.symmetry,
        )
        return transpose

    @property
    def H(self) -> "DIA":
        """The conjugate transpose of the matrix."""
        if self.symmetry == "hermitian":
            return self
        if self.symmetry == "symmetric":
            return self.conjugate()

        return self.conjugate().T

    def conjugate(self) -> "DIA":
        """The complex conjugate of the matrix."""
        conjugate = DIA(
            self.offsets,
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

        rr, __ = self._get_diag_indices(offset)

        if offset not in self.offsets:
            return np.zeros(len(rr), dtype=self.dtype)

        diag = self.data[self.offsets == offset][0]
        if offset < 0:
            return diag[: len(rr)]
        return diag[offset : offset + len(rr)]

    def copy(self) -> "DIA":
        """Returns a copy of the matrix."""
        new = DIA(
            self.offsets.copy(),
            self.data.copy(),
            self.shape,
            self.dtype,
            self.symmetry,
        )
        return new

    def astype(self, dtype: np.dtype) -> "DIA":
        """Returns a copy of the matrix with a different data type."""
        new = DIA(
            self.offsets.copy(),
            self.data.copy(),
            self.shape,
            dtype,
            self.symmetry,
        )
        return new

    def _get_diag_indices(self, offset: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns the indices of the k-th diagonal."""
        start = (-offset) * self.shape[1] if offset < 0 else offset

        rows = []
        cols = []
        for flat_ind in range(start, self.shape[0] * self.shape[1], self.shape[1] + 1):
            if flat_ind // self.shape[1] >= self.shape[1] - offset:
                break
            rows.append(flat_ind // self.shape[1])
            cols.append(flat_ind % self.shape[1])

        return np.array(rows, dtype=int), np.array(cols, dtype=int)

    def toarray(self) -> np.ndarray:
        """Returns a dense matrix."""
        arr = np.zeros(self.shape, dtype=self.dtype)
        for offset, diag in zip(self.offsets, self.data):
            rr, cc = self._get_diag_indices(offset)
            data = diag[: len(rr)] if offset < 0 else diag[offset : offset + len(rr)]
            arr[rr, cc] = data

        if self.symmetry == "symmetric":
            arr = arr + arr.T - np.diag(arr.diagonal())
        if self.symmetry == "hermitian":
            arr = arr + arr.conj().T - np.diag(arr.diagonal())
        return arr

    def tocoo(self) -> "Sparse":
        """Returns a COOrdinate representation of the matrix."""
        from bsparse.sparse.coo import COO

        rows = []
        cols = []
        data = []
        for offset, diag in zip(self.offsets, self.data):
            rr, cc = self._get_diag_indices(offset)
            diag = diag[: len(rr)] if offset < 0 else diag[offset : offset + len(rr)]
            mask = np.nonzero(diag)[0]
            rows.append(rr[mask])
            cols.append(cc[mask])
            data.append(diag[mask])

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)

        return COO(rows, cols, data, self.shape, self.dtype, self.symmetry)

    def tocsr(self) -> "Sparse":
        """Returns a Compressed Sparse Row representation of the matrix."""
        return self.tocoo().tocsr()

    def todia(self) -> "Sparse":
        """Returns a DIAgonal representation of the matrix."""
        return self

    def save_npz(self, filename: str) -> None:
        """Saves the matrix as ``.npz`` archive."""
        np.savez_compressed(
            filename,
            format="dia",
            offsets=self.offsets,
            data=self.data,
            shape=self.shape,
            dtype=self.dtype,
            symmetry=self.symmetry,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray, symmetry: str | None = None) -> "DIA":
        """Creates a sparse matrix from a dense `numpy.ndarray`."""
        arr = np.asarray(arr)
        rows, cols = arr.nonzero()
        coo_data = arr[rows, cols]

        offsets, offset_indices = np.unique(cols - rows, return_inverse=True)

        if len(coo_data) == 0:
            data = np.zeros((0, 0), dtype=arr.dtype)
            return cls(offsets, data, arr.shape, symmetry=symmetry)

        data = np.zeros((len(offsets), arr.shape[1]), dtype=arr.dtype)
        data[offset_indices, cols] = coo_data

        return cls(offsets, data, arr.shape, symmetry=symmetry)

    @classmethod
    def from_sparray(cls, mat: sp.sparray, symmetry: str | None = None) -> "DIA":
        """Creates a sparse matrix from a `scipy.sparse.sparray`."""
        from bsparse.sparse.coo import COO

        mat = COO.from_sparray(mat, symmetry=symmetry)
        return mat.todia()
