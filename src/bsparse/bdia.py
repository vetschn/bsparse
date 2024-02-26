from numbers import Integral, Number
from warnings import warn

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike

from bsparse import sparse
from bsparse.bsparse import BSparse


class BDIA(BSparse):
    """A sparse matrix in DIAgonal format.

    The `BDIA` class represents a sparse matrix using two arrays:

    * `offsets`: the offsets of the diagonals, where zero indicates
      the main diagonal, positive values are above the main diagonal,
      and negative values are below the main diagonal.
    * `data`: the values of the diagonals, *not* padded with zeros.
      Diagonals have varying lengths.

    .. figure:: ../figures/bdia.jpg
        :scale: 25%

    Parameters
    ----------
    offsets : array_like
        The offsets of the block diagonals.
    data : array_like
        The values of the block diagonals.
    bshape : tuple[int, int], optional
        The shape of the matrix. If not given, it is inferred from the
        offsets and data.
    dtype : dtype, optional
        The data type of the matrix. If not given, it is inferred from
        the data.
    sizes : tuple[array_like, array_like], optional
        The sizes of the row and column elements. If not given, they
        are inferred from the data.
    symmetry : str, optional
        The symmetry of the matrix. If not given, no symmetry is
        assumed. This is only applicable for square matrices, where
        possible values are ``'symmetric'`` and ``'hermitian'``. Note
        that when setting a symmetry, the lower triangular part of the
        matrix is discarded.

    Attributes
    ----------
    offsets : ndarray
        The offsets of the block diagonals.
    data : list of lists of ndarray, scipy.sparse, ...
        The values of the block diagonals. The inner lists represent
        the block diagonals, and the inner arrays represent the blocks.
    shape : tuple[int, int]
        The shape of the matrix.
    bshape : tuple[int, int]
        The block shape of the matrix.
    row_sizes : ndarray
        The sizes of the row elements.
    col_sizes : ndarray
        The sizes of the column elements.
    dtype : dtype
        The data type of the matrix.
    symmetry : str
        The symmetry of the matrix.
    nnz : int
        The number of stored elements in the matrix.
    bnnz : int
        The number of stored blocks in the matrix.
    T : BDIA
        The transpose of the matrix.
    H : BDIA
        The conjugate transpose of the matrix.

    Examples
    --------
    >>> import numpy as np
    >>> offsets = [0, 1, -1]
    >>> data = [
    ...     [np.array([[1, 0], [0, 2]]), np.array([[3, 0], [0, 4]])],
    ...     [np.array([[5, 0], [0, 6]])],
    ...     [np.array([[7, 0], [0, 8]])],
    ... ]
    >>> bdia = BDIA(offsets, data)
    >>> bdia
    BDIA(bshape=(2, 2), bnnz=4 | shape=(4, 4), nnz=16)
    >>> bdia.toarray()
    array([[1, 0, 5, 0],
           [0, 2, 0, 6],
           [7, 0, 3, 0],
           [0, 8, 0, 4]])

    """

    def __init__(
        self,
        offsets: ArrayLike,
        data: ArrayLike,
        bshape: tuple[int, int] | None = None,
        dtype: np.dtype | None = None,
        sizes: tuple[ArrayLike, ArrayLike] | None = None,
        symmetry: str | None = None,
    ) -> None:
        """Initializes a ``BDIA`` matrix."""
        self.offsets = np.asarray(offsets, dtype=int)

        data = [list(bdiag) for bdiag in data]
        self.data = self._validate_data(data)

        if dtype is None and len(self.data) != 0:
            dtype = np.result_type(*[b.dtype for bdiag in self.data for b in bdiag])
        if dtype is None:
            dtype = np.dtype(float)
        self.data = [[b.astype(dtype) for b in bdiag] for bdiag in self.data]
        self._dtype = dtype

        self._bshape = self._validate_bshape(bshape)
        self._symmetry = self._validate_symmetry(symmetry)
        self._sort_diagonals()

        if sizes is not None:
            self._row_sizes = np.asarray(sizes[0], dtype=int)
            self._col_sizes = np.asarray(sizes[1], dtype=int)
        else:
            self._row_sizes = self.row_sizes
            self._col_sizes = self.col_sizes

    def _validate_data(self, data: ArrayLike) -> list:
        """Validates the data blocks of the matrix."""
        # Check that the matrix blocks allow an equivalent array representation.
        for bdiag in data:
            for b in bdiag:
                if b.ndim != 2:
                    raise ValueError("Matrix blocks must be two dimensional.")
                if not hasattr(b, "shape"):
                    raise ValueError("Matrix blocks must have a `shape` attribute.")
                if not hasattr(b, "dtype"):
                    raise ValueError("Matrix blocks must have a `dtype` attribute.")
                if not hasattr(b, "astype"):
                    raise ValueError("Matrix blocks must implement an `astype` method.")

        max_dim = max(len(data), max(len(bdiag) for bdiag in data))
        for row in range(max_dim):
            row_data = []
            for offset, bdiag in zip(self.offsets, data):
                if 0 > row + offset or row + min(0, offset) >= len(bdiag):
                    continue
                row_data.append(bdiag[row + min(0, offset)])
            if len(row_data) == 0:
                continue
            if len(set([b.shape[0] for b in row_data])) != 1:
                raise ValueError("Matrix rows are not aligned.")

        for col in range(max_dim):
            col_data = []
            for offset, bdiag in zip(self.offsets, data):
                if 0 > col - offset or col - max(0, offset) >= len(bdiag):
                    continue
                col_data.append(bdiag[col - max(0, offset)])
            if len(col_data) == 0:
                continue
            if len(set([b.shape[1] for b in col_data])) != 1:
                raise ValueError("Matrix columns are not aligned.")

        return data

    def _validate_bshape(self, bshape: tuple[int, int] | None) -> tuple[int, int]:
        """Validate the bshape of the matrix."""
        if bshape is None:
            if len(self.data[0]) == 1:
                raise ValueError("Cannot instantiate empty matrix without bshape.")
            max_dim = max(len(bdiag) for bdiag in self.data)
            return (max_dim, max_dim)
        if len(self.data[0]) != 0:
            if self.offsets.min() <= -bshape[0] or self.offsets.max() >= bshape[1]:
                raise ValueError("Offsets exceed bshape.")
        if bshape[0] < 0 or bshape[1] < 0:
            raise ValueError("Matrix has negative bshape.")
        return bshape

    def _validate_symmetry(self, symmetry: str | None) -> str | None:
        """Validate the symmetry of the matrix."""
        if symmetry is None:
            return symmetry
        if symmetry not in ("symmetric", "hermitian"):
            raise ValueError("Invalid symmetry.")
        if self.bshape[0] != self.bshape[1]:
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
            self.data = [bdiag for bdiag, m in zip(self.data, mask) if m]

    def _desymmetrize(self) -> "BDIA":
        """Remove symmetry from the matrix."""
        if self.symmetry is None:
            return self

        mask = self.offsets > 0
        if not np.any(mask):
            return BDIA(
                self.offsets.copy(),
                self.data.copy(),
                self.bshape,
                self.dtype,
                (self.row_sizes, self.col_sizes),
                None,
            )
        lower_offsets = -self.offsets[mask]
        lower_data = [
            [b.T for b in bdiag]
            for bdiag in [self.data[i] for i, m in enumerate(mask) if m]
        ]
        if self.symmetry == "hermitian":
            lower_data = [[b.conjugate() for b in bdiag] for bdiag in lower_data]

        return BDIA(
            np.concatenate((self.offsets, lower_offsets)),
            self.data + lower_data,
            self.bshape,
            self.dtype,
            (self.row_sizes, self.col_sizes),
            None,
        )

    def _sort_diagonals(self) -> None:
        """Sort the diagonals by offset."""
        ind = np.argsort(self.offsets)
        self.offsets = self.offsets[ind]
        self.data = [self.data[i] for i in ind]
        if len(self.data) == 0:
            self.data = [[]]

    def _unsign_index(self, row: int, col: int) -> tuple:
        """Adjusts the sign to allow negative indices and checks bounds."""
        row = self.bshape[0] + row if row < 0 else row
        col = self.bshape[1] + col if col < 0 else col
        if not (0 <= row < self.bshape[0] and 0 <= col < self.bshape[1]):
            raise IndexError("Index out of bounds.")

        return row, col

    def _getitem_symmetry(self, row: int | slice, col: int | slice):
        """Returns the element at the given coordinates."""
        if isinstance(row, Integral) and isinstance(col, Integral):
            row, col = self._unsign_index(row, col)
            if row <= col:
                ind = np.nonzero(self.offsets == col - row)[0]
                if ind.size == 0:
                    return sparse.zeros(
                        (self.row_sizes[row], self.col_sizes[col]), self.dtype
                    )
                return self.data[ind[0]][min(row, col)]

            if self.symmetry == "symmetric":
                return self[col, row]
            if self.symmetry == "hermitian":
                return self[col, row].conjugate()

        if isinstance(row, Integral):
            row = slice(row, row + 1)
        if isinstance(col, Integral):
            col = slice(col, col + 1)

        # Slicing these hurts my brain, so I'm just going to do it the
        # dumb way for now.
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
            ind = np.nonzero(self.offsets == col - row)[0]
            if ind.size == 0:
                return sparse.zeros(
                    (self.row_sizes[row], self.col_sizes[col]), self.dtype
                )
            return self.data[ind[0]][min(row, col)]

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

        offset = col - row
        if offset in self.offsets:
            ind = np.nonzero(self.offsets == offset)[0][0]
            data = self.data.copy()
            data[int(ind)][min(row, col)] = value
            self.data = self._validate_data(data)
            return

        all_zero = (
            (isinstance(value, sparse.Sparse) or sp.issparse(value))
            and value.nnz == 0
            or isinstance(value, np.ndarray)
            and np.all(value == 0)
        )

        if all_zero:
            return

        bdiag = self.diagonal(offset)
        bdiag[min(row, col)] = value
        self.offsets = np.append(self.offsets, offset)
        self._row_sizes[row] = value.shape[0]
        self._col_sizes[col] = value.shape[1]
        if len(self.data[0]) == 0:
            self.data = [bdiag]
            return
        self.data = self._validate_data(self.data + [bdiag])
        self._sort_diagonals()

    def __add__(
        self, other: Number | BSparse | np.ndarray | sp.sparray
    ) -> "BDIA | np.ndarray":
        """Adds another matrix or a scalar to this matrix."""
        if isinstance(other, Number):
            if other == 0:
                return self.copy()
            raise NotImplementedError
        if isinstance(other, np.ndarray):
            return self.toarray() + other
        if isinstance(other, BSparse):
            other = other.todia()
        if sp.issparse(other):
            other = BDIA.from_sparray(other, (self.row_sizes, self.col_sizes))

        if not isinstance(other, BDIA):
            raise TypeError("Invalid type.")

        if self.bshape != other.bshape:
            raise ValueError("Incompatible matrix shapes.")
        if np.any(self.row_sizes != other.row_sizes) or np.any(
            self.col_sizes != other.col_sizes
        ):
            raise ValueError("Incompatible block sizes.")

        if self.symmetry != other.symmetry:
            return self._desymmetrize() + other._desymmetrize()

        offsets = np.unique(np.concatenate((self.offsets, other.offsets)))
        data = []
        for offset in offsets:
            self_ind = np.nonzero(self.offsets == offset)[0]
            other_ind = np.nonzero(other.offsets == offset)[0]
            if self_ind.size == 0:
                data.append(other.data[other_ind[0]])
                continue
            if other_ind.size == 0:
                data.append(self.data[self_ind[0]])
                continue
            data.append(
                [
                    a + b
                    for a, b in zip(self.data[self_ind[0]], other.data[other_ind[0]])
                ]
            )

        return BDIA(
            offsets,
            data,
            self.bshape,
            self.dtype,
            (self.row_sizes, self.col_sizes),
            self.symmetry,
        )

    def __radd__(
        self, other: Number | BSparse | np.ndarray | sp.sparray
    ) -> "BDIA | np.ndarray":
        """Adds this matrix to another matrix or a scalar."""
        return self + other

    def __sub__(
        self, other: Number | BSparse | np.ndarray | sp.sparray
    ) -> "BDIA | np.ndarray":
        """Subtracts another matrix or a scalar from this matrix."""
        return self + (-other)

    def __rsub__(
        self, other: Number | BSparse | np.ndarray | sp.sparray
    ) -> "BDIA | np.ndarray":
        """Subtracts this matrix from another matrix or a scalar."""
        return (-self) + other

    def __mul__(
        self, other: Number | BSparse | np.ndarray | sp.sparray
    ) -> "BDIA | np.ndarray":
        """Multiplies another matrix or a scalar by this matrix."""
        if isinstance(other, Number):
            if self.symmetry == "hermitian" and np.iscomplexobj(other):
                self = self._desymmetrize()
            result = BDIA(
                self.offsets.copy(),
                [[b * other for b in bdiag] for bdiag in self.data],
                self.bshape,
                np.result_type(self, other),
                (self.row_sizes, self.col_sizes),
                self.symmetry,
            )
            return result

        if isinstance(other, np.ndarray):
            return self.toarray() * other
        if isinstance(other, BSparse):
            other = other.todia()
        if sp.issparse(other):
            other = BDIA.from_sparray(other, (self.row_sizes, self.col_sizes))

        if not isinstance(other, BDIA):
            raise TypeError("Invalid type.")

        if self.bshape != other.bshape:
            raise ValueError("Incompatible matrix shapes.")
        if np.any(self.row_sizes != other.row_sizes) or np.any(
            self.col_sizes != other.col_sizes
        ):
            raise ValueError("Incompatible block sizes.")

        if self.symmetry != other.symmetry:
            return self._desymmetrize() * other._desymmetrize()

        offsets = np.intersect1d(self.offsets, other.offsets)
        data = []
        for offset in offsets:
            self_ind = np.nonzero(self.offsets == offset)[0]
            other_ind = np.nonzero(other.offsets == offset)[0]
            data.append(
                [
                    a * b
                    for a, b in zip(self.data[self_ind[0]], other.data[other_ind[0]])
                ]
            )

        return BDIA(
            offsets,
            data,
            self.bshape,
            np.result_type(self, other),
            (self.row_sizes, self.col_sizes),
            self.symmetry,
        )

    def __rmul__(
        self, other: Number | BSparse | np.ndarray | sp.sparray
    ) -> "BDIA | np.ndarray":
        """Multiplies this matrix by another matrix or a scalar."""
        return self * other

    def __truediv__(
        self, other: Number | BSparse | np.ndarray | sp.sparray
    ) -> "BDIA | np.ndarray":
        """Divides this matrix by another matrix or a scalar."""
        if isinstance(other, Number):
            if self.symmetry == "hermitian" and np.iscomplexobj(other):
                self = self._desymmetrize()
            result = BDIA(
                self.offsets.copy(),
                [[b / other for b in bdiag] for bdiag in self.data],
                self.bshape,
                np.result_type(self, other),
                (self.row_sizes, self.col_sizes),
                self.symmetry,
            )
            return result
        if isinstance(other, BSparse) or sp.issparse(other):
            other = other.toarray()

        if not isinstance(other, np.ndarray):
            raise TypeError("Invalid type.")

        return self.toarray() / other

    def __rtruediv__(
        self, other: Number | BSparse | np.ndarray | sp.sparray
    ) -> "BDIA | np.ndarray":
        """Divides another matrix or a scalar by this matrix."""
        if isinstance(other, Number):
            return other / self.toarray()
        if isinstance(other, BSparse) or sp.issparse(other):
            other = other.toarray()

        if not isinstance(other, np.ndarray):
            raise TypeError("Invalid type.")

        return other / self.toarray()

    def __neg__(self) -> "BDIA":
        """Negates this matrix."""
        return BDIA(
            self.offsets.copy(),
            [[-b for b in bdiag] for bdiag in self.data],
            self.bshape,
            self.dtype,
            (self.row_sizes, self.col_sizes),
            self.symmetry,
        )

    @staticmethod
    def _resolve_shifts(i: int, j: int) -> tuple[int, int, int]:
        """Resolves the shifts for matrix multiplication."""
        self_shift = 0
        other_shift = 0
        data_shift = 0
        if i >= 0 and j >= 0:
            other_shift = i
        elif i < 0 and j < 0:
            self_shift = -j
        elif i + j >= 0:
            if i >= 0:
                other_shift = i + j
            else:
                data_shift = -i
        else:
            if i >= 0:
                self_shift = -(i + j)
            else:
                data_shift = j

        return self_shift, other_shift, data_shift

    def __matmul__(
        self, other: Number | BSparse | np.ndarray | sp.sparray
    ) -> "BDIA | np.ndarray":
        """Multiplies this matrix by another matrix."""
        if isinstance(other, np.ndarray):
            return self.toarray() @ other
        if isinstance(other, BSparse):
            other = other.todia()
        if sp.issparse(other):
            warn(
                "Automatically inferring block sizes from sparse matrix. "
                "This may result in unexpected behavior. Consider "
                "converting to a `BSparse` matrix."
            )
            other = BDIA.from_sparray(other, (self.col_sizes, [1] * other.shape[1]))

        if not isinstance(other, BDIA):
            raise TypeError("Invalid type.")

        if self.bshape[1] != other.bshape[0]:
            raise ValueError("Incompatible matrix shapes.")
        if np.any(self.col_sizes != other.row_sizes):
            raise ValueError("Incompatible block sizes.")

        if self.symmetry is not None or other.symmetry is not None:
            return self._desymmetrize() @ other._desymmetrize()

        offsets = sorted(
            set(
                i + j
                for i in self.offsets
                for j in other.offsets
                if -self.bshape[0] < i + j < other.bshape[1]
            )
        )

        data = []
        for offset in offsets:
            bdiag = []
            for i in range(
                min(
                    self.bshape[0] + offset,
                    other.bshape[1] - offset,
                    self.bshape[0],
                    other.bshape[1],
                )
            ):
                row_ind = i - offset if offset < 0 else i
                col_ind = i if offset < 0 else i + offset
                bdiag.append(
                    sparse.zeros(
                        (self.row_sizes[row_ind], other.col_sizes[col_ind]),
                        self.dtype,
                    )
                )
            data.append(bdiag)

        for i in self.offsets:
            self_ind = np.nonzero(self.offsets == i)[0][0]
            for j in other.offsets:
                other_ind = np.nonzero(other.offsets == j)[0][0]
                if i + j not in offsets:
                    continue
                self_shift, other_shift, data_shift = self._resolve_shifts(i, j)

                self_bdiag = self.data[self_ind][self_shift:]
                other_bdiag = other.data[other_ind][other_shift:]
                for k, (a, b) in enumerate(zip(self_bdiag, other_bdiag)):
                    data[offsets.index(i + j)][k + data_shift] += a @ b

        return BDIA(
            np.array(offsets, dtype=int),
            data,
            (self.bshape[0], other.bshape[1]),
            np.result_type(self, other),
            (self.row_sizes, other.col_sizes),
            None,
        )

    def __rmatmul__(
        self, other: Number | BSparse | np.ndarray | sp.sparray
    ) -> "BDIA | np.ndarray":
        """Multiplies another matrix by this matrix."""
        if isinstance(other, np.ndarray):
            return other @ self.toarray()
        if isinstance(other, BSparse):
            return other.todia() @ self
        if sp.issparse(other):
            warn(
                "Automatically inferring block sizes from sparse matrix. "
                "This may result in unexpected behavior. Consider "
                "converting to a `BSparse` matrix."
            )
            return (
                BDIA.from_sparray(other, ([1] * other.shape[0], self.row_sizes)) @ self
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
        for row in range(self.bshape[0]):
            row_data = []
            for offset, bdiag in zip(self.offsets, self.data):
                if 0 > row + offset or row + min(0, offset) >= len(bdiag):
                    continue
                row_data.append(bdiag[row + min(0, offset)])
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
            col_data = []
            for offset, bdiag in zip(self.offsets, self.data):
                if 0 > col - offset or col - max(0, offset) >= len(bdiag):
                    continue
                col_data.append(bdiag[col - max(0, offset)])
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
        return sum([len(bdiag) for bdiag in self.data])

    @property
    def nnz(self) -> int:
        """The number of non-zero elements in the matrix."""
        return sum(
            b.size if hasattr(b, "size") else b.nnz
            for bdiag in self.data
            for b in bdiag
        )

    @property
    def symmetry(self) -> str | None:
        """The symmetry of the matrix."""
        return self._symmetry

    @property
    def T(self) -> "BDIA":
        """The transpose of the matrix."""
        if self.symmetry == "symmetric":
            return self
        if self.symmetry == "hermitian":
            return self.conjugate()

        offsets = -self.offsets
        data = [[b.T for b in bdiag] for bdiag in self.data]

        transpose = BDIA(
            offsets,
            data,
            (self.bshape[1], self.bshape[0]),
            self.dtype,
            (self.col_sizes, self.row_sizes),
            self.symmetry,
        )
        return transpose

    @property
    def H(self) -> "BDIA":
        """The conjugate transpose of the matrix."""
        if self.symmetry == "hermitian":
            return self
        if self.symmetry == "symmetric":
            return self.conjugate()

        return self.conjugate().T

    def conjugate(self) -> "BDIA":
        """The complex conjugate of the matrix."""
        conjugate = BDIA(
            self.offsets,
            [[b.conjugate() for b in bdiag] for bdiag in self.data],
            self.bshape,
            self.dtype,
            (self.row_sizes, self.col_sizes),
            self.symmetry,
        )
        return conjugate

    def diagonal(self, offset: int = 0) -> np.ndarray:
        """Returns the diagonal of the matrix."""
        if not -self.shape[0] < offset < self.shape[1]:
            raise ValueError("Offset out of bounds.")

        if offset < 0:
            if self.symmetry == "hermitian":
                return [b.conjugate().T for b in self.diagonal(-offset)]
            if self.symmetry == "symmetric":
                return [b.T for b in self.diagonal(-offset)]

        if offset in self.offsets:
            ind = np.nonzero(self.offsets == offset)[0][0]
            return self.data[ind]

        diag = []
        rows, cols = self._get_diag_indices(offset)
        for row, col in zip(rows, cols):
            diag.append(
                sparse.zeros((self.row_sizes[row], self.col_sizes[col]), self.dtype)
            )
        return diag

    def copy(self) -> "BDIA":
        """Returns a copy of the matrix."""
        new = BDIA(
            self.offsets.copy(),
            self.data.copy(),
            self.bshape,
            self.dtype,
            (self.row_sizes, self.col_sizes),
            self.symmetry,
        )
        return new

    def astype(self, dtype: np.dtype) -> "BDIA":
        """Returns a copy of the matrix with a different data type."""
        new = BDIA(
            self.offsets.copy(),
            self.data.copy(),
            self.bshape,
            dtype,
            (self.row_sizes, self.col_sizes),
            self.symmetry,
        )
        return new

    def _get_diag_indices(self, offset: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns the indices of the k-th diagonal."""
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

        return np.array(rows, dtype=int), np.array(cols, dtype=int)

    def toarray(self) -> np.ndarray:
        """Returns a dense matrix."""
        arr = np.zeros(self.shape, dtype=self.dtype)
        row_offsets = np.cumsum(self.row_sizes) - self.row_sizes
        col_offsets = np.cumsum(self.col_sizes) - self.col_sizes

        for offset, bdiag in zip(self.offsets, self.data):
            for row, col, b in zip(*self._get_diag_indices(offset), bdiag):
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
        """Returns a COOrdinate representation of the matrix."""
        from bsparse import BCOO

        rows = []
        cols = []
        data = []
        for offset, bdiag in zip(self.offsets, self.data):
            rr, cc = self._get_diag_indices(offset)
            mask = []
            for i, b in enumerate(bdiag):
                if isinstance(b, np.ndarray) and not np.all(b == 0):
                    mask.append(i)
                elif b.nnz != 0:
                    mask.append(i)
            rows.append(rr[mask])
            cols.append(cc[mask])
            data.append([bdiag[m] for m in mask])

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = [b for bdiag in data for b in bdiag]

        return BCOO(
            rows,
            cols,
            data,
            self.bshape,
            self.dtype,
            (self.row_sizes, self.col_sizes),
            self.symmetry,
        )

    def tocsr(self) -> "BSparse":
        """Returns a Compressed Sparse Row representation of the matrix."""
        return self.tocoo().tocsr()

    def todia(self) -> "BDIA":
        """Returns a DIAgonal representation of the matrix."""
        return self

    def bapply(self, func: callable, copy: bool = False) -> "BDIA":
        """Applies a function to each matrix block."""
        if self.symmetry is not None:
            return self._desymmetrize().bapply(func, copy)
        if copy:
            self = self.copy()
        self.data = [[func(b) for b in bdiag] for bdiag in self.data]
        return self

    def save_npz(self, filename: str) -> None:
        """Saves the matrix as ``.npz`` archive."""
        np.savez_compressed(
            filename,
            format="bdia",
            offsets=self.offsets,
            data=np.asarray(self.data, dtype=object),
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
    ) -> "BDIA":
        """Creates a sparse matrix from a dense `numpy.ndarray`."""
        from bsparse import BCOO

        bcoo = BCOO.from_array(arr, sizes, symmetry=symmetry)
        return bcoo.todia()

    @classmethod
    def from_sparray(
        cls,
        mat: sp.sparray,
        sizes: tuple[ArrayLike, ArrayLike],
        symmetry: str | None = None,
    ) -> "BDIA":
        """Creates a `BDIA` matrix from a `scipy.sparse.sparray`."""
        from bsparse import BCOO

        bcoo = BCOO.from_sparray(mat, sizes, symmetry=symmetry)
        return bcoo.todia()
