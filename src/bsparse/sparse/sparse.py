from abc import ABC, abstractmethod
from numbers import Number

import numpy as np
import scipy.sparse as sp


class Sparse(ABC):
    """Base class for sparse matrices."""

    ndim = 2

    def __repr__(self) -> str:
        """Returns a string representation of the matrix."""
        s = (
            f"{self.__class__.__name__}"
            f"(shape={self.shape}, nnz={self.nnz}, dtype={self.dtype})"
        )
        return s

    def __str__(self) -> str:
        """Returns a string representation of the matrix."""
        return repr(self)

    @abstractmethod
    def __getitem__(
        self, key: int | slice | tuple[int | slice, int | slice]
    ) -> "Number | Sparse":
        """Returns a matrix element or a submatrix."""
        ...

    @abstractmethod
    def __setitem__(
        self,
        key: tuple[int, int],
        value: Number,
    ):
        """Sets a matrix element."""
        ...

    @abstractmethod
    def __add__(self, other: "Number | Sparse") -> "Sparse":
        """Adds another matrix or a scalar to this matrix."""
        ...

    def __radd__(self, other: "Number | Sparse") -> "Sparse":
        """Adds this matrix to another matrix or a scalar."""
        return self.__add__(other)

    @abstractmethod
    def __sub__(self, other: "Number | Sparse") -> "Sparse":
        """Subtracts another matrix or a scalar from this matrix."""
        ...

    @abstractmethod
    def __rsub__(self, other: "Number | Sparse") -> "Sparse":
        """Subtracts this matrix from another matrix or a scalar."""
        ...

    @abstractmethod
    def __mul__(self, other: "Number | Sparse") -> "Sparse":
        """Multiplies another matrix or a scalar by this matrix."""
        ...

    def __rmul__(self, other: "Number | Sparse") -> "Sparse":
        """Multiplies this matrix by another matrix or a scalar."""
        return self.__mul__(other)

    @abstractmethod
    def __truediv__(self, other: "Number | Sparse") -> "Sparse":
        """Divides this matrix by another matrix or a scalar."""
        ...

    @abstractmethod
    def __rtruediv__(self, other: "Number | Sparse") -> "Sparse":
        """Divides another matrix or a scalar by this matrix."""
        ...

    @abstractmethod
    def __neg__(self) -> "Sparse":
        """Negates this matrix."""
        ...

    @abstractmethod
    def __matmul__(self, other: "Sparse") -> "Sparse":
        """Multiplies this matrix by another matrix."""
        ...

    @abstractmethod
    def __rmatmul__(self, other: "Sparse") -> "Sparse":
        """Multiplies another matrix by this matrix."""
        ...

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """The data type of the matrix elements."""
        ...

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """The shape of the matrix."""
        ...

    @property
    @abstractmethod
    def nnz(self) -> int:
        """The number of stored elements in the matrix."""
        ...

    @property
    @abstractmethod
    def symmetry(self) -> str | None:
        """The symmetry of the matrix."""
        ...

    @property
    @abstractmethod
    def T(self) -> "Sparse":
        """The transpose of the matrix."""
        ...

    @property
    @abstractmethod
    def H(self) -> "Sparse":
        """The transpose of the matrix."""
        ...

    @abstractmethod
    def conj(self) -> "Sparse":
        """The complex conjugate of the matrix."""
        ...

    @abstractmethod
    def diagonal(self) -> np.ndarray:
        """Returns the diagonal of the matrix."""
        ...

    @abstractmethod
    def copy(self) -> "Sparse":
        """Returns a copy of the matrix."""
        ...

    @abstractmethod
    def toarray(self) -> np.ndarray:
        """Converts the matrix to a dense `numpy.ndarray`."""
        ...

    @abstractmethod
    def tocoo(self) -> "Sparse":
        """Converts the matrix to coordinate storage."""
        ...

    @abstractmethod
    def tocsr(self) -> "Sparse":
        """Converts the matrix to compressed row storage."""
        ...

    @abstractmethod
    def todia(self) -> "Sparse":
        """Converts the matrix to diagonal storage."""
        ...

    @abstractmethod
    def save_npz(self, filename: str) -> None:
        """Saves the matrix as ``.npz`` archive."""
        ...

    @classmethod
    @abstractmethod
    def from_array(cls, arr: np.ndarray, symmetry: str | None) -> "Sparse":
        """Creates a sparse matrix from a dense `numpy.ndarray`."""
        ...

    @classmethod
    @abstractmethod
    def from_spmatrix(cls, mat: sp.spmatrix, symmetry: str | None) -> "Sparse":
        """Creates a sparse matrix from a `scipy.sparse.spmatrix`."""
        ...
