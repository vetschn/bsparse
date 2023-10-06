from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp


class BSparse(ABC):
    """Base class for bsparse matrices."""

    ndim = 2

    def __repr__(self) -> str:
        """Returns a string representation of the matrix."""
        s = (
            f"{self.__class__.__name__}"
            f"(bshape={self.bshape}, bnnz={self.bnnz} | "
            f"shape={self.shape}, nnz={self.nnz})"
        )
        return s

    def __str__(self) -> str:
        """Returns a string representation of the matrix."""
        return repr(self)

    @abstractmethod
    def __getitem__(
        self, key: int | slice | tuple[int | slice, int | slice]
    ) -> "np.ndarray | int | float | complex | BSparse":
        """Returns a matrix element or a submatrix."""
        ...

    @abstractmethod
    def __setitem__(
        self,
        key: tuple[int | slice, int | slice],
        value: "np.ndarray | int | float | complex | BSparse",
    ):
        """Sets a matrix element or a submatrix."""
        ...

    @abstractmethod
    def __add__(self, other: "np.number | BSparse") -> "BSparse":
        """Adds another matrix or a scalar to this matrix."""
        ...

    def __radd__(self, other: "np.number | BSparse") -> "BSparse":
        """Adds this matrix to another matrix or a scalar."""
        return self.__add__(other)

    @abstractmethod
    def __sub__(self, other: "np.number | BSparse") -> "BSparse":
        """Subtracts another matrix or a scalar from this matrix."""
        ...

    @abstractmethod
    def __rsub__(self, other: "np.number | BSparse") -> "BSparse":
        """Subtracts this matrix from another matrix or a scalar."""
        ...

    @abstractmethod
    def __mul__(self, other: "np.number | BSparse") -> "BSparse":
        """Multiplies another matrix or a scalar by this matrix."""
        ...

    def __rmul__(self, other: "np.number | BSparse") -> "BSparse":
        """Multiplies this matrix by another matrix or a scalar."""
        return self.__mul__(other)

    @abstractmethod
    def __truediv__(self, other: "np.number | BSparse") -> "BSparse":
        """Divides this matrix by another matrix or a scalar."""
        ...

    @abstractmethod
    def __rtruediv__(self, other: "np.number | BSparse") -> "BSparse":
        """Divides another matrix or a scalar by this matrix."""
        ...

    @abstractmethod
    def __neg__(self) -> "BSparse":
        """Negates this matrix."""
        ...

    @abstractmethod
    def __matmul__(self, other: "BSparse") -> "BSparse":
        """Multiplies this matrix by another matrix."""
        ...

    @abstractmethod
    def __rmatmul__(self, other: "BSparse") -> "BSparse":
        """Multiplies another matrix by this matrix."""
        ...

    @property
    @abstractmethod
    def T(self) -> "BSparse":
        """The transpose of the matrix."""
        ...

    @property
    @abstractmethod
    def H(self) -> "BSparse":
        """The conjugate transpose of the matrix."""
        ...

    @property
    @abstractmethod
    def symmetry(self) -> str | None:
        """The symmetry of the matrix."""
        ...

    @property
    @abstractmethod
    def bshape(self) -> tuple[int, int]:
        """The shape of the sparse container."""
        ...

    @property
    def shape(self) -> tuple[int, int]:
        """The shape of the equivalent dense matrix."""
        return (self.row_sizes.sum(), self.col_sizes.sum())

    @property
    @abstractmethod
    def row_sizes(self) -> np.ndarray:
        """The sizes of the rows of the elements."""
        ...

    @property
    @abstractmethod
    def col_sizes(self) -> np.ndarray:
        """The sizes of the columns of the elements."""
        ...

    @property
    @abstractmethod
    def bnnz(self) -> int:
        """The number of non-zero constituents saved in the container."""
        ...

    @property
    @abstractmethod
    def nnz(self) -> int:
        """The total number of non-zero elements in the matrix."""
        ...

    @abstractmethod
    def copy(self) -> "BSparse":
        """Returns a copy of the matrix."""
        ...

    @abstractmethod
    def to_array(self) -> np.ndarray:
        """Converts the matrix to a dense `numpy.ndarray`."""
        ...

    @abstractmethod
    def to_coo(self) -> "BSparse":
        """Converts the matrix to a ``COO`` format."""
        ...

    @abstractmethod
    def to_csr(self) -> "BSparse":
        """Converts the matrix to a ``CSR`` format."""
        ...

    @abstractmethod
    def to_dia(self) -> "BSparse":
        """Converts the matrix to a ``DIA`` format."""
        ...

    @abstractmethod
    def save_npz(self, filename: str) -> None:
        """Saves the matrix to a ``.npz`` file."""
        ...

    @classmethod
    @abstractmethod
    def from_array(cls, arr: np.ndarray) -> "BSparse":
        """Creates a sparse matrix from a dense `numpy.ndarray`."""
        ...

    @classmethod
    @abstractmethod
    def from_spmatrix(cls, mat: sp.spmatrix) -> "BSparse":
        """Creates a sparse matrix from a `scipy.sparse.spmatrix`."""
        ...
