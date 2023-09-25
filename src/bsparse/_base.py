from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp


class BSparse(ABC):
    """Base class for sparse matrices."""

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}"
            f"(shape={self.shape}, dtype={self.dtype}, nnz={self.nnz})"
        )
        return s

    def __str__(self) -> str:
        return repr(self)

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        pass

    @abstractmethod
    def __add__(self, other) -> "BSparse":
        pass

    def __radd__(self, other) -> "BSparse":
        return self.__add__(other)

    @abstractmethod
    def __sub__(self, other) -> "BSparse":
        pass

    @abstractmethod
    def __rsub__(self, other) -> "BSparse":
        pass

    @abstractmethod
    def __mul__(self, other) -> "BSparse":
        pass

    def __rmul__(self, other) -> "BSparse":
        return self.__mul__(other)

    @abstractmethod
    def __truediv__(self, other) -> "BSparse":
        pass

    @abstractmethod
    def __rtruediv__(self, other) -> "BSparse":
        pass

    @abstractmethod
    def __neg__(self) -> "BSparse":
        pass

    @abstractmethod
    def __matmul__(self, other) -> "BSparse":
        pass

    @abstractmethod
    def __rmatmul__(self, other) -> "BSparse":
        pass

    @property
    @abstractmethod
    def T(self) -> "BSparse":
        """The transpose of the matrix."""
        pass

    @property
    @abstractmethod
    def H(self) -> "BSparse":
        """The conjugate transpose of the matrix."""
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """The shape of the equivalent dense matrix."""
        pass

    @property
    @abstractmethod
    def bshape(self) -> tuple[int, int]:
        """The shape of the sparse container."""
        pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """The data type of the matrix."""
        pass

    @property
    @abstractmethod
    def nnz(self) -> np.dtype:
        """The number of non-zero elements in the matrix."""
        pass

    @abstractmethod
    def copy(self) -> np.ndarray:
        """Returns a copy of the matrix."""
        pass

    @abstractmethod
    def to_array(self) -> np.ndarray:
        """Converts the matrix to a dense `numpy.ndarray`."""
        pass

    @abstractmethod
    def to_coo(self) -> "BSparse":
        """Converts the matrix to a ``COO`` format."""
        pass

    @abstractmethod
    def to_csr(self) -> "BSparse":
        """Converts the matrix to a ``CSR`` format."""
        pass

    @abstractmethod
    def to_dia(self) -> "BSparse":
        """Converts the matrix to a ``DIA`` format."""
        pass

    @abstractmethod
    def save_npz(self, filename: str) -> None:
        """Saves the matrix to a ``.npz`` file."""
        pass

    @classmethod
    @abstractmethod
    def from_array(cls, arr: np.ndarray) -> "BSparse":
        """Creates a sparse matrix from a dense `numpy.ndarray`."""
        pass

    @classmethod
    @abstractmethod
    def from_spmatrix(cls, mat: sp.spmatrix) -> "BSparse":
        """Creates a sparse matrix from a `scipy.sparse.spmatrix`."""
        pass
