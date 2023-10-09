from abc import abstractmethod

import numpy as np
import scipy.sparse as sp

from bsparse.sparse import Sparse


class BSparse(Sparse):
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

    @property
    @abstractmethod
    def bshape(self) -> tuple[int, int]:
        """The shape of the sparse container."""
        ...

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
    def shape(self) -> tuple[int, int]:
        """The shape of the equivalent dense matrix."""
        return (self.row_sizes.sum(), self.col_sizes.sum())

    @property
    @abstractmethod
    def bnnz(self) -> int:
        """The number of non-zero constituents saved in the container."""
        ...

    @abstractmethod
    def tocoo(self) -> "BSparse":
        """Converts the matrix to a ``COO`` format."""
        ...

    @abstractmethod
    def tocsr(self) -> "BSparse":
        """Converts the matrix to a ``CSR`` format."""
        ...

    @abstractmethod
    def todia(self) -> "BSparse":
        """Converts the matrix to a ``DIA`` format."""
        ...

    @abstractmethod
    def save_npz(self, filename: str) -> None:
        """Saves the matrix to a ``.npz`` file."""
        ...

    @classmethod
    @abstractmethod
    def from_array(cls, arr: np.ndarray, sizes: tuple[list, list]) -> "BSparse":
        """Creates a sparse matrix from a dense `numpy.ndarray`."""
        ...

    @classmethod
    @abstractmethod
    def from_spmatrix(cls, mat: sp.spmatrix, sizes: tuple[list, list]) -> "BSparse":
        """Creates a sparse matrix from a `scipy.sparse.spmatrix`."""
        ...
