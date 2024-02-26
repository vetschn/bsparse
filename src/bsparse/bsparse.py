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
    def sizes(self) -> tuple[list, list]:
        """The sizes of the rows and columns of the elements."""
        return (self.row_sizes.tolist(), self.col_sizes.tolist())

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
    def real(self) -> "BSparse":
        """The real part of the matrix."""
        return self.bapply(np.real, copy=True)

    @property
    def imag(self) -> "BSparse":
        """The imaginary part of the matrix."""
        return self.bapply(np.imag, copy=True)

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

    def asbformat(self, format: str) -> "BSparse":
        """Converts the underlying matrix blocks to a given format.

        Parameters
        ----------
        format : str
            The format to convert the matrix blocks to. Can be one of
            'coo', 'csc', 'csr', 'dia', 'dok', 'lil' or 'array'.

        Returns
        -------
        BSparse
            The matrix with the converted matrix blocks.

        """
        if format == "coo":
            return self.bapply(lambda b: b.tocoo())
        if format == "csc":
            return self.bapply(lambda b: b.tocsc())
        if format == "csr":
            return self.bapply(lambda b: b.tocsr())
        if format == "dia":
            return self.bapply(lambda b: b.todia())
        if format == "dok":
            return self.bapply(lambda b: b.todok())
        if format == "lil":
            return self.bapply(lambda b: b.tolil())
        if format == "array":
            return self.bapply(lambda b: b.toarray())
        raise ValueError("Invalid format.")

    @abstractmethod
    def bapply(self, func: callable, copy: bool = False) -> "BSparse":
        """Applies a function to each matrix block."""
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
    def from_sparray(cls, mat: sp.sparray, sizes: tuple[list, list]) -> "BSparse":
        """Creates a sparse matrix from a `scipy.sparse.sparray`."""
        ...
