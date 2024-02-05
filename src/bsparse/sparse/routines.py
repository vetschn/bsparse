import random as rnd
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from bsparse.sparse import COO, CSR, DIA
from bsparse.sparse.sparse import Sparse


def zeros(
    shape: tuple[int, int],
    dtype: np.dtype = float,
    symmetry: str | None = None,
    format: str = "coo",
) -> Sparse:
    """Create an empty sparse matrix of specified shape and dtype.

    Parameters
    ----------
    shape : tuple[int, int]
        The shape of the matrix.
    dtype : np.dtype, optional
        The data type of the matrix elements, by default float.
    symmetry : str, optional
        The symmetry of the matrix, by default None.
    format : str, optional
        The sparse format of the matrix, by default "coo".

    Returns
    -------
    Sparse
        An empty sparse matrix of specified shape and dtype.

    Examples
    --------
    >>> zeros((3, 3))
    COO(shape=(3, 3), nnz=0, dtype=float64)
    >>> zeros((3, 3), format="csr")
    CSR(shape=(3, 3), nnz=0, dtype=float64)

    """
    format = format.lower()
    if format == "coo":
        return COO([], [], [], shape, dtype, symmetry)
    if format == "csr":
        return CSR([], [], [], shape, dtype, symmetry)
    if format == "dia":
        return DIA([], [], shape, dtype, symmetry)
    raise ValueError(f"Unknown format {format}")


def eye(
    shape: tuple[int, int],
    offset: int = 0,
    dtype: np.dtype = float,
    format: str = "coo",
) -> Sparse:
    """Create a sparse identity matrix of specified shape and dtype.

    Parameters
    ----------
    shape : tuple[int, int]
        The shape of the matrix.
    offset : int, optional
        The offset of the diagonal, by default 0.
    dtype : np.dtype, optional
        The data type of the matrix elements, by default float.
    format : str, optional
        The sparse format of the matrix, by default "coo".

    Returns
    -------
    Sparse
        A sparse identity matrix of specified shape and dtype.

    Examples
    --------
    >>> eye((3, 3))
    COO(shape=(3, 3), nnz=3, dtype=float64)
    >>> eye((3, 3), format="csr")
    CSR(shape=(3, 3), nnz=3, dtype=float64)
    >>> eye((3, 3)).toarray()
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    """
    start = (-offset) * shape[1] if offset < 0 else offset

    rows = []
    cols = []
    for flat_ind in range(start, shape[0] * shape[1], shape[1] + 1):
        if flat_ind // shape[1] >= shape[1] - offset:
            break
        rows.append(flat_ind // shape[1])
        cols.append(flat_ind % shape[1])

    sparse = COO(rows, cols, np.ones(len(rows), dtype=dtype), shape, dtype)
    if format == "coo":
        return sparse
    if format == "csr":
        return sparse.tocsr()
    if format == "dia":
        return sparse.todia()
    raise ValueError(f"Unknown format {format}")


def diag(
    values: ArrayLike,
    offset: int = 0,
    shape: tuple[int, int] | None = None,
    dtype: np.dtype | None = None,
    format: str = "coo",
) -> Sparse:
    """Create a sparse diagonal matrix of specified values and offset.

    Parameters
    ----------
    values : ArrayLike
        The values of the diagonal.
    offset : int, optional
        The offset of the diagonal, by default 0.
    shape : tuple[int, int], optional
        The shape of the matrix, by default None.
    dtype : np.dtype, optional
        The data type of the matrix elements, by default None.
    format : str, optional
        The sparse format of the matrix, by default "coo".

    Returns
    -------
    Sparse
        A sparse diagonal matrix of specified values and offset.

    Examples
    --------
    >>> diag([1, 2, 3])
    COO(shape=(3, 3), nnz=3, dtype=int64)
    >>> diag([1, 2, 3], format="csr").toarray()
    array([[1, 0, 0],
           [0, 2, 0],
           [0, 0, 3]])
    >>> diag([1, 2, 3], offset=1).toarray()
    array([[0, 1, 0, 0],
           [0, 0, 2, 0],
           [0, 0, 0, 3],
           [0, 0, 0, 0]])

    """
    values = np.asarray(values)
    if shape is None:
        shape = (len(values) + abs(offset), len(values) + abs(offset))

    start = (-offset) * shape[1] if offset < 0 else offset

    rows = []
    cols = []
    for flat_ind in range(start, shape[0] * shape[1], shape[1] + 1):
        if flat_ind // shape[1] >= shape[1] - offset:
            break
        rows.append(flat_ind // shape[1])
        cols.append(flat_ind % shape[1])

    if dtype is None:
        dtype = values.dtype

    sparse = COO(rows, cols, values, shape, dtype)
    if format == "coo":
        return sparse
    if format == "csr":
        return sparse.tocsr()
    if format == "dia":
        return sparse.todia()
    raise ValueError(f"Unknown format {format}")


def random(
    shape: tuple[int, int],
    density: float = 0.1,
    dtype: np.dtype | None = None,
    format: str = "coo",
):
    """Create a sparse random matrix of specified shape and density.

    Parameters
    ----------
    shape : tuple[int, int]
        The shape of the matrix.
    density : float, optional
        The density of the matrix, by default 0.1.
    dtype : np.dtype, optional
        The data type of the matrix elements, by default None.
    format : str, optional
        The sparse format of the matrix, by default "coo".

    Returns
    -------
    Sparse
        A sparse random matrix of specified shape and density.

    Examples
    --------
    >>> random((3, 3)) # doctest: +SKIP
    COO(shape=(3, 3), nnz=1, dtype=float64)
    >>> random((3, 3), format="csr").toarray() # doctest: +SKIP
    array([[0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.04794364]])

    """
    if density < 0 or density > 1:
        raise ValueError("Density must be between 0 and 1.")

    samples = int(shape[0] * shape[1] * density)
    flat_ind = rnd.sample(range(shape[0] * shape[1]), samples)

    rows = [ind // shape[1] for ind in flat_ind]
    cols = [ind % shape[1] for ind in flat_ind]

    rng = np.random.default_rng()
    data = rng.random(samples)

    if dtype is None:
        dtype = data.dtype

    sparse = COO(rows, cols, data, shape, dtype)
    if format == "coo":
        return sparse
    if format == "csr":
        return sparse.tocsr()
    if format == "dia":
        return sparse.todia()
    raise ValueError(f"Unknown format {format}")


def load_npz(file: Path) -> Sparse:
    """Loads a sparse matrix from a `.npz` archive.

    Parameters
    ----------
    filename : str
        The name of the file.

    Returns
    -------
    Sparse
        The sparse matrix.

    """
    npz = np.load(file, allow_pickle=True)
    if "format" not in npz:
        raise ValueError("No format specified.")

    shape, dtype, symmetry = npz["shape"], npz["dtype"].item(), npz["symmetry"].item()

    if npz["format"] == "coo":
        return COO(npz["rows"], npz["cols"], npz["data"], shape, dtype, symmetry)
    if npz["format"] == "csr":
        return CSR(npz["rowptr"], npz["cols"], npz["data"], shape, dtype, symmetry)
    if npz["format"] == "dia":
        return DIA(npz["offsets"], npz["data"], shape, dtype, symmetry)
    raise ValueError(f"Unknown format {npz['format']}")
