import random as rnd
from pathlib import Path

import numpy as np

from bsparse import BCOO, BCSR, BDIA
from bsparse.bsparse import BSparse


def zeros(
    bshape: tuple[int, int],
    dtype: np.dtype = float,
    sizes: tuple[np.ndarray, np.ndarray] | None = None,
    symmetry: str | None = None,
    format: str = "bcoo",
) -> BSparse:
    """Create an empty sparse matrix of specified bshape and dtype.

    Parameters
    ----------
    bshape : tuple[int, int]
        The bshape of the matrix.
    dtype : np.dtype, optional
        The data type of the matrix elements, by default float.
    symmetry : str, optional
        The symmetry of the matrix, by default None.
    format : str, optional
        The sparse format of the matrix, by default "coo".

    Returns
    -------
    BSparse
        An empty sparse matrix of specified bshape and dtype.

    """
    format = format.lower()
    if format == "bcoo":
        return BCOO([], [], [], bshape, dtype, sizes, symmetry)
    if format == "bcsr":
        return BCSR([], [], [], bshape, dtype, sizes, symmetry)
    if format == "bdia":
        return BDIA([], [[]], bshape, dtype, sizes, symmetry)
    raise ValueError(f"Unknown bsparse format {format}")


def eye(
    bshape: tuple[int, int],
    offset: int = 0,
    dtype: np.dtype = float,
    format: str = "bcoo",
) -> BSparse:
    """Create a sparse identity matrix of specified bshape and dtype.

    Parameters
    ----------
    bshape : tuple[int, int]
        The bshape of the matrix.
    offset : int, optional
        The offset of the diagonal, by default 0.
    dtype : np.dtype, optional
        The data type of the matrix elements, by default float.
    format : str, optional
        The sparse format of the matrix, by default "coo".

    Returns
    -------
    BSparse
        A sparse identity matrix of specified bshape and dtype.

    """
    start = (-offset) * bshape[1] if offset < 0 else offset

    rows = []
    cols = []
    for flat_ind in range(start, bshape[0] * bshape[1], bshape[1] + 1):
        if flat_ind // bshape[1] >= bshape[1] - offset:
            break
        rows.append(flat_ind // bshape[1])
        cols.append(flat_ind % bshape[1])

    bsparse = BCOO(
        rows, cols, [np.ones((1, 1)) for __ in range(len(rows))], bshape, dtype
    )
    if format == "bcoo":
        return bsparse
    if format == "bcsr":
        return bsparse.tocsr()
    if format == "bdia":
        return bsparse.todia()
    raise ValueError(f"Unknown bsparse format {format}")


def diag(
    values: list,
    offset: int = 0,
    bshape: tuple[int, int] | None = None,
    dtype: np.dtype | None = None,
    format: str = "bcoo",
) -> BSparse:
    """Create a sparse diagonal matrix of specified values and offset.

    Parameters
    ----------
    values : list
        The values of the diagonal.
    offset : int, optional
        The offset of the diagonal, by default 0.
    bshape : tuple[int, int], optional
        The bshape of the matrix, by default None.
    dtype : np.dtype, optional
        The data type of the matrix elements, by default None.
    format : str, optional
        The sparse format of the matrix, by default "coo".

    Returns
    -------
    BSparse
        A sparse diagonal matrix of specified values and offset.

    """
    if bshape is None:
        bshape = (len(values) + abs(offset), len(values) + abs(offset))

    start = (-offset) * bshape[1] if offset < 0 else offset

    rows = []
    cols = []
    for flat_ind in range(start, bshape[0] * bshape[1], bshape[1] + 1):
        if flat_ind // bshape[1] >= bshape[1] - offset:
            break
        rows.append(flat_ind // bshape[1])
        cols.append(flat_ind % bshape[1])

    bsparse = BCOO(rows, cols, values, bshape, dtype)
    if format == "bcoo":
        return bsparse
    if format == "bcsr":
        return bsparse.tocsr()
    if format == "bdia":
        return bsparse.todia()
    raise ValueError(f"Unknown bsparse format {format}")


def random(
    bshape: tuple[int, int],
    density: float = 0.1,
    dtype: np.dtype | None = None,
    format: str = "bcoo",
) -> BSparse:
    """Create a sparse random matrix of specified shape and density.

    Parameters
    ----------
    bshape : tuple[int, int]
        The bshape of the matrix.
    density : float, optional
        The density of the matrix, by default 0.1.
    dtype : np.dtype, optional
        The data type of the matrix elements, by default None.
    format : str, optional
        The sparse format of the matrix, by default "coo".

    Returns
    -------
    BSparse
        A sparse random matrix of specified shape and density.

    """
    if density < 0 or density > 1:
        raise ValueError("Density must be between 0 and 1.")

    samples = int(bshape[0] * bshape[1] * density)
    flat_ind = rnd.sample(range(bshape[0] * bshape[1]), samples)

    rows = [ind // bshape[1] for ind in flat_ind]
    cols = [ind % bshape[1] for ind in flat_ind]

    rng = np.random.default_rng()
    data = rng.random((samples, 1, 1))

    if dtype is None:
        dtype = data.dtype

    bsparse = BCOO(rows, cols, data, bshape, dtype)
    if format == "bcoo":
        return bsparse
    if format == "bcsr":
        return bsparse.tocsr()
    if format == "bdia":
        return bsparse.todia()
    raise ValueError(f"Unknown format {format}")


def load_npz(file: Path) -> BSparse:
    """Loads a bsparse matrix from a `.npz` archive.

    Parameters
    ----------
    filename : str
        The name of the file.

    Returns
    -------
    BSparse
        The sparse matrix.

    """
    npz = np.load(file, allow_pickle=True)
    if "format" not in npz:
        raise ValueError("No format specified.")

    bshape, dtype, symmetry = npz["bshape"], npz["dtype"].item(), npz["symmetry"].item()
    sizes = npz["row_sizes"], npz["col_sizes"]

    if npz["format"] == "bcoo":
        return BCOO(
            npz["rows"],
            npz["cols"],
            npz["data"].tolist(),
            bshape,
            dtype,
            sizes,
            symmetry,
        )
    if npz["format"] == "bcsr":
        return BCSR(
            npz["rowptr"],
            npz["cols"],
            npz["data"].tolist(),
            bshape,
            dtype,
            sizes,
            symmetry,
        )
    if npz["format"] == "bdia":
        return BDIA(
            npz["offsets"],
            npz["data"].tolist(),
            bshape,
            dtype,
            sizes,
            symmetry,
        )
    raise ValueError(f"Unknown bsparse format {npz['format']}")
