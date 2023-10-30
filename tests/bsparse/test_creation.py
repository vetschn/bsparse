import math
import random
from io import BytesIO

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.typing import ArrayLike

import bsparse
from bsparse import BCOO, BCSR, BDIA
from bsparse.bsparse import BSparse


@pytest.mark.parametrize(
    "sizes",
    [
        pytest.param((np.arange(1, 6), np.arange(1, 6)), id="5x5"),
        pytest.param((np.arange(1, 11), np.arange(1, 6)), id="10x5"),
        pytest.param((np.arange(1, 6), np.arange(1, 11)), id="5x10"),
    ],
)
@pytest.mark.parametrize(
    "symmetry",
    [
        pytest.param(None, id="None"),
        pytest.param("symmetric", id="symmetric"),
        pytest.param("hermitian", id="hermitian"),
    ],
)
@pytest.mark.parametrize(
    "bsparse_type",
    [
        pytest.param(BCOO, id="BCOO"),
        pytest.param(BCSR, id="BCSR"),
        pytest.param(BDIA, id="BDIA"),
    ],
)
def test_from_array(
    bsparse_type: BSparse,
    sizes: tuple[ArrayLike, ArrayLike],
    symmetry: str,
):
    """Tests the `from_array` classmethod."""
    shape = (sizes[0].sum(), sizes[1].sum())
    arr = np.random.random(shape) + 1j * np.random.random(shape)

    if symmetry is not None and shape[0] != shape[1]:
        with pytest.raises(ValueError):
            mat = bsparse_type.from_array(arr, sizes, symmetry=symmetry)

    elif symmetry is not None and shape[0] == shape[1]:
        if symmetry == "symmetric":
            arr = arr + arr.T
        if symmetry == "hermitian":
            arr = arr + arr.conj().T
        mat = bsparse_type.from_array(arr, sizes, symmetry=symmetry)
        assert np.allclose(mat.toarray(), arr)
    else:
        mat = bsparse_type.from_array(arr, sizes, symmetry=symmetry)
        assert np.allclose(mat.toarray(), arr)


@pytest.mark.parametrize(
    "sizes",
    [
        pytest.param((np.arange(1, 6), np.arange(1, 6)), id="5x5"),
        pytest.param((np.arange(1, 11), np.arange(1, 6)), id="10x5"),
        pytest.param((np.arange(1, 6), np.arange(1, 11)), id="5x10"),
    ],
)
@pytest.mark.parametrize(
    "symmetry",
    [
        pytest.param(None, id="None"),
        pytest.param("symmetric", id="symmetric"),
        pytest.param("hermitian", id="hermitian"),
    ],
)
@pytest.mark.parametrize(
    "bsparse_type",
    [
        pytest.param(BCOO, id="BCOO"),
        pytest.param(BCSR, id="BCSR"),
        pytest.param(BDIA, id="BDIA"),
    ],
)
def test_from_sparray(
    bsparse_type: BSparse, sizes: tuple[ArrayLike, ArrayLike], symmetry: str
):
    shape = (sizes[0].sum(), sizes[1].sum())
    spmat = sp.random(*shape, 0.5) + 1j * sp.random(*shape, 0.5)

    if symmetry is not None and shape[0] != shape[1]:
        with pytest.raises(ValueError):
            bsparse = bsparse_type.from_sparray(spmat, sizes, symmetry=symmetry)

    elif symmetry is not None and shape[0] == shape[1]:
        if symmetry == "symmetric":
            spmat = spmat + spmat.T
        if symmetry == "hermitian":
            spmat = spmat + spmat.conj().T
        bsparse = bsparse_type.from_sparray(spmat, sizes, symmetry=symmetry)
        assert np.allclose(bsparse.toarray(), spmat.toarray())
    else:
        bsparse = bsparse_type.from_sparray(spmat, sizes, symmetry=symmetry)
        assert np.allclose(bsparse.toarray(), spmat.toarray())


@pytest.mark.parametrize(
    "bshape",
    [
        pytest.param((5, 5), id="5x5"),
        pytest.param((10, 5), id="10x5"),
        pytest.param((5, 10), id="5x10"),
    ],
)
@pytest.mark.parametrize(
    "offset",
    [
        pytest.param(0, id="diag"),
        pytest.param(2, id="super-diag"),
        pytest.param(-2, id="sub-diag"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(float, id="float"),
        pytest.param(complex, id="complex"),
    ],
)
@pytest.mark.parametrize(
    "format",
    [
        pytest.param("bcoo", id="BCOO"),
        pytest.param("bcsr", id="BCSR"),
        pytest.param("bdia", id="BDIA"),
    ],
)
def test_eye(bshape: tuple[int, int], offset: int, dtype: np.dtype, format: str):
    """Tests the `eye` matrix creation function."""
    mat = bsparse.eye(bshape, offset=offset, dtype=dtype, format=format)
    assert mat.bshape == bshape
    assert mat.dtype == dtype
    assert mat.symmetry is None
    assert mat.__class__.__name__.lower() == format
    assert np.allclose(mat.toarray(), np.eye(*bshape, k=offset, dtype=dtype))


@pytest.mark.parametrize(
    "bshape",
    [
        pytest.param((5, 5), id="5x5"),
        pytest.param((10, 5), id="10x5"),
        pytest.param((5, 10), id="5x10"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(float, id="float"),
        pytest.param(complex, id="complex"),
    ],
)
@pytest.mark.parametrize(
    "symmetry",
    [
        pytest.param(None, id="None"),
        pytest.param("symmetric", id="symmetric"),
        pytest.param("hermitian", id="hermitian"),
    ],
)
@pytest.mark.parametrize(
    "format",
    [
        pytest.param("bcoo", id="BCOO"),
        pytest.param("bcsr", id="BCSR"),
        pytest.param("bdia", id="BDIA"),
    ],
)
def test_zeros(
    bshape: tuple[int, int],
    dtype: np.dtype,
    symmetry: str | None,
    format: str,
):
    """Tests the `zeros` matrix creation function."""
    shape = (bshape[0], bshape[1])
    if symmetry is not None and shape[0] != shape[1]:
        with pytest.raises(ValueError):
            mat = bsparse.zeros(bshape, dtype=dtype, symmetry=symmetry, format=format)
    else:
        mat = bsparse.zeros(bshape, dtype=dtype, symmetry=symmetry, format=format)
        assert mat.shape == shape
        assert mat.dtype == dtype
        assert mat.symmetry == symmetry
        assert mat.__class__.__name__.lower() == format
        assert np.allclose(mat.toarray(), np.zeros(shape, dtype=dtype))


@pytest.mark.parametrize(
    "num_values",
    [
        pytest.param(5, id="5"),
        pytest.param(10, id="10"),
    ],
)
@pytest.mark.parametrize(
    "offset",
    [
        pytest.param(0, id="diag"),
        pytest.param(2, id="super-diag"),
        pytest.param(-2, id="sub-diag"),
    ],
)
@pytest.mark.parametrize(
    "format",
    [
        pytest.param("bcoo", id="BCOO"),
        pytest.param("bcsr", id="BCSR"),
        pytest.param("bdia", id="BDIA"),
    ],
)
def test_diag(num_values: int, offset: int, format: str):
    """Tests the `diag` matrix creation function."""
    values = [
        np.random.random((random.randint(1, 100), random.randint(1, 100)))
        for __ in range(num_values)
    ]
    mat = bsparse.diag(values, offset=offset, format=format)
    assert mat.symmetry is None
    assert mat.__class__.__name__.lower() == format
    mat.toarray()
    assert all(np.allclose(a, b) for a, b in zip(mat.diagonal(offset), values))


@pytest.mark.parametrize(
    "bshape",
    [
        pytest.param((50, 50), id="5x5"),
        pytest.param((50, 100), id="5x10"),
        pytest.param((100, 50), id="10x5"),
    ],
)
@pytest.mark.parametrize(
    "density",
    [
        pytest.param(0.1, id="diag"),
        pytest.param(0.2, id="super-diag"),
        pytest.param(0.3, id="sub-diag"),
    ],
)
@pytest.mark.parametrize(
    "format",
    [
        pytest.param("bcoo", id="BCOO"),
        pytest.param("bcsr", id="BCSR"),
        pytest.param("bdia", id="BDIA"),
    ],
)
def test_random(bshape: tuple[int, int], density: float, format: str):
    """Tests the `random` matrix creation function."""
    mat = bsparse.random(bshape, density=density, format=format)
    assert mat.bshape == bshape
    assert mat.symmetry is None
    assert mat.__class__.__name__.lower() == format
    assert math.isclose(mat.tocoo().bnnz / (mat.bshape[0] * mat.bshape[1]), density)


@pytest.mark.parametrize(
    "symmetry",
    [
        pytest.param(None, id="None"),
        pytest.param("symmetric", id="symmetric"),
        pytest.param("hermitian", id="hermitian"),
    ],
)
@pytest.mark.parametrize(
    "bsparse_type",
    [
        pytest.param(BCOO, id="BCOO"),
        pytest.param(BCSR, id="BCSR"),
        pytest.param(BDIA, id="BDIA"),
    ],
)
def test_npz(bsparse_type: BSparse, symmetry: str):
    """Tests the `save_npz` and `load_npz` matrix creation function."""
    spmat = sp.random(10, 10, 0.5) + 1j * sp.random(10, 10, 0.5)
    if symmetry == "symmetric":
        spmat = spmat + spmat.T
    if symmetry == "hermitian":
        spmat = spmat + spmat.conj().T
    mat = bsparse_type.from_sparray(spmat, sizes=([5, 5], [5, 5]), symmetry=symmetry)

    outfile = BytesIO()
    mat.save_npz(outfile)

    outfile.seek(0)

    mat = bsparse.load_npz(outfile)

    assert np.allclose(mat.toarray(), spmat.toarray())
    assert isinstance(mat, bsparse_type)
