from numbers import Number

import numpy as np
import pytest
import scipy.sparse as sp

from bsparse import BCOO, BCSR, BDIA, BSparse


@pytest.mark.parametrize(
    "sizes, symmetry",
    [
        pytest.param((np.arange(1, 6), np.arange(1, 6)), None, id="5x5-None"),
        pytest.param(
            (np.arange(1, 6), np.arange(1, 6)),
            "symmetric",
            id="5x5-symmetric",
        ),
        pytest.param(
            (np.arange(1, 6), np.arange(1, 6)),
            "hermitian",
            id="5x5-hermitian",
        ),
        pytest.param((np.arange(1, 11), np.arange(1, 6)), None, id="10x5"),
        pytest.param((np.arange(1, 6), np.arange(1, 11)), None, id="5x10"),
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
def test_add(
    bsparse_type: BSparse,
    sizes: tuple[np.ndarray, np.ndarray],
    symmetry: str,
):
    shape = (sizes[0].sum(), sizes[1].sum())
    spmat = sp.random(*shape, density=0.1, dtype=float)
    if symmetry == "symmetric":
        spmat = spmat + spmat.T
    if symmetry == "hermitian":
        spmat = spmat + spmat.T.conj()
    mat = bsparse_type.from_sparray(spmat, sizes=sizes, symmetry=symmetry)
    arr = mat.toarray()
    assert np.allclose((mat + mat).toarray(), arr + arr)
    assert np.allclose((mat + arr), arr + arr)
    assert np.allclose((spmat + mat).toarray(), arr + arr)
    assert np.allclose((mat + spmat).toarray(), arr + arr)


@pytest.mark.parametrize(
    "sizes, symmetry",
    [
        pytest.param((np.arange(1, 6), np.arange(1, 6)), None, id="5x5-None"),
        pytest.param(
            (np.arange(1, 6), np.arange(1, 6)),
            "symmetric",
            id="5x5-symmetric",
        ),
        pytest.param(
            (np.arange(1, 6), np.arange(1, 6)),
            "hermitian",
            id="5x5-hermitian",
        ),
        pytest.param((np.arange(1, 11), np.arange(1, 6)), None, id="10x5"),
        pytest.param((np.arange(1, 6), np.arange(1, 11)), None, id="5x10"),
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
def test_subtract(
    bsparse_type: BSparse,
    sizes: tuple[np.ndarray, np.ndarray],
    symmetry: str,
):
    shape = (sizes[0].sum(), sizes[1].sum())
    spmat = sp.random(*shape, density=0.1, dtype=float)
    if symmetry == "symmetric":
        spmat = spmat + spmat.T
    if symmetry == "hermitian":
        spmat = spmat + spmat.T.conj()
    mat = bsparse_type.from_sparray(spmat, sizes=sizes, symmetry=symmetry)
    arr = mat.toarray()
    assert np.allclose((mat - 2 * mat).toarray(), arr - 2 * arr)
    assert np.allclose((mat - 2 * arr), arr - 2 * arr)
    assert np.allclose((spmat - 2 * mat).toarray(), arr - 2 * arr)
    assert np.allclose((mat - 2 * spmat).toarray(), arr - 2 * arr)


@pytest.mark.parametrize(
    "factor",
    [
        pytest.param(2.5j, id="2.5j"),
    ],
)
@pytest.mark.parametrize(
    "sizes, symmetry",
    [
        pytest.param((np.arange(1, 6), np.arange(1, 6)), None, id="5x5-None"),
        pytest.param(
            (np.arange(1, 6), np.arange(1, 6)),
            "symmetric",
            id="5x5-symmetric",
        ),
        pytest.param(
            (np.arange(1, 6), np.arange(1, 6)),
            "hermitian",
            id="5x5-hermitian",
        ),
        pytest.param((np.arange(1, 11), np.arange(1, 6)), None, id="10x5"),
        pytest.param((np.arange(1, 6), np.arange(1, 11)), None, id="5x10"),
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
def test_multiply(
    bsparse_type: BSparse,
    sizes: tuple[np.ndarray, np.ndarray],
    symmetry: str,
    factor: Number,
):
    shape = (sizes[0].sum(), sizes[1].sum())
    spmat = sp.random(*shape, density=0.5, dtype=float)
    if symmetry == "symmetric":
        spmat = spmat + spmat.T
    if symmetry == "hermitian":
        spmat = spmat + spmat.T.conj()
    mat = bsparse_type.from_sparray(spmat, sizes=sizes, symmetry=symmetry)
    arr = mat.toarray()
    assert np.allclose((mat * factor).toarray(), arr * factor)
    assert np.allclose((factor * mat).toarray(), arr * factor)
    assert np.allclose((mat * factor * mat).toarray(), arr * factor * arr)
    assert np.allclose((spmat * factor * mat).toarray(), arr * factor * arr)
    assert np.allclose((mat * factor * spmat).toarray(), arr * factor * arr)
    assert np.allclose((mat * factor * arr), arr * factor * arr)


@pytest.mark.parametrize(
    "factor",
    [
        pytest.param(2.5j, id="2.5j"),
    ],
)
@pytest.mark.parametrize(
    "sizes, symmetry",
    [
        pytest.param((np.arange(1, 6), np.arange(1, 6)), None, id="5x5-None"),
        pytest.param(
            (np.arange(1, 6), np.arange(1, 6)),
            "symmetric",
            id="5x5-symmetric",
        ),
        pytest.param(
            (np.arange(1, 6), np.arange(1, 6)),
            "hermitian",
            id="5x5-hermitian",
        ),
        pytest.param((np.arange(1, 11), np.arange(1, 6)), None, id="10x5"),
        pytest.param((np.arange(1, 6), np.arange(1, 11)), None, id="5x10"),
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
def test_divide(
    bsparse_type: BSparse,
    sizes: tuple[np.ndarray, np.ndarray],
    symmetry: str,
    factor: Number,
):
    shape = (sizes[0].sum(), sizes[1].sum())
    spmat = sp.random(*shape, density=0.1, dtype=float)
    if symmetry == "symmetric":
        spmat = spmat + spmat.T
    if symmetry == "hermitian":
        spmat = spmat + spmat.T.conj()

    # NOTE: SciPy Sparse is somehow broken for typecasting lil_matrix to
    # csr_matrix.
    arr = spmat.toarray()
    mat = bsparse_type.from_array(arr, sizes=sizes, symmetry=symmetry)
    assert np.allclose((mat / factor).toarray(), arr / factor)
    assert np.allclose(factor / mat, factor / arr, equal_nan=True)
    assert np.allclose((mat / factor / mat), arr / factor / arr, equal_nan=True)
    assert np.allclose((spmat / factor / mat), arr / factor / arr, equal_nan=True)
    assert np.allclose((mat / factor / spmat), arr / factor / arr, equal_nan=True)
    assert np.allclose((mat / factor / arr), arr / factor / arr, equal_nan=True)


@pytest.mark.parametrize(
    "sizes, symmetry",
    [
        pytest.param((np.arange(1, 6), np.arange(1, 6)), None, id="5x5-None"),
        pytest.param(
            (np.arange(1, 6), np.arange(1, 6)),
            "symmetric",
            id="5x5-symmetric",
        ),
        pytest.param(
            (np.arange(1, 6), np.arange(1, 6)),
            "hermitian",
            id="5x5-hermitian",
        ),
        pytest.param((np.arange(1, 11), np.arange(1, 6)), None, id="10x5"),
        pytest.param((np.arange(1, 6), np.arange(1, 11)), None, id="5x10"),
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
def test_matmul(
    bsparse_type: BSparse,
    sizes: tuple[np.ndarray, np.ndarray],
    symmetry: str,
):
    shape = (sizes[0].sum(), sizes[1].sum())
    spmat = sp.random(*shape, density=0.5, dtype=float)
    if symmetry == "symmetric":
        spmat = spmat + spmat.T
    if symmetry == "hermitian":
        spmat = spmat + spmat.T.conj()
    mat = bsparse_type.from_sparray(spmat, sizes=sizes, symmetry=symmetry)
    arr = mat.toarray()
    assert np.allclose((mat @ (2 * mat.T)).toarray(), arr @ (2 * arr.T))
    assert np.allclose((mat @ (2 * arr.T)), arr @ (2 * arr.T))
    assert np.allclose((spmat @ (2 * mat.T)).toarray(), arr @ (2 * arr.T))
    assert np.allclose((mat @ (2 * spmat.T)).toarray(), arr @ (2 * arr.T))
