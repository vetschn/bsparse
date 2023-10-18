from numbers import Number

import numpy as np
import pytest
import scipy.sparse as sp

from bsparse.sparse import COO, CSR, DIA, Sparse


@pytest.mark.parametrize(
    "shape, symmetry",
    [
        pytest.param((5, 5), None, id="5x5-None"),
        pytest.param((5, 5), "symmetric", id="5x5-symmetric"),
        pytest.param((5, 5), "hermitian", id="5x5-hermitian"),
        pytest.param((5, 10), None, id="5x10"),
        pytest.param((10, 5), None, id="10x5"),
    ],
)
@pytest.mark.parametrize(
    "sparse_type",
    [
        pytest.param(COO, id="COO"),
        pytest.param(CSR, id="CSR"),
        pytest.param(DIA, id="DIA"),
    ],
)
def test_add(
    sparse_type: Sparse,
    shape: tuple[int, int],
    symmetry: str,
):
    spmat = sp.random(*shape, density=0.1, dtype=float)
    if symmetry == "symmetric":
        spmat = spmat + spmat.T
    if symmetry == "hermitian":
        spmat = spmat + spmat.T.conj()
    mat = sparse_type.from_spmatrix(spmat, symmetry=symmetry)
    arr = mat.toarray()
    assert np.allclose((mat + mat).toarray(), arr + arr)
    assert np.allclose((mat + arr), arr + arr)
    assert np.allclose((spmat + mat).toarray(), arr + arr)
    assert np.allclose((mat + spmat).toarray(), arr + arr)


@pytest.mark.parametrize(
    "shape, symmetry",
    [
        pytest.param((5, 5), None, id="5x5-None"),
        pytest.param((5, 5), "symmetric", id="5x5-symmetric"),
        pytest.param((5, 5), "hermitian", id="5x5-hermitian"),
        pytest.param((5, 10), None, id="5x10"),
        pytest.param((10, 5), None, id="10x5"),
    ],
)
@pytest.mark.parametrize(
    "sparse_type",
    [
        pytest.param(COO, id="COO"),
        pytest.param(CSR, id="CSR"),
        pytest.param(DIA, id="DIA"),
    ],
)
def test_subtract(
    sparse_type: Sparse,
    shape: tuple[int, int],
    symmetry: str,
):
    spmat = sp.random(*shape, density=0.1, dtype=float)
    if symmetry == "symmetric":
        spmat = spmat + spmat.T
    if symmetry == "hermitian":
        spmat = spmat + spmat.T.conj()
    mat = sparse_type.from_spmatrix(spmat, symmetry=symmetry)
    arr = mat.toarray()
    assert np.allclose((mat - 2 * mat).toarray(), arr - 2 * arr)
    assert np.allclose((mat - 2 * arr), arr - 2 * arr)
    assert np.allclose((spmat - 2 * mat).toarray(), arr - 2 * arr)
    assert np.allclose((mat - 2 * spmat).toarray(), arr - 2 * arr)


@pytest.mark.parametrize(
    "factor",
    [
        pytest.param(2.5j, id="2j"),
    ],
)
@pytest.mark.parametrize(
    "shape, symmetry",
    [
        pytest.param((5, 5), None, id="5x5-None"),
        pytest.param((5, 5), "symmetric", id="5x5-symmetric"),
        pytest.param((5, 5), "hermitian", id="5x5-hermitian"),
        pytest.param((5, 10), None, id="5x10"),
        pytest.param((10, 5), None, id="10x5"),
    ],
)
@pytest.mark.parametrize(
    "sparse_type",
    [
        pytest.param(COO, id="COO"),
        pytest.param(CSR, id="CSR"),
        pytest.param(DIA, id="DIA"),
    ],
)
def test_multiply(
    sparse_type: Sparse,
    shape: tuple[int, int],
    symmetry: str,
    factor: Number,
):
    spmat = sp.random(*shape, density=0.1, dtype=float)
    if symmetry == "symmetric":
        spmat = spmat + spmat.T
    if symmetry == "hermitian":
        spmat = spmat + spmat.T.conj()
    mat = sparse_type.from_spmatrix(spmat, symmetry=symmetry)
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
    "shape, symmetry",
    [
        pytest.param((5, 5), None, id="5x5-None"),
        pytest.param((5, 5), "symmetric", id="5x5-symmetric"),
        pytest.param((5, 5), "hermitian", id="5x5-hermitian"),
        pytest.param((5, 10), None, id="5x10"),
        pytest.param((10, 5), None, id="10x5"),
    ],
)
@pytest.mark.parametrize(
    "sparse_type",
    [
        pytest.param(COO, id="COO"),
        pytest.param(CSR, id="CSR"),
        pytest.param(DIA, id="DIA"),
    ],
)
def test_divide(
    sparse_type: Sparse,
    shape: tuple[int, int],
    symmetry: str,
    factor: Number,
):
    spmat = sp.random(*shape, density=0.05, dtype=float)
    if symmetry == "symmetric":
        spmat = spmat + spmat.T
    if symmetry == "hermitian":
        spmat = spmat + spmat.T.conj()
    arr = spmat.toarray()
    mat = sparse_type.from_array(arr, symmetry=symmetry)
    assert np.allclose((mat / factor).toarray(), arr / factor)
    assert np.allclose(factor / mat, factor / arr, equal_nan=True)
    assert np.allclose((mat / factor / mat), arr / factor / arr, equal_nan=True)
    assert np.allclose((spmat / factor / mat), arr / factor / arr, equal_nan=True)
    assert np.allclose((mat / factor / spmat), arr / factor / arr, equal_nan=True)
    assert np.allclose((mat / factor / arr), arr / factor / arr, equal_nan=True)


@pytest.mark.parametrize(
    "shape, symmetry",
    [
        pytest.param((5, 5), None, id="5x5-None"),
        pytest.param((5, 5), "symmetric", id="5x5-symmetric"),
        pytest.param((5, 5), "hermitian", id="5x5-hermitian"),
        pytest.param((5, 10), None, id="5x10"),
        pytest.param((10, 5), None, id="10x5"),
    ],
)
@pytest.mark.parametrize(
    "sparse_type",
    [
        pytest.param(COO, id="COO"),
        pytest.param(CSR, id="CSR"),
        pytest.param(DIA, id="DIA"),
    ],
)
def test_matmul(sparse_type: Sparse, shape: tuple[int, int], symmetry: str):
    spmat = sp.random(*shape, density=0.5, dtype=float)
    if symmetry == "symmetric":
        spmat = spmat + spmat.T
    if symmetry == "hermitian":
        spmat = spmat + spmat.T.conj()
    mat = sparse_type.from_spmatrix(spmat, symmetry=symmetry)
    arr = mat.toarray()
    assert np.allclose((mat @ (2 * mat.T)).toarray(), arr @ (2 * arr.T))
    assert np.allclose((mat @ (2 * arr.T)), arr @ (2 * arr.T))
    assert np.allclose((spmat @ (2 * mat.T)).toarray(), arr @ (2 * arr.T))
    assert np.allclose((mat @ (2 * spmat.T)).toarray(), arr @ (2 * arr.T))
