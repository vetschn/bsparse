import math
from io import BytesIO

import numpy as np
import pytest
import scipy.sparse as sp

from bsparse import sparse
from bsparse.sparse import COO, CSR, DIA
from bsparse.sparse.sparse import Sparse


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((5, 5), id="5x5"),
        pytest.param((10, 5), id="10x5"),
        pytest.param((5, 10), id="5x10"),
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
    "sparse_type",
    [
        pytest.param(COO, id="COO"),
        pytest.param(CSR, id="CSR"),
        pytest.param(DIA, id="DIA"),
    ],
)
def test_from_array(
    sparse_type: Sparse,
    shape: tuple[int, int],
    symmetry: str,
):
    """Tests the `from_array` classmethod."""
    arr = np.random.random(shape) + 1j * np.random.random(shape)

    if symmetry is not None and shape[0] != shape[1]:
        with pytest.raises(ValueError):
            mat = sparse_type.from_array(arr, symmetry=symmetry)

    elif symmetry is not None and shape[0] == shape[1]:
        if symmetry == "symmetric":
            arr = arr + arr.T
        if symmetry == "hermitian":
            arr = arr + arr.conj().T
        mat = sparse_type.from_array(arr, symmetry=symmetry)
        assert np.allclose(mat.toarray(), arr)
    else:
        mat = sparse_type.from_array(arr, symmetry=symmetry)
        assert np.allclose(mat.toarray(), arr)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((5, 5), id="5x5"),
        pytest.param((10, 5), id="10x5"),
        pytest.param((5, 10), id="5x10"),
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
    "sparse_type",
    [
        pytest.param(COO, id="COO"),
        pytest.param(CSR, id="CSR"),
        pytest.param(DIA, id="DIA"),
    ],
)
def test_from_spmatrix(sparse_type: Sparse, shape: tuple[int, int], symmetry: str):
    spmat = sp.random(*shape, 0.5) + 1j * sp.random(*shape, 0.5)

    if symmetry is not None and shape[0] != shape[1]:
        with pytest.raises(ValueError):
            mat = sparse_type.from_spmatrix(spmat, symmetry=symmetry)

    elif symmetry is not None and shape[0] == shape[1]:
        if symmetry == "symmetric":
            spmat = spmat + spmat.T
        if symmetry == "hermitian":
            spmat = spmat + spmat.conj().T
        mat = sparse_type.from_spmatrix(spmat, symmetry=symmetry)
        assert np.allclose(mat.toarray(), spmat.toarray())
    else:
        mat = sparse_type.from_spmatrix(spmat, symmetry=symmetry)
        assert np.allclose(mat.toarray(), spmat.toarray())

    assert np.allclose(spmat.toarray(), spmat.toarray())


@pytest.mark.parametrize(
    "shape",
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
        pytest.param("coo", id="COO"),
        pytest.param("csr", id="CSR"),
        pytest.param("dia", id="DIA"),
    ],
)
def test_eye(shape: tuple[int, int], offset: int, dtype: np.dtype, format: str):
    """Tests the `eye` matrix creation function."""
    mat = sparse.eye(shape, offset=offset, dtype=dtype, format=format)
    assert mat.shape == shape
    assert mat.dtype == dtype
    assert mat.symmetry is None
    assert mat.__class__.__name__.lower() == format
    assert np.allclose(mat.toarray(), np.eye(*shape, k=offset, dtype=dtype))


@pytest.mark.parametrize(
    "shape",
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
        pytest.param("coo", id="COO"),
        pytest.param("csr", id="CSR"),
        pytest.param("dia", id="DIA"),
    ],
)
def test_zeros(
    shape: tuple[int, int],
    dtype: np.dtype,
    symmetry: str | None,
    format: str,
):
    """Tests the `zeros` matrix creation function."""
    if symmetry is not None and shape[0] != shape[1]:
        with pytest.raises(ValueError):
            mat = sparse.zeros(shape, dtype=dtype, symmetry=symmetry, format=format)
    else:
        mat = sparse.zeros(shape, dtype=dtype, symmetry=symmetry, format=format)
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
        pytest.param("coo", id="COO"),
        pytest.param("csr", id="CSR"),
        pytest.param("dia", id="DIA"),
    ],
)
def test_diag(num_values: int, offset: int, format: str):
    """Tests the `diag` matrix creation function."""
    values = np.random.random(num_values) + 1j * np.random.random(num_values)
    mat = sparse.diag(values, offset=offset, format=format)
    assert np.allclose(mat.diagonal(offset), values)
    assert mat.symmetry is None
    assert mat.__class__.__name__.lower() == format
    assert np.allclose(mat.toarray(), np.diag(values, k=offset))


@pytest.mark.parametrize(
    "shape",
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
        pytest.param("coo", id="COO"),
        pytest.param("csr", id="CSR"),
        pytest.param("dia", id="DIA"),
    ],
)
def test_random(shape: tuple[int, int], density: float, format: str):
    """Tests the `random` matrix creation function."""
    mat = sparse.random(shape, density=density, format=format)
    assert mat.shape == shape
    assert mat.symmetry is None
    assert mat.__class__.__name__.lower() == format
    assert math.isclose(mat.tocoo().nnz / (mat.shape[0] * mat.shape[1]), density)


@pytest.mark.parametrize(
    "symmetry",
    [
        pytest.param(None, id="None"),
        pytest.param("symmetric", id="symmetric"),
        pytest.param("hermitian", id="hermitian"),
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
def test_npz(sparse_type: Sparse, symmetry: str):
    """Tests the `save_npz` and `load_npz` matrix creation function."""
    spmat = sp.random(10, 10, 0.5) + 1j * sp.random(10, 10, 0.5)
    if symmetry == "symmetric":
        spmat = spmat + spmat.T
    if symmetry == "hermitian":
        spmat = spmat + spmat.conj().T
    mat = sparse_type.from_spmatrix(spmat, symmetry=symmetry)

    outfile = BytesIO()
    mat.save_npz(outfile)

    outfile.seek(0)

    mat = sparse.load_npz(outfile)

    assert np.allclose(mat.toarray(), spmat.toarray())
    assert isinstance(mat, sparse_type)
