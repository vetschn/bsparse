from numbers import Number

import numpy as np
import pytest

# from bsparse import sparse
from bsparse.sparse import COO, CSR, DIA
from bsparse.sparse.sparse import Sparse

# import scipy.sparse as sp


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((5, 5), id="5x5"),
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
    "key",
    [
        pytest.param(1, id="int"),
        pytest.param(slice(0, 5, 2), id="slice"),
        pytest.param((1, 2), id="(int, int)"),
        pytest.param((-1, -3), id="(-int, -int)"),
        pytest.param((slice(0, 3), slice(0, 5, 2)), id="(slice, slice)"),
        pytest.param((slice(0, 5), slice(0, 5, 2)), id="(slice, slice)"),
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
def test_getitem(
    sparse_type: Sparse,
    shape: tuple[int, int],
    symmetry: str,
    key: int | slice | tuple[int | slice, int | slice],
) -> "Number | Sparse":
    """Tests the `__getitem__` method."""
    arr = np.random.random(shape) + 1j * np.random.random(shape)

    if symmetry == "symmetric":
        arr = arr + arr.T
    if symmetry == "hermitian":
        arr = arr + arr.T.conj()

    mat = sparse_type.from_array(arr, symmetry=symmetry)

    if (
        isinstance(key, (int, slice))
        or isinstance(key, tuple)
        and isinstance(key[0], slice)
    ):
        assert np.allclose(mat[key].toarray(), arr[key])
    else:
        assert np.allclose(mat[key], arr[key])


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((5, 5), id="5x5"),
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
    "key",
    [
        pytest.param((-2, -2), id="diag"),
        pytest.param((2, 3), id="super-diag"),
        pytest.param((3, 2), id="sub-diag"),
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(0, id="zero"),
        pytest.param(1.5, id="float"),
        pytest.param(5, id="int"),
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
def test_setitem(
    sparse_type: Sparse,
    shape: tuple[int, int],
    symmetry: str,
    key: tuple[int, int],
    value: Number,
):
    """Tests the `__setitem__` method."""
    full_arr = np.random.random(shape)
    zero_arr = np.zeros(shape)

    if symmetry == "symmetric":
        full_arr = full_arr + full_arr.T
    if symmetry == "hermitian":
        full_arr = full_arr + full_arr.T.conj()

    full_mat = sparse_type.from_array(full_arr, symmetry=symmetry)
    zero_mat = sparse_type.from_array(zero_arr, symmetry=symmetry)

    full_mat[key] = value
    zero_mat[key] = value

    assert full_mat[key] == value
    assert zero_mat[key] == value


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((5, 10), id="5x10"),
        pytest.param((10, 5), id="10x5"),
        pytest.param((5, 5), id="5x5"),
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
def test_transpose(sparse_type: Sparse, shape: tuple[int, int], symmetry: str):
    """Tests the `.T` property."""
    arr = np.random.random(shape)  # + 1j * np.random.random(shape)

    if symmetry is not None and shape[0] != shape[1]:
        with pytest.raises(ValueError):
            mat = sparse_type.from_array(arr, symmetry=symmetry)

    elif symmetry is not None and shape[0] == shape[1]:
        if symmetry == "symmetric":
            arr = arr + arr.T
        if symmetry == "hermitian":
            arr = arr + arr.T.conj()
        mat = sparse_type.from_array(arr, symmetry=symmetry)
        assert np.allclose(mat.T.toarray(), arr.T)

    else:
        mat = sparse_type.from_array(arr, symmetry=symmetry)
        assert np.allclose(mat.T.toarray(), arr.T)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((5, 5), id="5x5"),
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
def test_hermitian(sparse_type: Sparse, shape: tuple[int, int], symmetry: str):
    """Tests the `.H` property."""
    arr = np.random.random(shape) + 1j * np.random.random(shape)

    if symmetry == "symmetric":
        arr = arr + arr.T
    if symmetry == "hermitian":
        arr = arr + arr.T.conj()

    mat = sparse_type.from_array(arr, symmetry=symmetry)

    assert np.allclose(mat.H.toarray(), arr.conj().T)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((5, 5), id="5x5"),
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
def test_conjugate(sparse_type: Sparse, shape: tuple[int, int], symmetry: str):
    """Tests the `.conj()` method."""
    arr = np.random.random(shape) + 1j * np.random.random(shape)

    if symmetry == "symmetric":
        arr = arr + arr.T
    if symmetry == "hermitian":
        arr = arr + arr.T.conj()

    mat = sparse_type.from_array(arr, symmetry=symmetry)

    assert np.allclose(mat.conj().toarray(), arr.conj())


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((5, 5), id="5x5"),
        pytest.param((5, 10), id="5x10"),
        pytest.param((10, 5), id="10x5"),
    ],
)
@pytest.mark.parametrize(
    "offset",
    [
        pytest.param(0, id="0"),
        pytest.param(1, id="1"),
        pytest.param(-1, id="-1"),
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
def test_diagonal(sparse_type: Sparse, shape: tuple[int, int], offset: int):
    """Tests the `.diagonal()` method."""
    arr = np.random.random(shape) + 1j * np.random.random(shape)

    mat = sparse_type.from_array(arr)

    assert np.allclose(mat.diagonal(offset=offset), np.diagonal(arr, offset=offset))


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((5, 5), id="5x5"),
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
def test_copy(sparse_type: Sparse, shape: tuple[int, int]):
    """Tests the `.copy()` method."""
    arr = np.random.random(shape) + 1j * np.random.random(shape)

    mat = sparse_type.from_array(arr)
    copied_mat = mat.copy()

    assert np.allclose(mat.toarray(), copied_mat.toarray())
    assert mat.data is not copied_mat.data


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(float, id="float"),
        pytest.param(complex, id="complex"),
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
def test_astype(sparse_type: Sparse, dtype: tuple[int, int]):
    """Tests the `.astype()` method."""
    arr = np.random.randint(10, size=(5, 5))

    mat = sparse_type.from_array(arr)
    casted_mat = mat.astype(dtype)

    assert np.allclose(mat.toarray(), casted_mat.toarray())
    assert mat.data is not casted_mat.data
    assert mat.dtype != casted_mat.dtype
    assert casted_mat.dtype == dtype
