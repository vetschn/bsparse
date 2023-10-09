from numbers import Number

import numpy as np
import pytest

from bsparse import BCOO, BCSR, BDIA
from bsparse.bsparse import BSparse
from bsparse.sparse import Sparse


@pytest.mark.parametrize(
    "bshape",
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
    "bsparse_type",
    [
        pytest.param(BCOO, id="BCOO"),
        pytest.param(BCSR, id="BCSR"),
        pytest.param(BDIA, id="BDIA"),
    ],
)
def test_getitem(
    bsparse_type: BSparse,
    bshape: tuple[int, int],
    symmetry: str,
    key: int | slice | tuple[int | slice, int | slice],
) -> "Number | BSparse":
    """Tests the `__getitem__` method."""
    arr = np.random.random(bshape) + 1j * np.random.random(bshape)

    if symmetry == "symmetric":
        arr = arr + arr.T
    if symmetry == "hermitian":
        arr = arr + arr.T.conj()

    mat = bsparse_type.from_array(
        arr, ([1] * bshape[0], [1] * bshape[1]), symmetry=symmetry
    )

    if (
        isinstance(key, (int, slice))
        or isinstance(key, tuple)
        and isinstance(key[0], slice)
    ):
        assert np.allclose(mat[key].toarray(), arr[key])
    else:
        assert np.allclose(mat[key], arr[key])


@pytest.mark.parametrize(
    "bshape",
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
        pytest.param(np.array(0, ndmin=2), id="zero"),
        pytest.param(np.array(1.5, ndmin=2), id="float"),
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
def test_setitem(
    bsparse_type: BSparse,
    bshape: tuple[int, int],
    symmetry: str,
    key: tuple[int, int],
    value: np.ndarray,
):
    """Tests the `__setitem__` method."""
    full_arr = np.random.random(bshape)
    zero_arr = np.zeros(bshape)

    if symmetry == "symmetric":
        full_arr = full_arr + full_arr.T
    if symmetry == "hermitian":
        full_arr = full_arr + full_arr.T.conj()

    full_mat = bsparse_type.from_array(
        full_arr, ([1] * bshape[0], [1] * bshape[1]), symmetry=symmetry
    )
    zero_mat = bsparse_type.from_array(
        zero_arr, ([1] * bshape[0], [1] * bshape[1]), symmetry=symmetry
    )

    full_mat[key] = value
    zero_mat[key] = value

    if np.all(value == 0):
        assert full_mat[key].toarray() == value
        assert zero_mat[key].toarray() == value
        assert isinstance(full_mat[key], Sparse)
        assert isinstance(zero_mat[key], Sparse)
    else:
        assert full_mat[key] == value
        assert zero_mat[key] == value
        assert isinstance(full_mat[key], np.ndarray)
        assert isinstance(zero_mat[key], np.ndarray)


@pytest.mark.parametrize(
    "bshape",
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
    "bsparse_type",
    [
        pytest.param(BCOO, id="BCOO"),
        pytest.param(BCSR, id="BCSR"),
        pytest.param(BDIA, id="BDIA"),
    ],
)
def test_transpose(bsparse_type: BSparse, bshape: tuple[int, int], symmetry: str):
    """Tests the `.T` property."""
    arr = np.random.random(bshape)

    if symmetry is not None and bshape[0] != bshape[1]:
        with pytest.raises(ValueError):
            mat = bsparse_type.from_array(
                arr, ([1] * bshape[0], [1] * bshape[1]), symmetry=symmetry
            )

    elif symmetry is not None and bshape[0] == bshape[1]:
        if symmetry == "symmetric":
            arr = arr + arr.T
        if symmetry == "hermitian":
            arr = arr + arr.T.conj()
        mat = bsparse_type.from_array(
            arr, ([1] * bshape[0], [1] * bshape[1]), symmetry=symmetry
        )
        assert np.allclose(mat.T.toarray(), arr.T)

    else:
        mat = bsparse_type.from_array(
            arr, ([1] * bshape[0], [1] * bshape[1]), symmetry=symmetry
        )
        assert np.allclose(mat.T.toarray(), arr.T)


@pytest.mark.parametrize(
    "bshape",
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
    "bsparse_type",
    [
        pytest.param(BCOO, id="BCOO"),
        pytest.param(BCSR, id="BCSR"),
        pytest.param(BDIA, id="BDIA"),
    ],
)
def test_hermitian(bsparse_type: BSparse, bshape: tuple[int, int], symmetry: str):
    """Tests the `.H` property."""
    arr = np.random.random(bshape) + 1j * np.random.random(bshape)

    if symmetry == "symmetric":
        arr = arr + arr.T
    if symmetry == "hermitian":
        arr = arr + arr.T.conjugate()

    mat = bsparse_type.from_array(
        arr, ([1] * bshape[0], [1] * bshape[1]), symmetry=symmetry
    )

    assert np.allclose(mat.H.toarray(), arr.conj().T)


@pytest.mark.parametrize(
    "bshape",
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
    "bsparse_type",
    [
        pytest.param(BCOO, id="BCOO"),
        pytest.param(BCSR, id="BCSR"),
        pytest.param(BDIA, id="BDIA"),
    ],
)
def test_conjugate(bsparse_type: BSparse, bshape: tuple[int, int], symmetry: str):
    """Tests the `.conj()` method."""
    arr = np.random.random(bshape) + 1j * np.random.random(bshape)

    if symmetry == "symmetric":
        arr = arr + arr.T
    if symmetry == "hermitian":
        arr = arr + arr.T.conj()

    mat = bsparse_type.from_array(
        arr, ([1] * bshape[0], [1] * bshape[1]), symmetry=symmetry
    )

    assert np.allclose(mat.conjugate().toarray(), arr.conj())


@pytest.mark.parametrize(
    "bshape",
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
    "bsparse_type",
    [
        pytest.param(BCOO, id="BCOO"),
        pytest.param(BCSR, id="BCSR"),
        pytest.param(BDIA, id="BDIA"),
    ],
)
def test_diagonal(bsparse_type: BSparse, bshape: tuple[int, int], offset: int):
    """Tests the `.diagonal()` method."""
    arr = np.random.random(bshape) + 1j * np.random.random(bshape)

    mat = bsparse_type.from_array(arr, ([1] * bshape[0], [1] * bshape[1]))

    assert np.allclose(
        np.array(mat.diagonal(offset=offset)).flatten(), np.diagonal(arr, offset=offset)
    )


@pytest.mark.parametrize(
    "bshape",
    [
        pytest.param((5, 5), id="5x5"),
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
def test_copy(bsparse_type: BSparse, bshape: tuple[int, int]):
    """Tests the `.copy()` method."""
    arr = np.random.random(bshape) + 1j * np.random.random(bshape)

    mat = bsparse_type.from_array(arr, ([1] * bshape[0], [1] * bshape[1]))
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
    "bsparse_type",
    [
        pytest.param(BCOO, id="BCOO"),
        pytest.param(BCSR, id="BCSR"),
        pytest.param(BDIA, id="BDIA"),
    ],
)
def test_astype(bsparse_type: BSparse, dtype: tuple[int, int]):
    """Tests the `.astype()` method."""
    arr = np.random.randint(10, size=(5, 5))

    mat = bsparse_type.from_array(arr, ([1] * 5, [1] * 5))
    cast_mat = mat.astype(dtype)

    assert np.allclose(mat.toarray(), cast_mat.toarray())
    assert mat.data is not cast_mat.data
    assert mat.dtype != cast_mat.dtype
    assert cast_mat.dtype == dtype
