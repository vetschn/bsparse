import numpy as np
import pytest
import scipy.sparse as sp

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
    "bsparse_type",
    [
        pytest.param(BCOO, id="BCOO"),
        pytest.param(BCSR, id="BCSR"),
        pytest.param(BDIA, id="BDIA"),
    ],
)
def test_todia(sizes: tuple[np.ndarray, np.ndarray], bsparse_type: BSparse):
    """Tests the `todia` method."""
    shape = (sizes[0].sum(), sizes[1].sum())
    arr = np.random.random(shape)
    mat = bsparse_type.from_array(arr, sizes)

    dia = mat.todia()

    assert isinstance(dia, BDIA)
    assert np.allclose(dia.toarray(), arr)


@pytest.mark.parametrize(
    "sizes",
    [
        pytest.param((np.arange(1, 6), np.arange(1, 6)), id="5x5"),
        pytest.param((np.arange(1, 11), np.arange(1, 6)), id="10x5"),
        pytest.param((np.arange(1, 6), np.arange(1, 11)), id="5x10"),
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
def test_tocsr(sizes: tuple[np.ndarray, np.ndarray], bsparse_type: BSparse):
    """Tests the `tocsr` method."""
    shape = (sizes[0].sum(), sizes[1].sum())
    arr = np.random.random(shape)
    mat = bsparse_type.from_array(arr, sizes)

    csr = mat.tocsr()

    assert isinstance(csr, BCSR)
    assert np.allclose(csr.toarray(), arr)


@pytest.mark.parametrize(
    "sizes",
    [
        pytest.param((np.arange(1, 6), np.arange(1, 6)), id="5x5"),
        pytest.param((np.arange(1, 11), np.arange(1, 6)), id="10x5"),
        pytest.param((np.arange(1, 6), np.arange(1, 11)), id="5x10"),
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
def test_tocoo(sizes: tuple[np.ndarray, np.ndarray], bsparse_type: BSparse):
    """Tests the `tocoo` method."""
    shape = (sizes[0].sum(), sizes[1].sum())
    arr = np.random.random(shape)
    mat = bsparse_type.from_array(arr, sizes)

    coo = mat.tocoo()

    assert isinstance(coo, BCOO)
    assert np.allclose(coo.toarray(), arr)


@pytest.mark.parametrize(
    "sizes",
    [
        pytest.param((np.arange(1, 6), np.arange(1, 6)), id="5x5"),
        pytest.param((np.arange(1, 11), np.arange(1, 6)), id="10x5"),
        pytest.param((np.arange(1, 6), np.arange(1, 11)), id="5x10"),
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
def test_asbformat(sizes: tuple[np.ndarray, np.ndarray], bsparse_type: BSparse):
    """Tests the `asbformat` method."""
    shape = (sizes[0].sum(), sizes[1].sum())
    spmat = (1 + 1j) * sp.random(*shape, 1.0)

    bsparse = bsparse_type.from_sparray(spmat, sizes)

    for format in ["coo", "csc", "csr", "dia", "dok", "lil"]:
        new_bsparse = bsparse.asbformat(format)
        assert new_bsparse[0, 0].format == format

    new_bsparse = bsparse.asbformat("array")
    assert isinstance(new_bsparse[0, 0], np.ndarray)


@pytest.mark.parametrize(
    "sizes",
    [
        pytest.param((np.arange(1, 6), np.arange(1, 6)), id="5x5"),
        pytest.param((np.arange(1, 11), np.arange(1, 6)), id="10x5"),
        pytest.param((np.arange(1, 6), np.arange(1, 11)), id="5x10"),
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
def test_bapply(sizes: tuple[np.ndarray, np.ndarray], bsparse_type: BSparse):
    """Tests the `tocoo` method."""
    shape = (sizes[0].sum(), sizes[1].sum())
    arr = np.random.random(shape)
    mat = bsparse_type.from_array(arr, sizes)

    for func in (np.sin, np.exp, lambda x: x * 2):
        new_mat = mat.bapply(func, copy=True)
        assert np.allclose(new_mat.toarray(), func(arr))
