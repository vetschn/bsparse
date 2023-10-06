import numpy as np
import pytest

from bsparse.sparse import COO, CSR, DIA
from bsparse.sparse.sparse import Sparse


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((5, 5), id="5x5"),
        pytest.param((5, 10), id="5x10"),
        pytest.param((10, 5), id="10x5"),
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
def test_todia(shape: tuple[int, int], sparse_type: Sparse):
    """Tests the `todia` method."""
    arr = np.random.random(shape)
    mat = sparse_type.from_array(arr)

    dia = mat.todia()

    assert isinstance(dia, DIA)
    assert np.allclose(dia.toarray(), arr)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((5, 5), id="5x5"),
        pytest.param((5, 10), id="5x10"),
        pytest.param((10, 5), id="10x5"),
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
def test_tocsr(shape: tuple[int, int], sparse_type: Sparse):
    """Tests the `tocsr` method."""
    arr = np.random.random(shape)
    mat = sparse_type.from_array(arr)

    csr = mat.tocsr()

    assert isinstance(csr, CSR)
    assert np.allclose(csr.toarray(), arr)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((5, 5), id="5x5"),
        pytest.param((5, 10), id="5x10"),
        pytest.param((10, 5), id="10x5"),
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
def test_tocoo(shape: tuple[int, int], sparse_type: Sparse):
    """Tests the `tocoo` method."""
    arr = np.random.random(shape)
    mat = sparse_type.from_array(arr)

    coo = mat.tocoo()

    assert isinstance(coo, COO)
    assert np.allclose(coo.toarray(), arr)
