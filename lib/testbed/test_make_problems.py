import pytest

import numpy

from hphmm.model import (make_csr_matrix_from_dense)
from .make_problems import (transpose_csr_matrix)



def make_diag_matrix(n):
    return numpy.diag(numpy.arange(1.0, n+1.0))


def make_dense_matrix(n):
    m = n * n
    return numpy.reshape(numpy.arange(m), newshape=(n, n))


def make_single_col_matrix(n):
    a = numpy.zeros(shape=(n, n), dtype=numpy.float64)
    a[:, 0] = 1.0
    return a


def make_single_row_matrix(n):
    a = numpy.zeros(shape=(n, n), dtype=numpy.float64)
    a[0, :] = 1.0
    return a


def make_zero_matrix(n):
    return numpy.zeros(shape=(n, n), dtype=numpy.float64)

@pytest.mark.parametrize("input_matrix", [
    make_zero_matrix(1),
    make_zero_matrix(2),
    make_diag_matrix(2),
    make_diag_matrix(5),
    make_dense_matrix(2),
    make_dense_matrix(3),
    make_dense_matrix(4),
    make_single_col_matrix(5),
    make_single_col_matrix(6),
    make_single_row_matrix(7),
    make_single_col_matrix(8),
])
def test_transpose_csr_matrix_transpose_is_self_inverse(input_matrix):
    a_sparse = make_csr_matrix_from_dense(input_matrix)

    a_t_sparse = transpose_csr_matrix(a_sparse)
    a_t_t_sparse = transpose_csr_matrix(a_t_sparse)


    assert numpy.all(a_sparse.indptr == a_t_t_sparse.indptr)
    assert numpy.all(a_sparse.data == a_t_t_sparse.data)
    assert numpy.all(a_sparse.cols == a_t_t_sparse.cols)
