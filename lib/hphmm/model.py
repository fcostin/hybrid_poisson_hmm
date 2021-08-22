r"""

s' :   dest state
s  : source state
w  : natural >= 0
k  : observed event count at time t+1


sum_s sum_w c_{s',s,w,k,t}

c := A_{s',s} B_{k-w,s'} \gamma_{s,t} Neg-Bin(w ; alpha_{s,t} + w, beta_{s,t} + 1)

where \gamma_{s,t} in [0, 1] \approx p(s | y_{1:t})

so we need

Neg-Bin(k ; a, b) = (a + k - 1 choose k) (b/(b+1))^a (1/(b+1))^k

(a + k - 1 choose k) = Gamma(a+k) / (k! Gamma(a))

"""


import numpy
from scipy.special import binom, gamma
import numpy.typing
import typing

from gamma_approx import (
    fit_gamma_dist_to_gamma_mixture,
    rough_fit_batch_gamma_dists_to_gamma_mixtures,
)
from .base import BaseHMM
from .libhphmm import forward as _forward


def neg_bin(k, a, b):
    # naive. may suffer from
    # (i) slowness
    # (ii) numerical difficulties

    return (
        binom(a + k - 1, k) *
        ((b/(b+1)) ** a) *
        ((1.0/(b+1)) ** k)
    )


def ensure_sane_transition_matrix(transition_matrix):
    assert numpy.all(transition_matrix >= 0)
    assert numpy.all(numpy.isclose(numpy.sum(transition_matrix[:, :], axis=1), 1.0))
    return transition_matrix


class CSRMatrix(typing.NamedTuple):
    indptr: numpy.typing.NDArray
    cols: numpy.typing.NDArray
    data: numpy.typing.NDArray


def make_csr_matrix_from_dense(a):
    n, m = a.shape
    nonzeros = 0
    for i in range(n):
        for j in range(m):
            if a[i][j] != 0.0:
                nonzeros += 1
    indptr = numpy.zeros(shape=(n + 1, ), dtype=numpy.int64)
    cols = numpy.zeros(shape=(nonzeros, ), dtype=numpy.int64)
    data = numpy.zeros(shape=(nonzeros, ), dtype=numpy.float64)

    indptr[0] = 0
    k = 0
    for i in range(n):
        for j in range(m):
            if a[i][j] != 0.0:
                cols[k] = j
                data[k] = a[i, j]
                k += 1
        indptr[i+1] = k

    return CSRMatrix(
        indptr=indptr,
        cols=cols,
        data=data,
    )

class HybridPoissonHMM(BaseHMM):

    def __init__(self, transition_matrix, signal_matrix):
        super().__init__()
        self._transition_matrix = ensure_sane_transition_matrix(transition_matrix) # n by n state transition matrix
        self._signal_matrix = signal_matrix # K by n
        self._max_k = numpy.shape(signal_matrix)[0] - 1


    def transition_operator(self, q):
        # TODO reimplement sparse version of this.
        # Since the terms cannot be reduced until they get compressed, this
        # probably needs to be combined with the observation operator and
        # rewritten so it processes one row at a time (& hence can reduce all
        # the terms).
        n, m = numpy.shape(q)
        assert m == 3
        assert numpy.shape(self._transition_matrix) == (n, n)
        tq = numpy.empty(shape=(n, n, m), dtype=numpy.float64)
        tq[:, :, 0] = self._transition_matrix * q[:, 0]
        tq[:, :, 1:3] = q[numpy.newaxis, :, 1:3]
        return tq


    def observation_operator(self, k, tq):
        n, n2, m = numpy.shape(tq)
        assert n == n2
        assert m == 3
        assert numpy.all(tq[:, :, 0] >= 0)
        assert numpy.all(tq[:, :, 1] > 0)
        assert numpy.all(tq[:, :, 2] > 0)

        # input: indexed by state pair [s', s]
        # (c, a, b)
        # where
        # c = A_{s', s} \gamma_{s,t}

        # output: indexed by pair [s', j]
        # (c, a, b)
        # where
        # ...

        c = tq[:, :, 0]
        alpha = tq[:, :, 1]
        beta = tq[:, :, 2]

        # Discrete convolution of signal with noise wrt observation k.

        # Signal matrix[k-w] is zero unless 0 <= k-w <= max_k
        # Neg-Bin(w, alpha, beta) is zero unless w >= 0
        # equivalently:
        # 0 <= w
        # - max_k+ k <= w
        # w <= k
        w_lo = max(0, -self._max_k+ k)
        w_hi = k + 1
        p = w_hi - w_lo

        # This explodes our linear combination of n gamma distributions into
        # a linear combination of n*p gamma distributions, so we need a bunch
        # more memory to track all the gamma distribution parameters and their
        # corresponding coefficients.

        # TODO: what if we compress before we normalise? rewrite the below loop
        # to be obviously in terms of the destination state s', compute the
        # explosion of terms for that, then immedidately compress (project) it
        # back down to a single gamma distribution with some coefficent.
        # Maybe in C it could be fast.

        otq = numpy.zeros(shape=(n, n*p, m), dtype=numpy.float64)
        for w in range(w_lo, w_hi):
            j = w - w_lo
            alpha_ = alpha + w
            beta_ = beta + 1.0
            otq[:, (n*j):(n*(j+1)), 0] += c * self._signal_matrix[k-w] * neg_bin(w, alpha, beta)
            otq[:, (n*j):(n*(j+1)), 1] = alpha_
            otq[:, (n*j):(n*(j+1)), 2] = beta_
        return otq


    def normalise(self, otq):
        assert numpy.all(otq[:, :, 0] >= 0)
        z = numpy.sum(otq[:, :, 0])
        assert z > 0.0
        notq = numpy.empty(shape=otq.shape, dtype=numpy.float64)
        notq[:, :, 0] = otq[:, :, 0] / z
        notq[:, :, 1:3] = otq[:, :, 1:3]
        return notq, z


    def compression_operator_naive(self, notq):
        n, n2, m = numpy.shape(notq)
        assert m == 3

        q_prime = numpy.empty(shape=(n, 3), dtype=numpy.float64)
        for i in range(n):
            cs = notq[i, :, 0]
            alphas = notq[i, :, 1]
            betas = notq[i, :, 2]
            z_i = numpy.sum(cs)
            cs_prime = cs / z_i
            fit_result = fit_gamma_dist_to_gamma_mixture(cs_prime, alphas, betas)
            assert not fit_result['error'], repr(fit_result)
            alpha_star = fit_result['alpha']
            beta_star = fit_result['beta']
            q_prime[i, 0] = z_i
            q_prime[i, 1] = alpha_star
            q_prime[i, 2] = beta_star
        return q_prime

    def compression_operator_bulk(self, notq):
        n, n2, m = numpy.shape(notq)
        assert m == 3

        # TODO reimplement without copying and reshaping
        cs = notq[:, :, 0]
        z = numpy.sum(notq[:, :, 0], axis=1)
        cs = numpy.ravel(cs/ z[:, numpy.newaxis])
        alphas = numpy.ravel(notq[:, :, 1])
        betas = numpy.ravel(notq[:, :, 2])
        lengths = n2*numpy.ones((n, ), dtype=int)

        cab = numpy.empty((len(cs), 3), dtype=numpy.float64)
        cab[:, 0] = cs
        cab[:, 1] = alphas
        cab[:, 2] = betas

        alpha_star = numpy.zeros((n, ), dtype=numpy.float64)
        beta_star = numpy.zeros((n, ), dtype=numpy.float64)

        result = rough_fit_batch_gamma_dists_to_gamma_mixtures(lengths, cab, alpha_star, beta_star)
        assert result['status'] == 0, repr(result)

        q_prime = numpy.empty(shape=(n, 3), dtype=numpy.float64)
        q_prime[:, 0] = z
        q_prime[:, 1] = alpha_star
        q_prime[:, 2] = beta_star
        return q_prime


    def forward(self, observations, q0):
        q = q0

        for t, y_t in enumerate(observations):
            # print('[t=%r]: observed y_t = %r' % (t, y_t, ))
            # q: structured array, shape (n, ) of records (c, alpha, beta)
            # tq: structured array, shape  (n, n) of records (c', alpha, beta)
            # otq: structured array, shape (n, n*p) of records (c'', alpha, beta)
            # notq: structured array, shape (n, n*p) of records (c''', alpha, beta)
            # q_prime: structured array, shape (n, ) of records (c'''', alpha', beta')
            tq = self.transition_operator(q)
            # print('tq = %r' % (tq, ))
            otq = self.observation_operator(y_t, tq)
            # print('otq = %r' % (otq,))
            notq, _ = self.normalise(otq)
            # print('notq = %r' % (notq,))
            q_prime = self.compression_operator_bulk(notq)
            # print('q_prime = %r' % (q_prime,))
            q = q_prime
        return q


class HybridPoissonHMMv2(BaseHMM):

    def __init__(self, transition_matrix, signal_matrix):
        super().__init__()
        self._transition_matrix = ensure_sane_transition_matrix(transition_matrix) # n by n state transition matrix
        self._csr_transition_matrix = make_csr_matrix_from_dense(self._transition_matrix)
        self._signal_matrix = signal_matrix # K by n
        self._max_k = numpy.shape(signal_matrix)[0] - 1


    def forward(self, observations, q0):
        observations = numpy.asarray(observations, dtype=numpy.int32)
        return _forward(
            self._csr_transition_matrix.indptr,
            self._csr_transition_matrix.cols,
            self._csr_transition_matrix.data,
            self._signal_matrix,
            observations,
            q0,
        )