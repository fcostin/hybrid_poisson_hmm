cimport cython
from libc.math cimport pow, log

ctypedef double dtype_t
ctypedef int int_t
ctypedef Py_ssize_t index_t

DEF StatusOK = 0
DEF StatusInvalidInput = 1
DEF FailedToConverge = 2

cdef struct HalleyStep:
    dtype_t f
    dtype_t step

cdef struct BatchFitResult:
    int status
    size_t iters
    size_t n_issues


import numpy


@cython.cdivision(True)
cdef inline dtype_t neg_bin(const int_t k, const dtype_t a, const dtype_t b):
    # negative-binomial distribution at k with parameters a and b.

    # Express the binomial coefficient "a+k-1 choose k" in terms of the
    # Gamma function G(z):
    #                           G(a + k)
    #    (a+k-1 choose k)  =    -------
    #                           G(a) k!
    #
    # As the Gamma function satisfies G(z+1) = z G(z) we can reduce G(a+k)
    # to G(a) * prod_{i=1}^{k-1} (a-i), giving
    #
    #   (a+k-1 choose k)   =   prod_{i=0}^{k-1} (a+i) / (i+1) .

    cdef dtype_t c, p, q
    cdef int_t i

    c = 1.0
    for i in range(0, k):
        c *= (a + i) / (i + 1)
    p = b/(b+1.0)
    q = 1.0 - p
    return c * pow(p, a) * pow(q, k)


# The transition matrix tr_matrix is encoded as a Compressed Sparse Matrix.
# For each row index i, the corresponding column indices j and coefficients
# a_{i,j} are given by:
# j           in  cols[indptr[i]:indptr[i+1]
# a_{i,j}     in  data[indptr[i]:indptr[i+1]
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef const dtype_t[:, :] forward(
        const index_t[:] tr_matrix_indptr, # transition matrix
        const index_t[:] tr_matrix_cols, # transition matrix
        const dtype_t[:] tr_matrix_data, # transition matrix
        const dtype_t[:, :] s_matrix, # signal matrix
        const int_t[:] observations,
        const dtype_t[:, :] q0):

    cdef index_t n, max_k, max_p, y_t, k, w_lo, w_hi, p, np, w, i, j, t, n_obs, start, wj, iota
    cdef dtype_t alpha, beta, alpha_, beta_, z_i, inv_z_i, z, inv_z
    cdef dtype_t mixture_expected_rate, mixture_expected_log_rate, c2j
    cdef BatchFitResult result

    cdef const dtype_t[:, :] q
    cdef dtype_t[:, :] basis
    cdef dtype_t[:, :] mixture_chi
    cdef dtype_t[:, :] q_prime

    n = q0.shape[0]
    n_obs = observations.shape[0]
    max_k = s_matrix.shape[0] - 1

    # Some of the work buffers dimensions need to have capacity proportional
    # to the observed event count + 1, in order to store coefficients and
    # parameters for Gamma distribution variations generated after conditioning
    # on the observed event count.
    max_p = max(observations) + 1

    # Allocate work buffers.
    basis = numpy.zeros((n * max_p, 3), dtype=numpy.float64)
    mixture_chi = numpy.zeros(shape=(n, 2), dtype=numpy.float64)
    q_prime = numpy.zeros(shape=(n, 3), dtype=numpy.float64)

    q = q0

    for t in range(n_obs):
        k = observations[t]

        # Discrete convolution of signal with noise wrt observation k.
        # Signal matrix[k-w] is zero unless 0 <= k-w <= max_k
        # Neg-Bin(w, alpha, beta) is zero unless w >= 0
        # equivalently:
        # 0 <= w
        # - max_k+ k <= w
        # w <= k
        w_lo = max(0, -max_k + k)
        w_hi = k + 1
        p = w_hi - w_lo

        assert p <= max_p

        np = n*p

        # Construct a "basis" of Gamma distributions by taking all the
        # individual Gamma distributions we are tracking, and conditioning
        # them on this timestep's current observation.
        basis[:np, :] = 0.0
        for w in range(w_lo, w_hi):
            j = w - w_lo
            start = (n * j)
            for i in range(n):
                alpha = q[i, 1]
                beta = q[i, 2]
                alpha_ = alpha + w
                beta_ = beta + 1.0

                basis[start + i, 0] = s_matrix[k - w][i] * neg_bin(w, alpha, beta)

                # Precompute and store the expected rate and log rate of each
                # Gamma distribution:
                #
                #   E[rate] = alpha / beta
                #   E[log(rate)] = digamma(alpha) - log(beta)
                #
                # This avoids recomputing some terms that that are shared
                # across many of the mixture-sums we compute below. Note that
                # we do not need to store the alpha and beta of each
                # distribution once we have characterised the distribution by
                # these expected values.
                #
                # TODO many alphas differ by integer amounts -- could try to
                #  exploit digamma(z+1) = digamma(z) + 1/z
                basis[start + i, 1] = alpha_ / beta_
                basis[start + i, 2] = _approx_digamma_minus_log_b(alpha_, beta_)

        # i : destination state index
        for i in range(n):
            # Compute expected characteristics of the mixture of Gamma
            # distributions associated with destination state index i .
            # This is just a convex combination of the characteristics of
            # the "basis" Gamma distributions present in the the mixture.
            z_i = 0.0
            mixture_expected_rate = 0.0
            mixture_expected_log_rate = 0.0
            for w in range(w_lo, w_hi):
                start = (n * (w - w_lo))
                for iota in range(tr_matrix_indptr[i], tr_matrix_indptr[i + 1]):
                    j = tr_matrix_cols[iota]
                    wj = start + j
                    c2j = tr_matrix_data[iota] * q[j, 0] * basis[wj, 0]
                    z_i += c2j
                    mixture_expected_rate += c2j * basis[wj, 1]
                    mixture_expected_log_rate += c2j * basis[wj, 2]

            # Normalise so that mixture coefficients sum to unity
            inv_z_i = 1.0 / z_i
            mixture_chi[i, 0] = inv_z_i * mixture_expected_rate
            mixture_chi[i, 1] = inv_z_i * mixture_expected_log_rate
            q_prime[i, 0] = z_i

        result = batch_fit_from_expectations(
            mixture_chi,
            q_prime[:, 1],
            q_prime[:, 2],
        )
        assert result.status == 0, repr(result)

        # Normalise
        z = 0.0
        for i in range(n):
            z += q_prime[i, 0]
        assert z > 0.0
        inv_z = 1.0 / z
        for i in range(n):
            q_prime[i, 0] = inv_z * q_prime[i, 0]
        q = q_prime
    return q


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef BatchFitResult batch_fit_from_expectations(
        const dtype_t[:, :] chi,
        dtype_t[:] out_alpha,
        dtype_t[:] out_beta) nogil:
    """
    By the expectations of p, a convex combination of
    Gamma distributions, we mean:

    chi_0 := E_p [ rate ]
    chi_1 := E_p [ log(rate) ]
    """

    cdef index_t n_mixtures, i
    cdef dtype_t y, x
    cdef HalleyStep hs

    n_mixtures = chi.shape[0]

    cdef BatchFitResult result
    result.status = StatusOK
    result.n_issues = 0
    result.iters = 2 * n_mixtures

    # Rough compute time breakdown:
    # 37% for first halley step
    # 37% for second halley step
    # 26% for everything else

    for i in range(n_mixtures):
        y = chi[i, 1] - log(chi[i, 0])
        x = rough_inverse_f(y)
        hs = _approx_f_halley_step(x, y)
        x += hs.step
        hs = _approx_f_halley_step(x, y)
        x += hs.step
        out_alpha[i] = x
        out_beta[i] = x / chi[i, 0]
    return result


@cython.cdivision(True)
cdef inline dtype_t rough_inverse_f(const dtype_t y) nogil:
    # Approximate inverse f^-1(y) where f(x) = digamma(x) - log(x)
    # assumption: y < 0
    #
    # Explanation of this approximation:
    #
    # Let x_i range over the grid numpy.geomspace(0.001, 100.0, 3000)
    # Define y_i := f(x_i) for f(x) = digamma(x) - log(x)
    #
    # Note that we used scipy's digamma implementation to define the
    # data y_i, unrelated to the digamma approximations in this file.
    #
    # Assume the relationship between x and y can be approximated for
    # x in [0.001, 100] by:
    #
    # log(x) = a + b log(-1/y)
    #
    # This isn't motivated by theory, but empirically, by graphing
    # log(x) as a function of y and making a guess.
    #
    # Fit parameters (a, b) using least-squares regression evaluated
    # over the points y_i to approximate the target function log(x_i).
    #
    # Then x is approximated by exp(a + b log(-1/y)) = exp(a) (-y)^b
    #
    # The parameters (a, b) found from the procedure above are:
    #
    # a = -0.42338657
    # b =  0.9256158
    #
    # We can rearrange for x0 to give:  x0 approx 0.6548 * pow(-y, -0.9256)
    #
    # This gives us a relatively cheap way to initialise a guess before
    # starting Halley's method iterations when attempting to invert f.
    #
    # After switching to this initial guess for negative y, instead of
    # the naive guess used above, it reduced the mean number of Halley's
    # method iterations required to invert f on a testbed of 80,000
    # small mixture approximation problems from 3.07 to 2.00.
    return 0.6548254483045827 * pow(-y, -0.9256158)


@cython.cdivision(True)
cdef inline HalleyStep _approx_f_halley_step(dtype_t x0, dtype_t y) nogil:
    """
    Computes a step of Halley method [1] for an approximation of the function

    f(x) = f(x;y) at point x=x_0 > 0.

    where f(x) := psi(x) - log(x) - y

    x1 = x0 + step
    where
    step = - 2 f0 f1 / (2 f1^2 - f0 f2)
    """
    cdef dtype_t digamma_0_minus_log_x0 = 0.0, digamma_1 = 0.0, digamma_2 = 0.0
    cdef dtype_t f_0, f_1, f_2
    cdef dtype_t x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10
    cdef dtype_t z0, z1, z2, z3, z4, z5, z6
    cdef HalleyStep result


    # The series approximation of digamma(x) is only accurate for sufficiently
    # large x. We use the recurrence relation digamma(x) = -x^-1 + digamma(x+1)
    # to reduce the problem for small x to a problem for large x.
    # Differentiating the recurrence produces similar recurrences for the first
    # and second derivatives.

    # Minor optimisation (around -0.7% running time of a rough batch fit): this
    # used to be a "while x < 7:" loop. x must be positive, so the loop runs at
    # most 7 times. Most values of x we see are small. If we run the loop
    # exactly 7 times regardless of the value of x then we completely eliminate
    # branches at the cost of occasionally doing some unnecessary work.

    z0 = 1.0/x0
    z1 = 1.0/(x0+1.0)
    z2 = 1.0/(x0+2.0)
    z3 = 1.0/(x0+3.0)
    z4 = 1.0/(x0+4.0)
    z5 = 1.0/(x0+5.0)
    z6 = 1.0/(x0+6.0)

    digamma_0_minus_log_x0 = -z0 - z1 -z2 -z3 -z4 -z5 -z6
    digamma_1 = z0*z0 + z1*z1 + z2*z2 + z3*z3 + z4*z4 + z5*z5 + z6*z6
    digamma_2 = -2.0 * (z0*z0*z0 + z1*z1*z1 + z2*z2*z2 + z3*z3*z3 + z4*z4*z4 + z5*z5*z5 + z6*z6*z6)

    x = x0 + 7.0 - 0.5
    x_1 = 1.0 / x
    x_2 = x_1 * x_1
    x_3 = x_2 * x_1
    x_4 = x_2 * x_2
    x_5 = x_3 * x_2
    x_6 = x_4 * x_2
    x_7 = x_5 * x_2
    x_8 = x_4 * x_4
    x_9 = x_7 * x_2
    x_10 = x_8 * x_2
    # We do log(x/x0) in preference to log(x) - log(x0) to reduce log calls.
    # where log(x) is part of digamma0 but -log(x0) is not.
    digamma_0_minus_log_x0 += log(x/x0) +   (1./24.) * x_2 -   (7.0/960.0) * x_4 +   (31.0/8064.0) * x_6  - (127.0/30720.0) * x_8
    digamma_1 +=    x_1 - (2.0/24.0) * x_3 +  (28.0/960.0) * x_5 -  (186.0/8064.0) * x_7 + (1016.0/30720.0) * x_9
    digamma_2 +=   -x_2 + (6.0/24.0) * x_4 - (140.0/960.0) * x_6 + (1302.0/8064.0) * x_8 - (9144.0/30720.0) * x_10

    # combining the two log calls into one speeds up pass1 by about 4%

    f_0 = digamma_0_minus_log_x0 - y
    f_1 = digamma_1 - z0
    f_2 = digamma_2 + (z0*z0)
    result.f = f_0
    result.step = -2.0 * f_0 * f_1 / (2.0 *  f_1 * f_1 - f_0 * f_2)
    return result


@cython.cdivision(True)
cdef inline dtype_t _approx_digamma_minus_log_b(const dtype_t x0, const dtype_t b) nogil except 0.0:
    cdef dtype_t result = 0, x, xx, xx2, xx4
    result = -1.0/x0 - 1.0/(x0+1.0) -1.0/(x0+2.0) -1.0/(x0+3.0) -1.0/(x0+4.0) -1.0/(x0+5.0) -1.0/(x0+6.0)

    x = x0 + 7.0 - 1.0/2.0
    xx = 1.0/x
    xx2 = xx*xx
    xx4 = xx2*xx2
    result += log(x/b)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4
    return result


# Adapted from Mark Johnson's digamma.c
# ref: http://web.science.mq.edu.au/~mjohnson/code/digamma.c
@cython.cdivision(True)
cdef inline dtype_t _approx_digamma(const dtype_t x0) nogil except 0.0:
    cdef dtype_t result = 0, x, xx, xx2, xx4

    # This used to be a "while x < 7" loop. If most x values are small, then we need
    # to do 7 iterations anyway. This form has no branches, it is also one big
    # expression that doesn't suggest a particular eval order. Let the compiler
    # and the hardware decide.
    result = -1.0/x0 - 1.0/(x0+1.0) -1.0/(x0+2.0) -1.0/(x0+3.0) -1.0/(x0+4.0) -1.0/(x0+5.0) -1.0/(x0+6.0)

    x = x0 + 7.0 - 1.0/2.0
    xx = 1.0/x
    xx2 = xx*xx
    xx4 = xx2*xx2
    result += log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4
    return result